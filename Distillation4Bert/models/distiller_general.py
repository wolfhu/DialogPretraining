# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" The distiller to distil the student.
    Adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
import math
import os
import time

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset
from tqdm import tqdm

from data_utils import GroupedBatchSampler, create_lengths_groups
# from lm_seqs_dataset import LmSeqsDataset
from data_utils import LmSeqsDataset
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW

from config import DistillConfigutation
from .distill_loss import *

import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


class Distiller:
    def __init__(
        self, params: dict, dataset: TensorDataset, token_probs: torch.tensor, student: nn.Module, teacher: nn.Module,
                distill_configuration : DistillConfigutation
    ):
        logger.info("Initializing Distiller")
        self.params = params
        self.dump_path = params.output_dir
        self.multi_gpu = params.n_gpu > 1
        self.fp16 = params.fp16

        self.student = student
        self.teacher = teacher

        self.student_config = student.config
        self.vocab_size = student.config.vocab_size
        self.distill_configuration = distill_configuration

        if params.n_gpu <= 1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)

        if params.group_by_size:
            groups = create_lengths_groups(lengths=dataset.lengths, k=params.max_model_input_size)
            sampler = GroupedBatchSampler(sampler=sampler, group_ids=groups, batch_size=params.batch_size)
        else:
            sampler = BatchSampler(sampler=sampler, batch_size=params.batch_size, drop_last=False)

        if self.distill_configuration.general_distill:
            self.dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=dataset.batch_sequences)
        elif self.distill_configuration.task_distill:
            self.dataloader = DataLoader(dataset=dataset, batch_sampler=sampler)

        self.temperature = params.temperature
        assert self.temperature > 0.0

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse_loss_fct = nn.MSELoss(reduction="sum")
        self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

        self.mlm = params.mlm
        if self.distill_configuration.general_distill:
            if self.distill_configuration.mlm:
                logger.info(f"Using MLM loss for LM step.")
                self.mlm_mask_prop = params.mlm_mask_prop
                assert 0.0 <= self.mlm_mask_prop <= 1.0
                assert params.word_mask + params.word_keep + params.word_rand == 1.0
                self.pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])
                self.pred_probs = self.pred_probs.to(f"cuda:{params.local_rank}") if params.n_gpu > 0 else self.pred_probs
                self.token_probs = token_probs.to(f"cuda:{params.local_rank}") if params.n_gpu > 0 else token_probs
                if self.fp16:
                    self.pred_probs = self.pred_probs.half()
                    self.token_probs = self.token_probs.half()
            else:
                logger.info(f"Using CLM loss for LM step.")

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        self.last_loss_mlm = 0
        self.last_loss_clm = 0
        self.last_loss_mse = 0
        self.last_loss_cos = 0
        self.last_hidden_loss = 0
        self.last_attention_loss = 0
        self.last_predictlayer_ce = 0
        self.last_predictlayer_cos = 0
        self.last_predictlayer_output = 0
        self.last_predictlayer_pkd = 0
        self.last_log = 0

        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.dataloader)
        num_train_optimization_steps = (
            int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": params.weight_decay,
            },
            {
                "params": [
                    p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon, betas=(0.9, 0.98)
        )

        warmup_steps = params.warmup_steps if params.warmup_steps > 0 else math.ceil(num_train_optimization_steps * params.warmup_percentage)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
        )

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            logger.info(f"Using fp16 training: {self.params.fp16_opt_level} level")
            self.student, self.optimizer = amp.initialize(
                self.student, self.optimizer, opt_level=self.params.fp16_opt_level
            )
            self.teacher = self.teacher.half()

        if self.multi_gpu:
            self.teacher = torch.nn.DataParallel(self.teacher)
            self.student = torch.nn.DataParallel(self.student)

        if params.local_rank not in [ -1, 0]:
            if self.fp16:
                from apex.parallel import DistributedDataParallel

                logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(self.student)
                self.teacher = DistributedDataParallel(self.teacher)
            else:
                from torch.nn.parallel import DistributedDataParallel

                logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(
                    self.student,
                    device_ids=[params.local_rank],
                    output_device=params.local_rank,
                    find_unused_parameters=True,
                )

                # double check teacher model shibushi keyi fenbushi jisuan
                self.teacher = DistributedDataParallel(self.teacher,
                                                       device_ids=[params.local_rank],
                                                       output_device=params.local_rank,
                                                       find_unused_parameters=True,
                                                       )

        self.is_master = params.is_master
        if self.is_master:
            logger.info("--- Initializing Tensorboard")
            self.tensorboard = SummaryWriter(log_dir=os.path.join(self.dump_path, "log", "train"))
            self.tensorboard.add_text(tag="config/training", text_string=str(self.params), global_step=0)
            self.tensorboard.add_text(tag="config/student", text_string=str(self.student_config), global_step=0)

    def prepare_batch_mlm(self, batch):
        """
        Prepare the batch: from the token_ids and the lenghts, compute the attention mask and the masked label for MLM.

        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            mlm_labels: `torch.tensor(bs, seq_length)` - The masked languge modeling labels. There is a -100 where there is nothing to predict.
        """
        token_ids, lengths = batch
        token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0)

        lengths = torch._cast_Long(lengths)
        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]

        bs, max_seq_len = token_ids.size()
        mlm_labels = token_ids.new(token_ids.size()).copy_(token_ids)

        x_prob = self.token_probs[token_ids.flatten()]
        n_tgt = math.ceil(self.mlm_mask_prop * lengths.sum().item())
        tgt_ids = torch.multinomial(x_prob / x_prob.sum(), n_tgt, replacement=False)
        pred_mask = torch.zeros(
            bs * max_seq_len, dtype=torch.bool, device=token_ids.device
        )  # previously `dtype=torch.uint8`, cf pytorch 1.2.0 compatibility
        pred_mask[tgt_ids] = 1
        pred_mask = pred_mask.view(bs, max_seq_len)

        pred_mask[token_ids == self.params.special_tok_ids["pad_token"]] = 0

        # mask a number of words == 0 [8] (faster with fp16)
        if self.fp16:
            n1 = pred_mask.sum().item()
            if n1 > 8:
                pred_mask = pred_mask.view(-1)
                n2 = max(n1 % 8, 8 * (n1 // 8))
                if n2 != n1:
                    pred_mask[torch.nonzero(pred_mask).view(-1)[: n1 - n2]] = 0
                pred_mask = pred_mask.view(bs, max_seq_len)
                assert pred_mask.sum().item() % 8 == 0, pred_mask.sum().item()

        _token_ids_real = token_ids[pred_mask]
        _token_ids_rand = _token_ids_real.clone().random_(self.vocab_size)
        _token_ids_mask = _token_ids_real.clone().fill_(self.params.special_tok_ids["mask_token"])
        probs = torch.multinomial(self.pred_probs, len(_token_ids_real), replacement=True)
        _token_ids = (
            _token_ids_mask * (probs == 0).long()
            + _token_ids_real * (probs == 1).long()
            + _token_ids_rand * (probs == 2).long()
        )
        token_ids = token_ids.masked_scatter(pred_mask, _token_ids)

        mlm_labels[~pred_mask] = -100  # previously `mlm_labels[1-pred_mask] = -1`, cf pytorch 1.2.0 compatibility

        # sanity checks
        assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size

        return token_ids, attn_mask, mlm_labels

    def prepare_batch_clm(self, batch):
        """
        Prepare the batch: from the token_ids and the lenghts, compute the attention mask and the labels for CLM.

        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            clm_labels: `torch.tensor(bs, seq_length)` - The causal languge modeling labels. There is a -100 where there is nothing to predict.
        """
        token_ids, lengths = batch
        token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0)

        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]
        clm_labels = token_ids.new(token_ids.size()).copy_(token_ids)
        clm_labels[~attn_mask] = -100  # previously `clm_labels[1-attn_mask] = -1`, cf pytorch 1.2.0 compatibility

        # sanity checks
        assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size

        return token_ids, attn_mask, clm_labels

    def round_batch(self, x: torch.tensor, lengths: torch.tensor):
        """
        For float16 only.
        Sub-sample sentences in a batch, and add padding, so that each dimension is a multiple of 8.

        Input:
        ------
            x: `torch.tensor(bs, seq_length)` - The token ids.
            lengths: `torch.tensor(bs, seq_length)` - The lengths of each of the sequence in the batch.

        Output:
        -------
            x:  `torch.tensor(new_bs, new_seq_length)` - The updated token ids.
            lengths: `torch.tensor(new_bs, new_seq_length)` - The updated lengths.
        """
        if not self.fp16 or len(lengths) < 8:
            return x, lengths

        # number of sentences == 0 [8]
        bs1 = len(lengths)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
            lengths = lengths[idx]
            slen = lengths.max().item()
            x = x[idx, :slen]
        else:
            idx = None

        # sequence length == 0 [8]
        ml1 = x.size(1)
        if ml1 % 8 != 0:
            pad = 8 - (ml1 % 8)
            ml2 = ml1 + pad
            if self.mlm:
                pad_id = self.params.special_tok_ids["pad_token"]
            else:
                pad_id = self.params.special_tok_ids["unk_token"]
            padding_tensor = torch.zeros(bs2, pad, dtype=torch.long, device=x.device).fill_(pad_id)
            x = torch.cat([x, padding_tensor], 1)
            assert x.size() == (bs2, ml2)

        assert x.size(0) % 8 == 0
        assert x.size(1) % 8 == 0
        return x, lengths

    def prepare_batch_task(self, batch):
        input_ids, input_mask, segment_ids, label_ids = batch
        return input_ids, input_mask, segment_ids, label_ids

    def train(self):
        """
        The real training loop.
        """
        if self.is_master:
            logger.info("Starting training")
        self.last_log = time.time()
        self.student.train()
        self.teacher.eval()

        for _ in range(self.params.n_epoch):
            if self.is_master:
                logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")
            if self.multi_gpu:
                torch.distributed.barrier()

            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                if self.params.n_gpu > 0:
                    batch = tuple(t.to(f"cuda:{self.params.local_rank}") for t in batch)

                if self.distill_configuration.general_distill:
                    if self.mlm:
                        token_ids, attn_mask, lm_labels = self.prepare_batch_mlm(batch=batch)
                    else:
                        token_ids, attn_mask, lm_labels = self.prepare_batch_clm(batch=batch)

                    self.step_general(input_ids=token_ids, attention_mask=attn_mask, lm_labels=lm_labels)
                elif self.distill_configuration.task_distill:
                    input_ids, input_mask, segment_ids, label_ids = self.prepare_batch_task(batch = batch)

                    self.step_task(input_ids, input_mask, segment_ids, label_ids)

                iter_bar.update()
                iter_bar.set_postfix(
                    {"Last_loss": f"{self.last_loss:.2f}", "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}"}
                )
            iter_bar.close()

            if self.is_master:
                logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch-1}")
            self.end_epoch()

        if self.is_master:
            logger.info(f"Save very last checkpoint as `pytorch_model.bin`.")
            self.save_checkpoint(checkpoint_name=f"pytorch_model.bin")
            logger.info("Training is finished")

    def step_general(self, input_ids: torch.tensor, attention_mask: torch.tensor, lm_labels: torch.tensor):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        Input:
        ------
        input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        lm_labels: `torch.tensor(bs, seq_length)` - The language modeling labels (mlm labels for MLM and clm labels for CLM).
        """
        if self.mlm:
            s_logits, s_hidden_states, s_attentions = self.student(
                input_ids=input_ids, attention_mask=attention_mask
            )  # (bs, seq_length, voc_size)
            with torch.no_grad():
                t_logits, t_hidden_states, t_attentions = self.teacher(
                    input_ids=input_ids, attention_mask=attention_mask
                )  # (bs, seq_length, voc_size)
        else:
            s_logits, _, s_hidden_states = self.student(
                input_ids=input_ids, attention_mask=None
            )  # (bs, seq_length, voc_size)
            with torch.no_grad():
                t_logits, _, t_hidden_states = self.teacher(
                    input_ids=input_ids, attention_mask=None
                )  # (bs, seq_length, voc_size)
        assert s_logits.size() == t_logits.size()

        # https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
        # https://github.com/peterliht/knowledge-distillation-pytorch/issues/2
        if self.params.restrict_ce_to_mask:
            mask = (lm_labels > -1).unsqueeze(-1).expand_as(s_logits)  # (bs, seq_lenth, voc_size)
        else:
            mask = attention_mask.unsqueeze(-1).expand_as(s_logits)  # (bs, seq_lenth, voc_size)

        loss_ce = kl_loss(s_logits, t_logits, mask, self.temperature)
        loss = self.distill_configuration.alpha_ce * loss_ce

        if self.distill_configuration.alpha_mlm > 0.0:
            loss_mlm = self.lm_loss_fct(s_logits.view(-1, s_logits.size(-1)), lm_labels.view(-1))
            loss += self.distill_configuration.alpha_mlm * loss_mlm
        elif self.distill_configuration.alpha_clm > 0.0:
            shift_logits = s_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_clm = self.lm_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss += self.distill_configuration.alpha_clm * loss_clm

        if self.distill_configuration.alpha_mse > 0.0:
            loss_mse = mse_loss(s_logits, t_logits, mask)
            loss += self.distill_configuration.alpha_mse * loss_mse

        if self.distill_configuration.alpha_cos > 0.0:
            loss_cos = cos_loss(s_logits, t_logits, mask)
            loss += self.distill_configuration.alpha_cos*loss_cos

        block = int(len(t_attentions) / len(s_attentions))
        if self.distill_configuration.use_hidden_states:
            mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states[-1])  # (bs, seq_length, dim)
            hidden_loss = self.step_hidden(s_hidden_states, t_hidden_states, block, mask)
            loss += hidden_loss

        if self.distill_configuration.use_attentions:
            attention_loss = self.step_attention(s_attentions, t_attentions, block)
            loss += attention_loss

        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_ce = loss_ce.item()
        if self.distill_configuration.alpha_mlm > 0.0:
            self.last_loss_mlm = loss_mlm.item()
        if self.distill_configuration.alpha_clm > 0.0:
            self.last_loss_clm = loss_clm.item()
        if self.distill_configuration.alpha_mse > 0.0:
            self.last_loss_mse = loss_mse.item()
        if self.distill_configuration.alpha_cos > 0.0:
            self.last_loss_cos = loss_cos.item()
        if self.distill_configuration.use_hidden_states:
            self.last_hidden_loss = hidden_loss.item()
        if self.distill_configuration.use_attentions:
            self.last_attention_loss = attention_loss.item()

        self.optimize(loss)

        self.n_sequences_epoch += input_ids.size(0)

    def step_hidden(self, s_hidden_states: List[torch.Tensor], t_hidden_states:List[torch.Tensor], block = 0, mask = None):
        loss = 0.0
        if self.distill_configuration.use_hidden_states:
            if self.distill_configuration.hidden_match_type == 'only_top':
                loss += self.step_hidden_single_layer(s_hidden_states[-1], t_hidden_states[-1])
            elif self.distill_configuration.hidden_match_type == 'only_bottom':
                loss += self.step_hidden_single_layer(s_hidden_states[0], t_hidden_states[0])
            elif self.distill_configuration.hidden_match_type == 'top':
                block = len(t_hidden_states) - len(s_hidden_states)
                new_t_states = [t_hidden_states[idx  + block] for idx in range(len(s_hidden_states))]
                for s, t in zip(s_hidden_states, new_t_states):
                    loss += self.step_hidden_single_layer(s, t, mask)
            elif self.distill_configuration.hidden_match_type == 'bottom':
                new_t_states = [t_hidden_states[idx] for idx in range(len(s_hidden_states) -1)]
                new_t_states.append(t_hidden_states[-1])
                for s, t in zip(s_hidden_states, new_t_states):
                    loss += self.step_hidden_single_layer(s, t, mask)
            elif self.distill_configuration.hidden_match_type == 'linear':
                assert block != 0
                new_t_states = [t_hidden_states[idx * block] for idx in range(len(s_hidden_states) )]
                for s, t in zip(s_hidden_states, new_t_states):
                    loss += self.step_hidden_single_layer(s, t, mask)
        return loss
    def step_attention(self, s_hidden_states: List[torch.Tensor], t_hidden_states:List[torch.Tensor], block = 0):
        loss = 0.0
        if self.distill_configuration.use_hidden_states:
            if self.distill_configuration.hidden_match_type == 'only_top':
                loss += self.step_hidden_single_layer(s_hidden_states[-1], t_hidden_states[-1])
            elif self.distill_configuration.hidden_match_type == 'only_bottom':
                loss += self.step_hidden_single_layer(s_hidden_states[0], t_hidden_states[0])
            elif self.distill_configuration.hidden_match_type == 'top':
                block = len(t_hidden_states) - len(s_hidden_states)
                new_t_states = [t_hidden_states[idx  + block] for idx in range(len(s_hidden_states))]
                for s, t in zip(s_hidden_states, new_t_states):
                    loss += self.step_hidden_single_layer(s, t)
            elif self.distill_configuration.hidden_match_type == 'bottom':
                new_t_states = [t_hidden_states[idx] for idx in range(len(s_hidden_states) -1)]
                new_t_states.append(t_hidden_states[-1])
                for s, t in zip(s_hidden_states, new_t_states):
                    loss += self.step_hidden_single_layer(s, t)
            elif self.distill_configuration.hidden_match_type == 'linear':
                assert block != 0
                new_t_states = [t_hidden_states[idx * block + block - 1] for idx in range(len(s_hidden_states))]
                for s, t in zip(s_hidden_states, new_t_states):
                    loss += self.step_hidden_single_layer(s, t)
        return loss

    def step_hidden_single_layer(self, s_state, t_state, mask= None):
        loss = 0.0
        if self.distill_configuration.hidden_mse > 0:
            hidden_mse = mse_loss(s_state, t_state, mask)
            loss += self.distill_configuration.hidden_mse * hidden_mse
        if self.distill_configuration.hidden_cos:
            hidden_cos = cos_loss(s_state, t_state, mask)
            loss += self.distill_configuration.hidden_cos * hidden_cos
        return loss

    def step_task(self, input_ids: torch.tensor, attention_mask: torch.tensor, segment_ids: torch.tensor, label_ids: torch.tensor):
        s_logits, s_hidden_states, s_attentions = self.student(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids = segment_ids
        )  # (bs, seq_length, num_label)
        with torch.no_grad():
            t_logits, t_hidden_states, t_attentions = self.teacher(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids = segment_ids
            )  # (bs, seq_length, num_label)

        assert s_logits.size() == t_logits.size()

        loss_mse = mse_loss(s_logits, t_logits)
        loss = self.distill_configuration.alpha_mse * loss_mse

        block = int(len(t_attentions) / len(s_attentions))
        if self.distill_configuration.use_hidden_states:
            mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states[-1])  # (bs, seq_length, dim)
            hidden_loss = self.step_hidden(s_hidden_states, t_hidden_states, block, mask)
            loss += hidden_loss

        if self.distill_configuration.use_attentions:
            attention_loss = self.step_attention(s_attentions, t_attentions, block)
            loss += attention_loss

        if self.distill_configuration.task_ce > 0:
            loss_ce = kl_loss(s_logits, t_logits, mask = None, temperature= self.temperature)
            loss += self.distill_configuration.task_ce * loss_ce

        if self.distill_configuration.task_cos > 0:
            loss_cos = cos_loss(s_logits, t_logits)
            loss += self.distill_configuration.task_cos * loss_cos

        if self.distill_configuration.task_cros > 0:
            if self.student_config.num_labels == 1:
                loss_fct = nn.MSELoss()
                output_loss = loss_fct(s_logits.view(-1), label_ids.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                output_loss = loss_fct(s_logits.view(-1, self.student_config.num_labels), label_ids.view(-1))
            loss += self.distill_configuration.task_cros * output_loss

        if self.distill_configuration.task_pkd > 0.0:
            _pkd_loss = pkd_loss(s_hidden_states[-1], t_hidden_states[-1])
            loss += self.distill_configuration.task_pkd * _pkd_loss

        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_mse = loss_mse.item()

        if self.distill_configuration.alpha_cos > 0.0:
            self.last_loss_cos = loss_cos.item()
        if self.distill_configuration.use_hidden_states:
            self.last_hidden_loss = hidden_loss.item()
        if self.distill_configuration.use_attentions:
            self.last_attention_loss = attention_loss.item()
        if self.distill_configuration.task_ce > 0:
            self.last_predictlayer_ce = loss_ce.item()
        if self.distill_configuration.task_cos > 0:
            self.last_predictlayer_cos = loss_cos.item()
        if self.distill_configuration.task_cros > 0:
            self.last_predictlayer_output = output_loss.item()
        if self.distill_configuration.task_pkd > 0:
            self.last_predictlayer_pkd = _pkd_loss.item()

        self.optimize(loss)

        self.n_sequences_epoch += input_ids.size(0)


    def optimize(self, loss):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        if self.multi_gpu:
            loss = loss.mean()
        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        if self.fp16:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        self.iter()
        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            if self.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.params.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def iter(self):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        self.n_iter += 1
        self.n_total_iter += 1

        if self.n_total_iter % self.params.logging_steps == 0:
            self.log_tensorboard()
            self.last_log = time.time()
        if self.n_total_iter % self.params.save_steps == 0:
            self.save_checkpoint()

    def log_tensorboard(self):
        """
        Log into tensorboard. Only by the master process.
        """
        if not self.is_master:
            return

        for param_name, param in self.student.named_parameters():
            self.tensorboard.add_scalar(
                tag="parameter_mean/" + param_name, scalar_value=param.data.mean(), global_step=self.n_total_iter
            )
            self.tensorboard.add_scalar(
                tag="parameter_std/" + param_name, scalar_value=param.data.std(), global_step=self.n_total_iter
            )
            if param.grad is None:
                continue
            self.tensorboard.add_scalar(
                tag="grad_mean/" + param_name, scalar_value=param.grad.data.mean(), global_step=self.n_total_iter
            )
            self.tensorboard.add_scalar(
                tag="grad_std/" + param_name, scalar_value=param.grad.data.std(), global_step=self.n_total_iter
            )

        self.tensorboard.add_scalar(
            tag="losses/cum_avg_loss_epoch",
            scalar_value=self.total_loss_epoch / self.n_iter,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(tag="losses/loss", scalar_value=self.last_loss, global_step=self.n_total_iter)
        self.tensorboard.add_scalar(
            tag="losses/loss_ce", scalar_value=self.last_loss_ce, global_step=self.n_total_iter
        )
        if self.distill_configuration.alpha_mlm > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_mlm", scalar_value=self.last_loss_mlm, global_step=self.n_total_iter
            )
        if self.distill_configuration.alpha_clm > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_clm", scalar_value=self.last_loss_clm, global_step=self.n_total_iter
            )
        if self.distill_configuration.alpha_mse > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_mse", scalar_value=self.last_loss_mse, global_step=self.n_total_iter
            )
        if self.distill_configuration.alpha_cos > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_cos", scalar_value=self.last_loss_cos, global_step=self.n_total_iter
            )
        if self.distill_configuration.use_hidden_states:
            self.tensorboard.add_scalar(
                tag="losses/loss_hidden_states", scalar_value=self.last_hidden_loss, global_step=self.n_total_iter
            )
        if self.distill_configuration.use_attentions:
            self.tensorboard.add_scalar(
                tag="losses/loss_attentions", scalar_value=self.last_attention_loss, global_step=self.n_total_iter
            )
        if self.distill_configuration.task_ce > 0:
            self.tensorboard.add_scalar(
                tag="losses/loss_task_ce", scalar_value=self.last_predictlayer_ce, global_step=self.n_total_iter
            )
        if self.distill_configuration.task_cos > 0:
            self.tensorboard.add_scalar(
                tag="losses/loss_predictlayer_cos", scalar_value=self.last_predictlayer_cos, global_step=self.n_total_iter
            )
        if self.distill_configuration.task_cros > 0:
            self.tensorboard.add_scalar(
                tag="losses/loss_predictlayer_output", scalar_value=self.last_predictlayer_output, global_step=self.n_total_iter
            )
        if self.distill_configuration.task_pkd > 0:
            self.tensorboard.add_scalar(
                tag="losses/last_predictlayer_pkd", scalar_value=self.last_predictlayer_pkd, global_step=self.n_total_iter
            )

        self.tensorboard.add_scalar(
            tag="learning_rate/lr", scalar_value=self.scheduler.get_lr()[0], global_step=self.n_total_iter
        )

        self.tensorboard.add_scalar(
            tag="global/memory_usage",
            scalar_value=psutil.virtual_memory()._asdict()["used"] / 1_000_000,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(
            tag="global/speed", scalar_value=time.time() - self.last_log, global_step=self.n_total_iter
        )

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")

        if self.is_master:
            self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}.pth")
            self.tensorboard.add_scalar(
                tag="epoch/loss", scalar_value=self.total_loss_epoch / self.n_iter, global_step=self.epoch
            )

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        if not self.is_master:
            return
        mdl_to_save = self.student.module if hasattr(self.student, "module") else self.student
        mdl_to_save.config.save_pretrained(self.dump_path)
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))
