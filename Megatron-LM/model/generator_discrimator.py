
import torch

from .modeling import BertConfig
from .modeling import BertForPreTraining, BertForMaskedLM
from .modeling import BertLayerNorm


def get_params_for_weight_decay_optimization(module):

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (BertLayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params


class GeneratorAndDiscrimitor(torch.nn.Module):

    def __init__(self, args):
        super(GeneratorAndDiscrimitor, self).__init__()
        if args.pretrained_bert:
            self.model = BertForPreTraining.from_pretrained(
                args.tokenizer_model_type,
                cache_dir=args.cache_dir,
                fp32_layernorm=args.fp32_layernorm,
                fp32_embedding=args.fp32_embedding,
                layernorm_epsilon=args.layernorm_epsilon)
        else:
            if args.intermediate_size is None:
                intermediate_size = 4 * args.hidden_size
            else:
                intermediate_size = args.intermediate_size
            self.config = BertConfig(
                args.tokenizer_num_tokens,
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_layers,
                num_attention_heads=args.num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=args.hidden_dropout,
                attention_probs_dropout_prob=args.attention_dropout,
                max_position_embeddings=args.max_position_embeddings,
                type_vocab_size=args.tokenizer_num_type_tokens,
                fp32_layernorm=args.fp32_layernorm,
                fp32_embedding=args.fp32_embedding,
                fp32_tokentypes=args.fp32_tokentypes,
                layernorm_epsilon=args.layernorm_epsilon,
                deep_init=args.deep_init)
            self.model = BertForPreTraining(self.config)

    def forward(self, input_tokens, token_type_ids=None,
                attention_mask=None, checkpoint_activations=False):
        return self.model(
            input_tokens, token_type_ids, attention_mask,
            checkpoint_activations=checkpoint_activations)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination=destination, prefix=prefix,
                                     keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)