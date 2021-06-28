
"""this implemention is for dc-bert"""
"""you can find details in https://arxiv.org/abs/2002.12591"""
import logging
import os
import tqdm
import json
import random

from .utils import DataProcessor, InputExample

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, input_ids_a, input_mask_a, segment_ids_a,
                                                                     input_ids_b, input_mask_b, segment_ids_b):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_ids_a = input_ids_a
        self.input_mask_a = input_mask_a
        self.segment_ids_a = segment_ids_a
        self.input_ids_b = input_ids_b
        self.input_mask_b = input_mask_b
        self.segment_ids_b = segment_ids_b
        self.label_id = label_id

def dc_convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        max_seq_length_a = 50
        max_seq_length_b = 78
        tokens_a_truncate = tokens_a[:max_seq_length_a-2]
        tokens_b_truncate = tokens_b[:max_seq_length_b-2]
        tokens_a_truncate = tokens_a_truncate + [sep_token]
        tokens_b_truncate = tokens_b_truncate + [sep_token]
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        segment_ids_a = [sequence_a_segment_id] * len(tokens_a_truncate)
        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
            segment_ids_b = [sequence_a_segment_id] * len(tokens_b_truncate)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]

            tokens_a_truncate = tokens_a_truncate + [cls_token]
            tokens_b_truncate = tokens_b_truncate + [cls_token]
            segment_ids_a = segment_ids_a + [cls_token_segment_id]
            segment_ids_b = segment_ids_b + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

            tokens_a_truncate =  [cls_token] + tokens_a_truncate
            segment_ids_a = [cls_token_segment_id] + segment_ids_a
            tokens_b_truncate = [cls_token] + tokens_b_truncate
            segment_ids_b = [cls_token_segment_id] + segment_ids_b

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a_truncate)
        input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b_truncate)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_mask_a = [1 if mask_padding_with_zero else 0] * len(input_ids_a)
        input_mask_b = [1 if mask_padding_with_zero else 0] * len(input_ids_b)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        padding_length_a = max_seq_length_a - len(input_ids_a)
        padding_length_b = max_seq_length_b - len(input_ids_b)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids

            input_ids_a = ([pad_token] * padding_length_a) + input_ids_a
            input_mask_a = ([0 if mask_padding_with_zero else 1] * padding_length_a) + input_mask_a
            segment_ids_a = ([pad_token_segment_id] * padding_length_a) + segment_ids_a

            input_ids_b = ([pad_token] * padding_length_b) + input_ids_b
            input_mask_b = ([0 if mask_padding_with_zero else 1] * padding_length_b) + input_mask_b
            segment_ids_b = ([pad_token_segment_id] * padding_length_b) + segment_ids_b

        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            input_ids_a = input_ids_a + ([pad_token] * padding_length_a)
            input_mask_a = input_mask_a + ([0 if mask_padding_with_zero else 1] * padding_length_a)
            segment_ids_a = segment_ids_a + ([pad_token_segment_id] * padding_length_a)
            input_ids_b = input_ids_b + ([pad_token] * padding_length_b)
            input_mask_b = input_mask_b + ([0 if mask_padding_with_zero else 1] * padding_length_b)
            segment_ids_b = segment_ids_b + ([pad_token_segment_id] * padding_length_b)

        assert len(input_ids_a) == max_seq_length_a
        assert len(input_mask_a) == max_seq_length_a
        assert len(segment_ids_a) == max_seq_length_a
        assert len(input_ids_b) == max_seq_length_b
        assert len(input_mask_b) == max_seq_length_b
        assert len(segment_ids_b) == max_seq_length_b
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("input_ids a : %s" % " ".join([str(x) for x in input_ids_a]))
            logger.info("input_mask a: %s" % " ".join([str(x) for x in input_mask_a]))
            logger.info("segment_ids a: %s" % " ".join([str(x) for x in segment_ids_a]))
            logger.info("input_ids b : %s" % " ".join([str(x) for x in input_ids_b]))
            logger.info("input_mask b: %s" % " ".join([str(x) for x in input_mask_b]))
            logger.info("segment_ids b: %s" % " ".join([str(x) for x in segment_ids_b]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              input_ids_a=input_ids_a,
                              input_mask_a=input_mask_a,
                              segment_ids_a=segment_ids_a,
                              input_ids_b=input_ids_b,
                              input_mask_b=input_mask_b,
                              segment_ids_b=segment_ids_b,
                              ))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class NqParaProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                os.path.join(data_dir, "nq-train.para"), set_type="train", negtive=0.8)
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                os.path.join(data_dir, "nq-dev.para"), set_type="dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, file_name, set_type='train', negtive=0.1):
        """Creates examples for the training and dev sets."""
        examples = []

        for line in tqdm.tqdm(self._read_json(file_name)):
            try:
                label = line['label']
                if label == '0' and set_type == 'train' and random.random() > negtive:
                    continue
                text_a = line['question']
                text_b = line['document']
                if len(text_b.split()) == 0:
                    continue
                guid = "%s-%s" % (set_type, line.get('squad_id', None))
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class NqSentProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        if 'para' in data_dir:
            return self._create_examples(
                self._read_nq_json_para(os.path.join(data_dir, "train.json.sent")), "train")
        elif 'sent' in data_dir:
            return self._create_examples(
                self._read_nq_json_sent(os.path.join(data_dir, "train.json.sent")), "train")
        else:
            raise ValueError('wrong data dir!')
    def get_dev_examples(self, data_dir):
        """See base class."""
        if 'para' in data_dir:
            return self._create_examples(
                self._read_nq_json_para(os.path.join(data_dir, "dev.json.sent")), "dev")
        elif 'sent' in data_dir:
            return self._create_examples(
                self._read_nq_json_sent(os.path.join(data_dir, "dev.json.sent")), "dev")
        else:
            raise ValueError('wrong data dir!')

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in tqdm.tqdm(enumerate(lines)):
            guid = "%s-%s" % (set_type, line['id'])
            try:
                text_a = line['question']
                text_b = line['sentence']
                label = line['label']
                if set_type == 'train' and label == '0' and random.random() < 0.8:
                    continue
            except IndexError:
                continue
            # if (label == '0' and i in [11, 55, 82, 145, 192, 235, 295]) or label == '1':
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @classmethod
    def _read_nq_json_para(cls,input_file):
        with open(input_file,'r',encoding='utf-8') as fin:
            data = json.load(fin)['data']
        lines=[]
        p = 0
        n = 0
        import random
        for article in data:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                question = paragraph['qas'][0]['question']
                id_ = paragraph['qas'][0]['id']
                for sent_span, sent_label, keep_label in zip(paragraph['context_para'],
                                                            paragraph['para_labels'],
                                                            paragraph['keep_or_not']):
                    line = {}
                    line['sentence'] = context[sent_span[0]:sent_span[1]]
                    line['question'] = question
                    line['label'] = sent_label
                    line['id'] = id_
                    if keep_label:
                        lines.append(line)
                        if line['label'] == '1':
                            p += 1
                        else:
                            n += 1
        logger.info('positive {} negative {}'.format(p, n))
        return lines

    @classmethod
    def _read_nq_json_sent(cls,input_file):
        with open(input_file,'r',encoding='utf-8') as fin:
            data = json.load(fin)['data']
        lines=[]
        for article in data:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                question = paragraph['qas'][0]['question']
                id_ = paragraph['qas'][0]['id']
                for sent_span,sent_label,keep_label in zip(paragraph['context_sent'],
                                                            paragraph['sent_labels'],
                                                            paragraph['keep_or_not']):
                    line = {}
                    line['sentence'] = context[sent_span[0]:sent_span[1]]
                    line['question'] = question
                    line['label'] = sent_label
                    line['id'] = id_
                    if keep_label:
                        lines.append(line)
        return lines