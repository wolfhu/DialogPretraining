# -*- coding: utf-8 -*-

import os, csv, random, torch, torch.nn as nn, numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
# from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.optimization import BertAdam

from transformers import BertModel, BertPreTrainedModel
from transformers import AdamW, BertTokenizer
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from sklearn.metrics import f1_score
import re

import xiaoice_jieba

import logging
import sys
import pickle
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id

class Processor(object):
	def get_train_examples(self, data_dir):
		return self._create_examples(os.path.join(data_dir,'train.data.pre.txt'),'train')
	def get_test_examples(self, data_dir):
		return self._create_examples(os.path.join(data_dir,'test.data.pre.txt'),'test')
	def get_dev_examples(self, data_dir):
		return self._create_examples(os.path.join(data_dir,'test.data.pre.txt'),'dev')
	def get_labels(self):
		return ['0','1']
	def _create_examples(self, data_path, set_type):
		examples = []
		with open(data_path, 'r', encoding='utf-8') as f:
			for i,line in enumerate(f):

				# if i > 2:
				# 	break
				text_a, text_b,label = line.strip().split('\t')
				text = text_a + "[SEP]" + text_b
				if label != '0':
					label = '1'
				guid = "{0}-{1}-{2}".format(set_type,label,i)
				examples.append(
					InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
		# random.shuffle(examples)
		return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
								 output_mode = 'classification'):
	label_map = {label:i for i,label in enumerate(label_list)}
	features = []
	for ex_index,example in enumerate(examples):
		if ex_index % 10000 == 0:
			print("Writing example %d of %d" % (ex_index, len(examples)))

		tokens_a = tokenizer.tokenize(example.text_a)
		tokens_b = None
		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)
			_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
		else:
			if len(tokens_a) > max_seq_length - 2:
				tokens_a = tokens_a[:(max_seq_length - 2)]

		tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
		segment_ids = [0] * len(tokens)

		if tokens_b:
			tokens += tokens_b + ["[SEP]"]
			segment_ids += [1] * (len(tokens_b) + 1)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1] * len(input_ids)
		seq_length = len(input_ids)

		padding = [0] * (max_seq_length - len(input_ids))
		input_ids += padding
		input_mask += padding
		segment_ids += padding

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length

		if output_mode == "classification":
			label_id = label_map[example.label]
		elif output_mode == "regression":
			label_id = float(example.label)
		else:
			raise KeyError(output_mode)

		if ex_index < 1:
			logger.info("*** Example ***")
			# logger.info("guid: %s" % (example.guid))
			# logger.info("tokens: %s" % " ".join(
			#     [str(x) for x in tokens]))
			logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			logger.info(
				"segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
			logger.info("label: {}".format(example.label))
			logger.info("label_id: {}".format(label_id))

		features.append(
			InputFeatures(input_ids=input_ids,
						  input_mask=input_mask,
						  segment_ids=segment_ids,
						  label_id=label_id,
						  seq_length=seq_length))

	return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
class BertClassification(BertPreTrainedModel):
	def __init__(self, config, num_labels=2):
		super(BertClassification,self).__init__(config)
		self.num_labels = num_labels
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, num_labels)
		self.init_weights()
		# self.apply(self.init_weights())
	def forward(self, input_ids, input_mask, token_type_ids, label_ids):
		_,pooled_output = self.bert(input_ids,token_type_ids,input_mask)
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)
		if label_ids is not None:
			loss_fct = CrossEntropyLoss()
			return loss_fct(logits.view(-1,self.num_labels),label_ids.view(-1))
		return logits

class BertTextCNN(BertPreTrainedModel):
	def __init__(self, config, hidden_size=128, num_labels=2):
		super(BertTextCNN,self).__init__(config)
		self.num_labels = num_labels
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.conv1 = nn.Conv2d(1,hidden_size,(3,config.hidden_size))
		self.conv2 = nn.Conv2d(1,hidden_size,(4,config.hidden_size))
		self.conv3 = nn.Conv2d(1,hidden_size,(5,config.hidden_size))
		self.classifier = nn.Linear(hidden_size*3,num_labels)
		self.apply(self.init_bert_weights)
	def forward(self, input_ids, input_mask, label_ids):
		sequence_output,_ = self.bert(input_ids,None,input_mask,output_all_encoded_layers=False)
		out = self.dropout(sequence_output).unsqueeze(1)
		c1 = torch.relu(self.conv1(out).squeeze(3))
		p1 = F.max_pool1d(c1,c1.size(2)).squeeze(2)
		c2 = torch.relu(self.conv2(out).squeeze(3))
		p2 = F.max_pool1d(c2,c2.size(2)).squeeze(2)
		c3 = torch.relu(self.conv3(out).squeeze(3))
		p3 = F.max_pool1d(c3,c3.size(2)).squeeze(2)
		pool = self.dropout(torch.cat((p1,p2,p3),1))
		logits = self.classifier(pool)
		if label_ids is not None:
			loss_fct = CrossEntropyLoss()
			return loss_fct(logits.view(-1,self.num_labels),label_ids.view(-1))
		return logits


class BertForSequenceClassification(BertPreTrainedModel):
	def __init__(self, config, num_labels=2):
		super(BertForSequenceClassification, self).__init__(config)
		self.num_labels = num_labels

		config.output_hidden_states = True
		config.output_attentions = True

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

		self.idf = self.init_idf()
		self.init_weights()
		# self.apply(self.init_bert_weights)
	def init_idf(self):
		idf_dict = dict()
		with open(r'F:\ruijie\topicgraph.idf.txt', 'r', encoding='utf-8') as f:
			for line in f:
				fields = line.strip().split('\t')
				if len(fields) != 2:
					continue
				idf_dict[fields[0]] = float(fields[1])
		return idf_dict

	def forward(self, input_ids, attention_mask=None, token_type_ids=None ):
		sequence_output,  pooled_output, hidden_states, att_output = self.bert(input_ids, token_type_ids, attention_mask
															  )

		logits = self.classifier(torch.relu(pooled_output))

		return logits, att_output, sequence_output, pooled_output

	def get_idf_weight(self, text):
		if not text:
			return None
		cut_text = text.split(' ')
		bias = 0.1
		weights =[]

		for term in cut_text:
			if term in self.idf:
				score = self.idf[term]
			else:
				score = 15.0

			weights.extend([score / len(term) + bias] * len(term))
		return [x/ sum(weights) for x in weights]

	def predict_infer(self, text_a, text_b, tokenizer, max_seq_length = 128):

		# text_a = re.sub('[^\u4e00-\u9fff]', '', text_a)
		# text_b = re.sub('[^\u4e00-\u9fff]', '', text_b)
		tokens_a = tokenizer.tokenize(text_a)
		tokens_b = tokenizer.tokenize(text_b)

		idf_weights_a = self.get_idf_weight(text_a)
		idf_weights_b = self.get_idf_weight(text_b)

		_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
		tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
		segment_ids = [0] * len(tokens)
		tokens += tokens_b + ["[SEP]"]
		segment_ids += [1] * (len(tokens_b) + 1)
		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1] * len(input_ids)
		seq_length = len(input_ids)

		idf_weights =[15.0] +  idf_weights_a +  [15.0] + idf_weights_b + [15.0]

		padding = [0] * (max_seq_length - len(input_ids))
		input_ids += padding
		input_mask += padding
		segment_ids += padding
		idf_weights +=padding

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length
		assert len(idf_weights) == max_seq_length

		input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
		segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(device)
		input_mask = torch.tensor([input_mask], dtype=torch.long).to(device)

		idf_weights = torch.tensor([idf_weights], dtype=torch.float).to(device)

		sequence_output, pooled_output, hidden_states, att_output = self.bert(input_ids, segment_ids, input_mask
																			  )

		idf_weights = torch.unsqueeze(idf_weights, -1)
		adjust_weight = sequence_output * idf_weights

		return sequence_output, adjust_weight

	def predict_infer_batch(self, text_as, text_bs, tokenizer, max_seq_length = 128):

		# text_a = re.sub('[^\u4e00-\u9fff]', '', text_a)
		# text_b = re.sub('[^\u4e00-\u9fff]', '', text_b)
		adjust_weights = []

		for text_a, text_b in zip(text_as, text_bs):
			adjust_weights.append(self.predict_infer(text_a, text_b, tokenizer)[1])

		adjust_weights = torch.cat(adjust_weights, 0 )

		return torch.tensor(adjust_weights, dtype=torch.float)

def get_tensor_data(  features, output_mode = 'classification'):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids

def compute_metrics(preds, labels):
	return {'ac':(preds==labels).mean(),'f1':f1_score(y_true=labels,y_pred=preds)}

def infer(modelmane_or_path = r'F:\ruijie\pytorch-transformers\examples\chat\bert-base-uncased-hardem-transformer_binary_epoch3', cache_dir='/tmp/data/',max_seq=128, batch_size=32, num_epochs=10, lr=2e-5):
	processor = Processor()
	train_examples = processor.get_train_examples(r'F:\ruijie\data\chat')
	label_list = processor.get_labels()
	tokenizer = BertTokenizer.from_pretrained(modelmane_or_path, do_lower_case=True)
	model = BertForSequenceClassification.from_pretrained(modelmane_or_path,
											  num_labels=len(label_list))

	result = []
	for example in train_examples:
		eee = model.predict_infer(example.text_a, example.text_b, tokenizer)
		result.append(eee[1].detach().numpy())

	_re = np.vstack(result)

	fout = open('test.txt', 'w', encoding='utf-8')
	for item in _re:
		for line in item:
			fout.write('\t'.join([str(x) for x in line]))
			fout.write('\n')
	fout.close()

	with open('data/cache/bert_finetune_adjust','wb') as fout: pickle.dump(_re,fout)
	print('a')


	train_features = convert_examples_to_features(train_examples, label_list,
												  max_seq, tokenizer)
	train_data, _ = get_tensor_data(train_features)
	train_sampler = SequentialSampler(train_data)
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

	result = []
	with torch.no_grad():
		for batch in train_dataloader:
			input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
			res = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids )
			result.append(res[2])
	with open('data/cache/finetune_tr.txt', 'w', encoding='utf-8') as fin:
		for line in result:
			for lin in line:
				for li in lin:
					fin.write('\t'.join([str(x) for x in li]))
					fin.write('\n')


def main(bert_model='bert-base-chinese', cache_dir='/tmp/data/',\
	max_seq=128, batch_size=32, num_epochs=10, lr=2e-5):
	processor = Processor()
	train_examples = processor.get_train_examples('data/chat')
	label_list = processor.get_labels()
	tokenizer = BertTokenizer.from_pretrained(bert_model,do_lower_case=True)
	model = BertClassification.from_pretrained(bert_model,\
		cache_dir=cache_dir,num_labels=len(label_list))
	# model = BertTextCNN.from_pretrained(bert_model,\
	# 	cache_dir=cache_dir,num_labels=len(label_list))
	model.to(device)
	param_optimizer = list(model.named_parameters())
	no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params':[p for n,p in param_optimizer if not\
			any(nd in n for nd in no_decay)],'weight_decay':0.01},
		{'params':[p for n,p in param_optimizer if\
			any(nd in n for nd in no_decay)],'weight_decay':0.00}]
	print('train...')
	num_train_steps = int(len(train_examples)/batch_size*num_epochs)
	optimizer = AdamW(optimizer_grouped_parameters,lr=lr,warmup=0.1,t_total=num_train_steps)
	train_features = convert_examples_to_features(train_examples,label_list,max_seq,tokenizer)
	all_input_ids = torch.tensor([f.input_ids for f in train_features],dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in train_features],dtype=torch.long)
	all_label_ids = torch.tensor([f.label_id for f in train_features],dtype=torch.long)
	train_data = TensorDataset(all_input_ids,all_input_mask,all_label_ids)
	train_sampler = RandomSampler(train_data)
	train_dataloader = DataLoader(train_data,sampler=train_sampler,batch_size=batch_size)
	model.train()
	for _ in trange(num_epochs,desc='Epoch'):
		tr_loss = 0
		for step,batch in enumerate(tqdm(train_dataloader,desc='Iteration')):
			input_ids,input_mask,label_ids = tuple(t.to(device) for t in batch)
			loss = model(input_ids,input_mask,label_ids)
			loss.backward(); optimizer.step(); optimizer.zero_grad()
			tr_loss += loss.item()
		print('tr_loss',tr_loss)
	print('eval...')
	eval_examples = processor.get_dev_examples('data/chat')
	eval_features = convert_examples_to_features(eval_examples,label_list,max_seq,tokenizer)
	eval_input_ids = torch.tensor([f.input_ids for f in eval_features],dtype=torch.long)
	eval_input_mask = torch.tensor([f.input_mask for f in eval_features],dtype=torch.long)
	eval_label_ids = torch.tensor([f.label_id for f in eval_features],dtype=torch.long)
	eval_data = TensorDataset(eval_input_ids,eval_input_mask,eval_label_ids)
	eval_sampler = SequentialSampler(eval_data)
	eval_dataloader = DataLoader(eval_data,sampler=eval_sampler,batch_size=batch_size)
	model.eval()
	preds = []
	for batch in tqdm(eval_dataloader,desc='Evaluating'):
		input_ids,input_mask,label_ids = tuple(t.to(device) for t in batch)
		with torch.no_grad():
			logits = model(input_ids,input_mask,None)
			preds.append(logits.detach().cpu().numpy())
	preds = np.argmax(np.vstack(preds),axis=1)
	print(compute_metrics(preds,eval_label_ids.numpy()))
	torch.save(model,'data/cache/model')

if __name__ == '__main__':
	infer()
	# main()
