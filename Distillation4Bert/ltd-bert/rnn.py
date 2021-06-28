# -*- coding: utf-8 -*-

import torch, numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from keras.preprocessing import sequence

USE_CUDA = torch.cuda.is_available()
if USE_CUDA: torch.cuda.set_device(0)
LTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
_FTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
	def __init__(self, x_dim, e_dim, h_dim, o_dim):
		super(RNN,self).__init__()
		self.h_dim = h_dim
		self.dropout = nn.Dropout(0.2)
		self.emb = nn.Embedding(x_dim,e_dim,padding_idx=0)
		# self.lstm = nn.LSTM(e_dim,h_dim,bidirectional=True,batch_first=True)

		self.blstm_layers_num = 3
		self.blstm_q = torch.nn.LSTM(e_dim, self.h_dim, self.blstm_layers_num, bidirectional=True,
									 batch_first=True)
		self.blstm_a = torch.nn.LSTM(e_dim, self.h_dim, self.blstm_layers_num, bidirectional=True,
									 batch_first=True)

		self.fc = nn.Linear(h_dim*4,o_dim)
		self.softmax = nn.Softmax(dim=1)
		self.log_softmax = nn.LogSoftmax(dim=1)

		self.idf = self.init_idf()

	def init_idf(self):
		idf_dict = dict()
		with open(r'F:\ruijie\topicgraph.idf.txt', 'r', encoding='utf-8') as f:
			for line in f:
				fields = line.strip().split('\t')
				if len(fields) != 2:
					continue
				idf_dict[fields[0]] = float(fields[1])
		return idf_dict

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

			weights.append(score)
		return [x/ sum(weights) for x in weights]

	def get_batch_idf(self, texts):
		weights = []
		for text in texts:
			weights.append(self.get_idf_weight(text))
		return weights

	def _step(self, x, tokenizer, is_q = True, max_len= 128):
		idf_weights = self.get_batch_idf(x)
		pad_idf_weight = sequence.pad_sequences(idf_weights, maxlen=max_len, padding='post', dtype='float', value=0.0)
		x_ids = tokenizer.texts_to_sequences(x)
		x_ids = sequence.pad_sequences(x_ids, maxlen=max_len, padding='post')

		x_ids = Variable(LTensor(x_ids))
		pad_idf_weight = _FTensor(pad_idf_weight)

		embed = self.dropout(self.emb(x_ids))

		idf_weights = torch.unsqueeze(pad_idf_weight, -1)
		embed = embed * idf_weights
		if is_q:
			out, (ht, ct) = self.blstm_q(embed)
		else:
			out, (ht, ct) = self.blstm_a(embed)

		return out

	def forward(self, text_as, text_bs, tokenizer,  max_len= 128):

		out_a  = self._step(text_as, tokenizer, is_q=True)
		out_b  = self._step(text_bs, tokenizer, is_q=False)

		adjust_weights = torch.cat((out_a, out_b), -1)

		hidden = self.fc(adjust_weights[:,-1,:])

		return  self.softmax(hidden), self.log_softmax(hidden), adjust_weights