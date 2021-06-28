# -*- coding: utf-8 -*-

import torch
#from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertTokenizer
from torch import nn
import torch.functional as F
import torch.optim as optim
from .elmo_fintune import ElmoFintune as elmo
from allennlp.modules.elmo import   batch_to_ids
from .utils import load_data_simple
from .bert_infer import BertForSequenceClassification
from torch.autograd import Variable
import numpy as np

USE_CUDA = torch.cuda.is_available()
FTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
device = torch.device('cuda' if USE_CUDA else 'cpu')

class Teacher(object):
	def __init__(self, bert_model='bert-base-chinese', max_seq=128):
		self.max_seq = max_seq
		self.tokenizer = BertTokenizer.from_pretrained(
			bert_model,do_lower_case=True)
		self.model = torch.load('data/cache/model')
		self.model.eval()
	def predict(self, text):
		tokens = self.tokenizer.tokenize(text)[:self.max_seq]
		input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1]*len(input_ids)
		padding = [0]*(self.max_seq-len(input_ids))
		input_ids = torch.tensor([input_ids+padding],dtype=torch.long).to(device)
		input_mask = torch.tensor([input_mask+padding],dtype=torch.long).to(device)
		logits = self.model(input_ids,input_mask,None)
		return F.softmax(logits,dim=1).detach().cpu().numpy()

if __name__ == '__main__':

	modelmane_or_path = r'F:\ruijie\pytorch-transformers\examples\models_bert\chat\bert-base-uncased-hardem-transformer_binary_epoch3'
	bert_tokenizer = BertTokenizer.from_pretrained(modelmane_or_path, do_lower_case=True)
	bert_model = BertForSequenceClassification.from_pretrained(modelmane_or_path,
															   num_labels=2)

	import pickle
	from tqdm import tqdm
	x_len = 50; b_size = 128; lr = 2e-5; epochs=20
	name = 'chat' # clothing, fruit, hotel, pda, shampoo
	(x_a_tr, x_b_tr, y_tr, _), (x_a_de, x_b_de, y_de, _), (x_a_te, x_b_te, y_te, _) = load_data_simple(name)
	# l_tr = list(map(lambda x:min(len(x),x_len),x_tr))
	# l_de = list(map(lambda x:min(len(x),x_len),x_de))
	# l_te = list(map(lambda x:min(len(x),x_len),x_te))
	# x_tr = sequence.pad_sequences(x_tr,maxlen=x_len)
	# x_de = sequence.pad_sequences(x_de,maxlen=x_len)
	# x_te = sequence.pad_sequences(x_te,maxlen=x_len)
	# # t_tr = np.vstack([teacher.predict(text) for text in tqdm(t_tr)])
	# # t_de = np.vstack([teacher.predict(text) for text in tqdm(t_de)])
	# # with open('data/cache/t_tr','wb') as fout: pickle.dump(t_tr,fout)
	# # with open('data/cache/t_de','wb') as fout: pickle.dump(t_de,fout)
	# with open('data/cache/t_tr','rb') as fin: t_tr = pickle.load(fin)
	# with open('data/cache/t_de','rb') as fin: t_de = pickle.load(fin)

	# model = RNN(v_size,256,256,2)
	model = elmo()
	# model = CNN(v_size,256,128,2)
	if USE_CUDA: model = model.cuda()
	opt = optim.Adam(model.parameters(),lr=lr)
	loss1 = nn.NLLLoss()
	loss2 = nn.MSELoss()


	for epoch in range(epochs):
		losses = []; accu = []
		model.train()
		for i in range(0,len(x_a_tr),b_size):
			model.zero_grad()
			bx_a = x_a_tr[i:i +  b_size]
			elmo_ids_a = batch_to_ids(bx_a)
			if USE_CUDA: elmo_ids_a = elmo_ids_a.cuda()

			bx_b = x_b_tr[i:i +  b_size]
			elmo_ids_b = batch_to_ids(bx_b)
			if USE_CUDA: elmo_ids_b = elmo_ids_b.cuda()

			by = Variable(LTensor(y_tr[i:i+b_size]))

			bt = bert_model.predict_infer_batch(bx_a, bx_b, bert_tokenizer, max_seq_length=x_len)

			py1,py2 = model(elmo_ids_a, elmo_ids_b)
			loss = loss1(py2,by)+loss2(py1,bt) # in paper, only mse is used
			loss.backward(); opt.step()
			losses.append(loss.item())
		model.eval()
		with torch.no_grad():
			for i in range(0, len(x_a_te), 1):
				bx_a = x_a_te[i:i + b_size]
				elmo_ids_a = batch_to_ids(bx_a)
				if USE_CUDA: elmo_ids_a = elmo_ids_a.cuda()


				bx_b = x_b_te[i:i + b_size]
				elmo_ids_b = batch_to_ids(bx_b)
				if USE_CUDA: elmo_ids_b = elmo_ids_b.cuda()

				by = Variable(LTensor(y_te[i:i + b_size]))

				logits, pred = model(elmo_ids_a, elmo_ids_b)

				_, py = torch.max(py2, 1)
				accu.append((py == by).float().mean().item())
		print(np.mean(losses), np.mean(accu))

		torch.save(model,'elmo_distill.pytorch.bin')
		model.eval()
		writer = open('elmo.result.txt', 'w')
		with torch.no_grad():
			for i in range(0,len(x_a_te), b_size):
				bx_a = x_a_te[i:i + b_size]
				elmo_ids_a = batch_to_ids(bx_a)
				if USE_CUDA: elmo_ids_a = elmo_ids_a.cuda()

				bx_b = x_b_te[i:i + b_size]
				elmo_ids_b = batch_to_ids(bx_b)
				if USE_CUDA: elmo_ids_b = elmo_ids_b.cuda()

				by = Variable(LTensor(y_te[i:i+b_size]))
				# bl = Variable(LTensor(l_te[i:i+1]))

				logits, pred = model(elmo_ids_a, elmo_ids_b)
				_,py = torch.max(pred,1)
				# logits,py = torch.max(model(bx,bl)[1],1)
				accu.append((py==by).float().mean().item())
				for l,p in zip(logits, by):
					writer.write('\t'.join(str(t) for t in l.cpu().numpy()) + '\t' + str(p.cpu().numpy()) + '\n')
		writer.close()
		print(np.mean(losses),np.mean(accu))
