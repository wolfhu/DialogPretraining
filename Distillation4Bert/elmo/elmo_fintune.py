

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

from allennlp.modules.elmo import Elmo, batch_to_ids
from utils import load_data, load_data_simple
from keras.preprocessing import sequence


import os
import numpy as np

elmo_options_file = 'options.json'
elmo_weights_file = 'pretrain.wiki.ep10.hdf5'
USE_CUDA = torch.cuda.is_available()
if USE_CUDA: torch.cuda.set_device(0)
LTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

class ElmoFintune(torch.nn.Module):
    def __init__(self, hidden_size = 512, num_labels = 2):
        super(ElmoFintune, self).__init__()

        self.model_name = 'ElmoFinetune'
        self.hidden_size = hidden_size

        self.elmo = Elmo(elmo_options_file, elmo_weights_file, num_output_representations=1, dropout=0.3)
        self.elmo_hidden_size = 1024

        self.blstm_layers_num = 2
        self.blstm_q = torch.nn.LSTM(self.elmo_hidden_size, self.hidden_size, self.blstm_layers_num, bidirectional=True, batch_first=True)
        self.blstm_a = torch.nn.LSTM(self.elmo_hidden_size, self.hidden_size, self.blstm_layers_num, bidirectional=True,
                                     batch_first=True)

        self.linear = torch.nn.Linear( 4 * self.hidden_size, num_labels)
        self.softmax = torch.nn.Softmax(dim = 1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self,elmo_ids_q, elmo_ids_a):
        elmo_embedding_q = self.elmo(elmo_ids_q)['elmo_representations'][-1]
        #elmo_embedding shape (batch_size, max_seq_len, 1024)
        out_q, _ = self.blstm_q(elmo_embedding_q)

        elmo_embedding_a = self.elmo(elmo_ids_a)['elmo_representations'][-1]
        out_a, _ = self.blstm_a(elmo_embedding_a)

        qa_concat = torch.cat((out_q[:,-1,:], out_a[:,-1,:]), -1)

        hidden = self.linear(qa_concat)
        return self.softmax(hidden), self.log_softmax(hidden)

class Model(object):
    def __init__(self, v_size):
        self.model = None
        self.b_size = 64
        self.lr = 0.001
        self.model = ElmoFintune(v_size)
        # self.model = CNN(v_size,256,128,2)

    def train(self, x_a_tr, x_b_tr, y_tr, x_a_te,x_b_te, y_te, epochs=15):
        assert self.model is not None
        if USE_CUDA: self.model = self.model.cuda()
        loss_func = nn.NLLLoss()
        opt = optim.Adam(self.model.parameters(),lr=self.lr)
        for epoch in range(epochs):
            losses = []; accu = []
            self.model.train()
            for i in range(0,len(x_a_tr),self.b_size):
                self.model.zero_grad()
                # bx = Variable(LTensor(x_tr[i:i+self.b_size]))
                bx_a = x_a_tr[i:i + self.b_size]
                elmo_ids_a = batch_to_ids(bx_a)
                if USE_CUDA: elmo_ids_a = elmo_ids_a.cuda()

                bx_b = x_b_tr[i:i + self.b_size]
                elmo_ids_b = batch_to_ids(bx_b)
                if USE_CUDA: elmo_ids_b = elmo_ids_b.cuda()

                by = Variable(LTensor(y_tr[i:i+self.b_size]))
                _,py = self.model(elmo_ids_a, elmo_ids_b)
                loss = loss_func(py,by)
                loss.backward(); opt.step()
                losses.append(loss.item())
            self.model.eval()
            with torch.no_grad():
                for i in range(0,len(x_a_te),self.b_size):
                    # bx = Variable(LTensor(x_te[i:i+self.b_size]))
                    bx_a = x_a_te[i:i + self.b_size]
                    elmo_ids_a = batch_to_ids(bx_a)
                    if USE_CUDA: elmo_ids_a = elmo_ids_a.cuda()

                    bx_b = x_b_te[i:i + self.b_size]
                    elmo_ids_b = batch_to_ids(bx_b)
                    if USE_CUDA: elmo_ids_b = elmo_ids_b.cuda()

                    by = Variable(LTensor(y_te[i:i+self.b_size]))
                    _,py = torch.max(self.model(elmo_ids_a, elmo_ids_b)[1],1)
                    accu.append((py==by).float().mean().item())
                print(np.mean(losses), np.mean(accu))


        self.model.eval()
        fout = open('elmo.finetune.score.txt', 'w', encoding='utf-8')
        with torch.no_grad():
            for i in range(0, len(x_a_te), self.b_size):
                # bx = Variable(LTensor(x_te[i:i+self.b_size]))
                bx_a = x_a_te[i:i + self.b_size]
                elmo_ids_a = batch_to_ids(bx_a)
                if USE_CUDA: elmo_ids_a = elmo_ids_a.cuda()

                bx_b = x_b_te[i:i + self.b_size]
                elmo_ids_b = batch_to_ids(bx_b)
                if USE_CUDA: elmo_ids_b = elmo_ids_b.cuda()

                by = Variable(LTensor(y_te[i:i + self.b_size]))
                logits, py = torch.max(self.model(elmo_ids_a, elmo_ids_b)[1], 1)
                # accu.append((py == by).float().mean().item())

                for pred, s in zip(logits, by):
                    fout.write('\t'.join([ str(x) for x in pred]) + '\t' + str(s))
                    fout.write('\n')
        fout.close()

if __name__ == '__main__':
    x_len = 50


    x= [['我','喜欢','吃','苹果'], ['我','喜欢','吃','菠萝' ]]
    y = [1,0]

    model = ElmoFintune()
    a,b = model(batch_to_ids(x), batch_to_ids(x))
    print('a')

    name = 'chat'  # clothing, fruit, hotel, pda, shampoo
    (x_a_tr,x_b_tr, y_tr, _), _, (x_a_te,x_b_te, y_te, _) = load_data_simple(name)
    # l_tr = list(map(lambda x: min(len(x), x_len), x_tr))
    # l_te = list(map(lambda x: min(len(x), x_len), x_te))
    # x_tr = sequence.pad_sequences(x_tr, maxlen=x_len)
    # x_te = sequence.pad_sequences(x_te, maxlen=x_len)
    clf = Model(x_len)
    clf.train(x_a_tr, x_b_tr, y_tr,   x_a_te, x_b_te, y_te )