import torch.nn as nn
import torch.nn.functional as F
import torch

class RCNN_Classifer(nn.Module):
    def __init__(self, word_num, input_dim, core_lens, output_channel, output_dim):
        super(RCNN_Classifer, self).__init__()
        self.word_num = word_num
        self.conv_list = nn.ModuleList()
        self.core_lens = core_lens
        assert input_dim % 2 == 0
        self.input_dim = input_dim
        self.hidden_dim = input_dim//2
        self.output_channel = output_channel
        self.output_dim = output_dim
        self.embedding = nn.Embedding(self.word_num, self.input_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, dropout=0.2, num_layers=2, bidirectional=True)
        for conv_len in self.core_lens:
            self.conv_list.append(nn.Conv1d(self.input_dim, self.output_channel, conv_len))
        self.output = nn.Linear(self.output_channel*len(self.core_lens), self.output_dim)

    def forward(self, x, T=1):
        results = []
        x = self.embedding(x)
        batch_size = x.size()[0]
        h0 = torch.zeros(2*self.lstm.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(2*self.lstm.num_layers, batch_size, self.hidden_dim).to(x.device)
        x_, (_, _) = self.lstm(x, (h0, c0))
        x = x+x_
        x = x.permute(0, 2, 1)
        for corv in self.conv_list:
            results.append(torch.max(F.relu(corv(x)), dim=-1)[0])
        results = torch.cat(tuple(results), dim=-1)
        results = self.output(results)/T
        output = F.softmax(results, dim=-1)
        return output

