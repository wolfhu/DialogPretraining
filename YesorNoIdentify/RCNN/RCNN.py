import torch.nn as nn
import torch.nn.functional as F
import torch

class RCNN_Classifer(nn.Module):
    def __init__(self, word_num, input_dim, hidden_dim, core_lens, output_channel, device):
        super(RCNN_Classifer, self).__init__()
        self.device = device
        self.word_num = word_num
        self.conv_list = nn.ModuleList()
        self.core_lens = core_lens
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channel = output_channel
        self.output_dim = 4
        self.embedding = nn.Embedding(self.word_num, self.input_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, dropout=0.2, bidirectional=True)
        for conv_len in self.core_lens:
            self.conv_list.append(nn.Conv1d(self.input_dim, self.output_channel, conv_len).to(device))
        self.output = nn.Linear(self.output_channel*len(self.core_lens), self.output_dim)

    def forward(self, x):
        results = []
        x = self.embedding(x)
        print(x.cpu())
        batch_size = x.size()[0]
#         h0 = torch.randn(2, batch_size, self.hidden_dim).to(self.device)
#         c0 = torch.randn(2, batch_size, self.hidden_dim).to(self.device)
        h0 = torch.zeros(2, batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(2, batch_size, self.hidden_dim).to(self.device)
        x_, (_, _) = self.lstm(x, (h0, c0))
        print(x_.cpu())
        x = x+x_
        print(x.cpu())
        x = x.permute(0, 2, 1)
        for corv in self.conv_list:
            results.append(torch.max(F.relu(corv(x)), dim=-1)[0])
        results = torch.cat(tuple(results), dim=-1)
        print(results.cpu())
        output = F.softmax(self.output(results), dim=-1)
        print(output.cpu())
        return output

