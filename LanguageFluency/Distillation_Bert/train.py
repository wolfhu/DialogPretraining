# coding: utf-8
from transformers import BertTokenizer
from tqdm import tqdm
from pprint import pprint
from random import shuffle

def read_data(input_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return [d[:-1] for d in data]


def save_file(input_data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines([d+'\n' for d in input_data])


def preprocess_raw_data():
    pos_data = []
    neg_data = []
    #tokenizer = BertTokenizer(vocab_file='./vocabulary/vocab_small.txt')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    neg_data.extend(read_data('./data/error_data_log.txt'))
    pos_data.extend(read_data('./data/correct_data_log.txt'))
    shuffle(pos_data)
    pos_data = pos_data[:len(neg_data)]
    pos_data.extend(read_data('./data/correct_data_sr.txt'))
    neg_data.extend(read_data('./data/error_data_sr.txt'))
    shuffle(pos_data)
    shuffle(neg_data)

    print('pos_num:', len(pos_data))
    print('neg_num:', len(neg_data))
    res = []
    for d in tqdm(pos_data, desc='pos'):
        context, distribution = d.split('\t')[0], d.split('\t')[1:]
        #d_ids = [tokenizer.cls_token_id]+[tokenizer.convert_tokens_to_ids(w) for w in context] + [tokenizer.sep_token_id]
        d_ids = tokenizer.encode(context, add_special_tokens=True)
        res.append(' '.join([str(w) for w in d_ids])+'\t'+'\t'.join(distribution))
    for d in tqdm(neg_data, desc='neg'):
        context, distribution = d.split('\t')[0], d.split('\t')[1:]
        #d_ids = [tokenizer.cls_token_id] + [tokenizer.convert_tokens_to_ids(w) for w in context] + [tokenizer.sep_token_id]
        d_ids = tokenizer.encode(context, add_special_tokens=True)
        res.append(' '.join([str(w) for w in d_ids]) + '\t' +'\t'.join(distribution))
    save_file(res, './data/all_dataset_ids.tsv')


#preprocess_raw_data()



from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.nn import MSELoss

class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index].strip()
        dirtribution, input_ids = input_ids.split('\t')[1:], input_ids.split('\t')[0]
        input_ids = [int(token_id) for token_id in input_ids.split(' ')]
        dirtribution = [float(dis) for dis in dirtribution]
        return dirtribution, input_ids

    def __len__(self):
        return len(self.data_list)

class Datasets:
    def __init__(self, input_file_path, test_ratio=0.05):
        #self.tokenizer = BertTokenizer(vocab_file='./vocabulary/vocab_small.txt')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        dataset = read_data(input_file_path)
        train_dataset, test_dataset = train_test_split(dataset, test_size=test_ratio, random_state=1)
        self.train_dataset = MyDataset(train_dataset)
        self.test_dataset = MyDataset(test_dataset)

    def get_train_dataloader(self, batch_size=100, is_shuffle=True, num_workers=0):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers, collate_fn=self.collate_fn)

    def get_test_dataloader(self, batch_size=100, is_shuffle=False, num_workers=0):
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        """
        ?????????batch????????????sample????????????input??????????????????input?????????????????????
        :param batch:
        :return:
        """
        pad_id = self.tokenizer.pad_token_id
        input_ids = []
        btc_size = len(batch)
        max_input_len = 0  # ???batch????????????input????????????batch???????????????
        # ?????????batch???input???????????????
        distribution, batch_ids = [d[0] for d in batch], [d[1] for d in batch]

        for btc_idx in range(btc_size):
            if max_input_len < len(batch_ids[btc_idx]):
                max_input_len = len(batch_ids[btc_idx])
        # ??????pad_id?????????max_input_len???input_id????????????

        att_mask = []
        for btc_idx in range(btc_size):
            input_len = len(batch_ids[btc_idx])
            att_mask.append([1] * input_len + [0] * (max_input_len - input_len))
            input_ids.append(batch_ids[btc_idx])
            input_ids[btc_idx].extend([pad_id] * (max_input_len - input_len))

        distribution = torch.tensor(distribution, dtype=torch.float)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        att_mask = torch.tensor(att_mask, dtype=torch.long)
        return distribution, input_ids, att_mask

class Id2sen:
    def __init__(self):
        #self.tokenizer = BertTokenizer(vocab_file='./vocabulary/vocab_small.txt')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def translate(self, ids):
        ids = [d for d in ids if d!=0]
        tmp = ids[1:-1]
        tmp = [self.tokenizer.ids_to_tokens[w] for w in tmp]
        return ''.join(tmp)

def train_model(epoch_num=10):
    dataset = Datasets('./data/all_dataset_ids.tsv')
    train_dataloader = dataset.get_train_dataloader(batch_size=1000)
    test_dataloader = dataset.get_test_dataloader(batch_size=1000)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #from rcnn_model import RCNN_Classifer
    #model = RCNN_Classifer(word_num=dataset.tokenizer.vocab_size, input_dim=200, core_lens=(1,2,3,4), output_channel=10, output_dim=2).to(device)
    from transformers import BertForSequenceClassification, BertConfig
    #configuration = BertConfig(num_hidden_layers=3, num_labels=2, num_attention_heads=12, hidden_size=192, max_position_embeddings=32)
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese').to(device)
    model.bert.encoder.layer = model.bert.encoder.layer[:3]
    model = torch.nn.DataParallel(model).to(device)
    learning_rate = 0.0001
    criterion = MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    pre_avg_acc = -1
    max_acc = -1
    count = 0
    Temperature = 3

    for epoch in range(epoch_num):
        model.train()  # set the model to train mode (dropout=True)
        avg_cost = 0
        total_train_batch = len(train_dataloader)

        for distribution, input_ids, att_mask in tqdm(train_dataloader):
            # ?????????GPT2?????????forward()???????????????????????????context???????????????token????????????????????????token
            # GPT2Model????????????n???token_id??????????????????n???hidden_state????????????n???hidden_state?????????n+1???token
            distribution = distribution.to(device)
            input_ids = input_ids.to(device)
            att_mask = att_mask.to(device)
            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=att_mask)[0]
            output = F.softmax(output / Temperature, dim=-1)
            distribution = F.softmax(distribution / Temperature, dim=-1)
            cost = criterion(output.view(-1), distribution.view(-1))
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_train_batch
            print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, cost))
        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

        with torch.no_grad():
            model.eval()
            avg_acc = 0
            total_valid_batch = len(test_dataloader)
            for distribution, input_ids, att_mask in tqdm(test_dataloader):
                distribution = distribution.to(device)
                input_ids = input_ids.to(device)
                att_mask = att_mask.to(device)

                output = model(input_ids=input_ids, attention_mask=att_mask)[0]
                output = F.softmax(output / Temperature, dim=-1)
                distribution = F.softmax(distribution / Temperature, dim=-1)
                output = torch.argmax(output, dim=-1)
                distribution = torch.argmax(distribution, dim=-1)
                correct_prediction = distribution == output

                avg_acc += correct_prediction.float().mean().item() / total_valid_batch
            print('Acc ', avg_acc)

        if avg_acc < pre_avg_acc:
            count += 1
            if count >= 1:
                learning_rate = learning_rate / 2
                count = 0
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                print('learing rate adjust: ', learning_rate)
        else:
            #save_file_name = './model/rcnn/' + str(epoch) + '_' + str(avg_acc)[:5]
            save_file_name = './model/bert/param_0.00005'
            if epoch == 0:
                max_save_file_name = save_file_name
                max_acc = avg_acc
            elif epoch != 0 and avg_acc > max_acc:
                print('max changed')
                max_save_file_name = save_file_name
                max_acc = avg_acc
            torch.save(model, save_file_name)
            print('saveed..')
        count = 0
        pre_avg_acc = avg_acc

    print('Learning Finished!')

train_model(10)

class Sen2id:
    def __init__(self):
        #self.tokenizer = BertTokenizer(vocab_file='./vocabulary/vocab_small.txt')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def translate(self, sentence):
        #tmp = [self.tokenizer.cls_token_id]+[self.tokenizer.convert_tokens_to_ids(w) for w in sentence]+[self.tokenizer.sep_token_id]
        tmp = self.tokenizer.encode(sentence, add_special_tokens=True)
        if len(tmp) <= 3:
            tmp += [self.tokenizer.pad_token_id]*(4-len(tmp))
        tmp = torch.tensor(tmp).unsqueeze(0)
        return tmp

def predict_from_input():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load('./model/bert/param_layer_3_hidden_192', map_location=device)
    model.eval()
    sen2id = Sen2id()

    while True:
        sentence = input('??????????????????')
        sentence = sen2id.translate(sentence)
        sentence = sentence.to(device)
        label = model(sentence)[0]
        label = F.softmax(label, dim=-1)
        label_n = label.argmax(dim=-1).tolist()[0]
        label_v = label.tolist()[0][label_n]
        label_n = 'pos' if label_n == 0 else 'neg'
        print(label_n, label_v)

def predict_from_file():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load('./model/bert/param_layer_3_hidden_192', map_location=device)
    model.eval()
    sen2id = Sen2id()
    f1 = open('./pred_data/pred.txt', 'r', encoding='utf-8')
    f2 = open('./pred_data/pred_result.txt', 'w', encoding='utf-8')
    line = f1.readline().split('\t')[0]
    i=0
    while line:
        line_id = sen2id.translate(line)
        line_id = line_id.to(device)
        label = model(line_id)[0]
        label_n = F.softmax(label, dim=-1).tolist()[0]
        f2.write('\t'.join([str(d) for d in label_n])+'\n')
        line = f1.readline().split('\t')[0]
        i += 1
        if i % 1000 == 0:
            print(i)
    f1.close()
    f2.close()

#predict_from_input()

print(1)