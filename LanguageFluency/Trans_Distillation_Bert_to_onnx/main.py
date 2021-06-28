

import torch
import torchvision
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig
import onnxruntime as ort
import numpy as np
import torch.nn.functional as F


def torch2onnx():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese').to(device)
    model.bert.encoder.layer = model.bert.encoder.layer[:3]
    model_dict = torch.load('./param', map_location=device).module.state_dict()
    model.load_state_dict(model_dict)
    input_ids = torch.randint(1, (1, 20)).to(device)
    attention_mask = torch.randint(1, (1, 20)).to(device)
    token_type_ids = torch.randint(1, (1, 20)).to(device)
    position_ids = torch.randint(1, (1, 20)).to(device)
    torch.onnx.export(model, (input_ids, attention_mask, token_type_ids, position_ids), "alexnet.onnx",
                      input_names=['input_ids', 'attention_mask', 'token_type_ids', 'position_ids'], output_names=['pred_result'], verbose=True)


class Sen2id:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def translate(self, sentence):
        tmp = self.tokenizer.encode(sentence, add_special_tokens=True)
        if len(tmp) > 20:
            tmp = tmp[:19]+[tmp[-1]]
        input_len = len(tmp)
        input_ids = np.array([tmp+[0]*(20-input_len)]).astype(np.int64)
        attention_mask = np.array([[1]*input_len+[0]*(20-input_len)]).astype(np.int64)
        #token_type_ids = np.array([[1]*input_len+[0]*(20-input_len)]).astype(np.int64)
        token_type_ids = np.array([[0]*(20)]).astype(np.int64)
        position_ids = np.array([list(range(input_len))+[0]*(20-input_len)]).astype(np.int64)
        return input_ids, attention_mask, token_type_ids, position_ids



def calcu_torch():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese').to(device)
    model.bert.encoder.layer = model.bert.encoder.layer[:3]
    model_dict = torch.load('./param', map_location=device).module.state_dict()
    model.load_state_dict(model_dict)
    input_ids = [[1, 3124, 445, 557, 657, 775, 758, 57] + [0] * (7)]
    attention_mask = [[1] * 8 + [0] * 7]
    token_type_ids = [[1] * 8 + [0] * 7]
    position_ids = [[0, 1, 2, 3, 4, 5, 6, 7] + [0] * 7]
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    position_ids = torch.tensor(position_ids)
    model.eval()
    output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids)
    print(output[0])



def predict_from_file():
    ort_session = ort.InferenceSession('alexnet.onnx')
    sen2id = Sen2id()
    f1 = open('./pred_data/pred.txt', 'r', encoding='utf-8')
    f2 = open('./pred_data/pred_result.txt', 'w', encoding='utf-8')
    line = f1.readline().split('\t')[0]
    i = 0
    while line:
        input_ids, attention_mask, token_type_ids, position_ids = sen2id.translate(line)
        label = ort_session.run(['pred_result'], {'input_ids': input_ids, 'attention_mask': attention_mask,
                                                  'token_type_ids':token_type_ids, 'position_ids':position_ids})[0]
        label_n = F.softmax(torch.tensor(label), dim=-1).tolist()[0]
        f2.write('\t'.join([str(d) for d in label_n]) + '\n')
        line = f1.readline().split('\t')[0]
        i += 1
        if i % 1000 == 0:
            print(i)
    f1.close()
    f2.close()


#torch2onnx()
predict_from_file()
