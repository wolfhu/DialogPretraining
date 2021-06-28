import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import pickle
import random
import torch.utils.data
from tqdm import tqdm
from pprint import pprint
import os
import shutil

def Padding_Cutting(data, max_len):
    result = [l[:max_len] for l in data]
    result_len = [len(l) for l in result]
    max_len_tmp = max(result_len)
    result = [l+[word_list.index('PAD')]*(max_len_tmp-len(l)) if len(l)<max_len_tmp else l for l in result]
    return result, result_len

def preprocess(data, word_list, max_len=28):
    results = [[word_list.index('SOS')]+
               [word_list.index(w) if w in word_list else word_list.index('UNK') for w in d]+
               [word_list.index('EOS')] for d in data]
    results, results_len = Padding_Cutting(results, max_len)
    return results, results_len

def from_input_file(input_file_path, output_file_dir, word_list, model, device):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        _data = [d.split('\t')[0] for d in f.readlines()]


    data, data_len = preprocess(_data, word_list)
    data = [(data[i], data_len[i]) for i in range(len(data))]
    batch_size = 500
    predict_data_loader = torch.utils.data.DataLoader(dataset=data,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      drop_last=False)
    prediction_all = []
    transed_input = []
    with torch.no_grad():
        model.eval()
        for X, X_len in tqdm(predict_data_loader):
            X = torch.cat(tuple([xxx.unsqueeze(1) for xxx in X]), dim=1)
            X = X.to(device)
            prediction = model(X)
            #_, prediction = torch.max(prediction, -1)
            prediction = prediction.cpu().numpy().tolist()
            X = X.cpu().numpy().tolist()
            X_len = X_len.cpu().numpy().tolist()
            X = [''.join(list(map(lambda w:word_list[w], X[i][1:X_len[i]-1]))) for i in range(len(X))]
            transed_input.extend(X)
            prediction_all.extend(prediction)
            #data_all.extend(X)
    assert len(prediction_all) == len(_data)

    prediction_all = [[str(round(v, 6)) for v in d] for d in prediction_all]
    with open(output_file_dir + 'resutls.txt', 'w', encoding='utf-8') as f:
        f.writelines(['\t'.join(d)+'\n' for d in prediction_all])
    with open(output_file_dir + 'transed_input.txt', 'w', encoding='utf-8') as f:
        f.writelines([d+'\n' for d in transed_input])
    return


def eval_result(predict_path, label_file, type_list, output_file_dir, threshold=None):
    with open(predict_path, 'r', encoding='utf-8') as f:
        pred_ = f.readlines()
    pred_ = [[float(w) for w in d[:-1].split('\t')] for d in pred_]
    pred_v = list(np.array(pred_).max(axis=-1))
    pred = [type_list[t] for t in list(np.array(pred_).argmax(axis=-1))]
    assert len(pred) == len(pred_v)

    with open(label_file, 'r', encoding='utf-8') as f:
        label = f.readlines()
    label = [d[:-1].split('\t')[:2] for d in label]
    if threshold is not None:
        can_predict = [True if pred_v[i] > threshold[pred[i]] else False for i in range(len(pred))]
        pred = [pred[i] for i in range(len(pred)) if can_predict[i]]
        label = [label[i] for i in range(len(label)) if can_predict[i]]
    assert len(pred) == len(label)

    certain_err = []
    neg_err = []
    uncertain_err = []
    question_err = []
    label_num = dict()
    correct_num = dict()

    for k in type_list:
        label_num[k] = len([d for d in label if d[-1] == k])
        correct_num[k] = len([d for d in label if d[-1] == k])

    for i in range(len(label)):
        if pred[i] != label[i][1]:
            if label[i][1] == 'certain':
                certain_err.append(pred[i] + '\t' + label[i][0] + '\n')
                correct_num['certain'] -= 1
            elif label[i][1] == 'neg':
                neg_err.append(pred[i] + '\t' + label[i][0] + '\n')
                correct_num['neg'] -= 1
            elif label[i][1] == 'uncertain':
                uncertain_err.append(pred[i] + '\t' + label[i][0] + '\n')
                correct_num['uncertain'] -= 1
            elif label[i][1] == 'question':
                question_err.append(pred[i] + '\t' + label[i][0] + '\n')
                correct_num['question'] -= 1

    pred_num = {'certain': 0, 'neg': 0, 'uncertain': 0, 'question': 0}
    for k in type_list:
        pred_num[k] = len([d for d in pred if d == k])

    p_result = dict()
    r_result = dict()
    f1_result = dict()
    for k in type_list:
        p_result[k] = correct_num[k]/label_num[k]
        r_result[k] = correct_num[k]/pred_num[k]
        f1_result[k] = 2*(p_result[k]*r_result[k]/(p_result[k]+r_result[k]))
    print('label_num', label_num)
    print('pred_num', pred_num)
    print('correct_num', correct_num)

    eval_output = []
    tmp = '\t'
    for k in type_list:
        tmp += k+'\t'
    eval_output.append(tmp+'\n')
    print('p', p_result)
    tmp = 'p\t'
    for k in type_list:
        tmp += str(round(p_result[k], 5))+'\t'
    eval_output.append(tmp + '\n')
    print('r', r_result)
    tmp = 'r\t'
    for k in type_list:
        tmp += str(round(r_result[k], 5)) + '\t'
    eval_output.append(tmp + '\n')
    print('f1', f1_result)
    tmp = 'f1\t'
    for k in type_list:
        tmp += str(round(f1_result[k], 5)) + '\t'
    eval_output.append(tmp + '\n')

    with open(output_file_dir + 'eval.txt', 'w', encoding='utf-8') as f:
        f.writelines(eval_output)
    with open(output_file_dir + 'errcertain.txt', 'w', encoding='utf-8') as f:
        f.writelines(certain_err)
    with open(output_file_dir + 'errneg.txt', 'w', encoding='utf-8') as f:
        f.writelines(neg_err)
    with open(output_file_dir + 'erruncertain.txt', 'w', encoding='utf-8') as f:
        f.writelines(uncertain_err)
    with open(output_file_dir + 'errquestion.txt', 'w', encoding='utf-8') as f:
        f.writelines(question_err)


def predict(model_file_path, predict_file, word_list, type_list, save_dir, device):
    model = torch.load(model_file_path, map_location=device)
    model.device = device
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    shutil.copy(model_file_path, save_dir+'param')
    from_input_file(predict_file, save_dir, word_list, model, device)
    pred_input_path = save_dir + 'resutls.txt'
    eval_result(pred_input_path, predict_file, type_list, save_dir)


def predict_all(input_dir, predict_file, word_list, type_list, save_dir, device):
    model_result_list = os.listdir(input_dir)
    for f_n in model_result_list:
        with open(input_dir+f_n + '/best_para.txt', 'r', encoding='utf-8') as f:
            model_file_path = f.readlines()[0]
        predict(model_file_path, predict_file, word_list, type_list, save_dir+f_n+'/', device)

def predict_single(model, word_list, type_list):
    input_text = [input('input: ')]
    data, _ = preprocess(input_text, word_list)
    data = torch.tensor(data)
    with torch.no_grad():
        model.eval()
        data = data.to(device)
        prediction = model(data)
    prediction = prediction.argmax(dim=-1)
    print('predict:', type_list[prediction])
    return


if __name__ == '__main__':
    threshold = 20
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    print(device)
    dir_path = './data/'
    wordlist_file_name = 'word_list_%s.txt' % (threshold)
    with open(dir_path + wordlist_file_name, 'r', encoding='utf-8') as f:
        word_list = [d[:-1] for d in f.readlines()]
    with open(dir_path + 'type_list.pk', 'rb') as f:
        type_list = pickle.load(f)
    #model_save_dir_path = './save/'
    predict_file = './data/xiaoice.log.predict.4label.label.txt'
    save_dir = './eval/'
    #predict_all(model_save_dir_path, predict_file, word_list, type_list, save_dir, device)
    model_file_path = './params/param'
    predict(model_file_path, predict_file, word_list, type_list, save_dir, device)
    




