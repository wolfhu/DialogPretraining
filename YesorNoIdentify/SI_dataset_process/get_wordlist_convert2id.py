from collections import Counter
from tqdm import tqdm
import pickle


def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    data = [d[:-1] for d in data]
    return data


def count_dataset(data):
    results = Counter()
    for d in tqdm(data):
        results += Counter(d)
    return results


def Convert_word_to_ids(word_list, data):
    result = []
    for l in tqdm(data):
        tmp = [word_list.index('SOS')]
        for w in l:
            if w in word_list:
                tmp.append(word_list.index(w))
            else:
                tmp.append(word_list.index('UNK'))
        tmp.append(word_list.index('EOS'))
        result.append(tmp)
    return result


def Padding_Cutting(data, max_len):
    result = [l[:max_len] for l in data]
    result_len = [len(l) for l in result]
    result = [l+[word_list.index('PAD')]*(max_len-len(l)) if len(l)<max_len else l for l in result]
    return result, result_len

if __name__ == '__main__':
    file_list = ['certain', 'neg', 'question', 'uncertain']
    word_list = Counter()
    dataset = dict()
    for f_n in file_list:
        print(f_n, 'count_word')
        data = read_txt_file('./new_'+f_n+'.txt')
        word_list += count_dataset(data)
        dataset[f_n] = data
    word_list = sorted(word_list.items(), key=lambda x:x[1], reverse=True)

    threshold = 20
    word_list = [x for x in word_list if x[1]>threshold]
    word_list = ['PAD', 'SOS', 'EOS', 'UNK'] + [x[0] for x in word_list]

    result_datas = dict()
    for f_n in file_list:
        print(f_n, 'Convert_word_to_ids')
        data = Convert_word_to_ids(word_list, dataset[f_n])
        data, data_len = Padding_Cutting(data, 28)
        result_datas[f_n] = {'data': data, 'data_len': data_len}
        
    with open('./'+'dataset_%s.pk' % (threshold), 'wb') as f:
        pickle.dump(result_datas, f)
    with open('./'+'word_list_%s.txt' % (threshold), 'w', encoding='utf-8') as f:
        f.writelines([w+'\n' for w in word_list])
    print(1)
