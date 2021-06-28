# encoding: utf-8

import sys
import pickle
import jieba
import math
import time

user_dict_path = '/home/t-yuniu/xiaoice/yuniu/dataset/domain/Keywords/keywords.dict'
jieba.load_userdict(user_dict_path)

if __name__ == '__main__':
    co_occur_dict = {}
    df_dict = {}
    start = time.time()
    while True:
        line = sys.stdin.readline().strip()
        if line:
            seq = jieba.lcut(line)
            seq = set(seq)
            seq = list(seq)
            for idx, word1 in enumerate(seq):
                df_dict.setdefault(word1, 0.0)
                df_dict[word1] += 1
                for word2 in seq[idx+1:]:
                    co_occur_dict.setdefault(word1, {})
                    co_occur_dict[word1].setdefault(word2, 0.1)
                    co_occur_dict[word1][word2] += 1
                    co_occur_dict.setdefault(word2, {})
                    co_occur_dict[word2].setdefault(word1, 0.1)
                    co_occur_dict[word2][word1] += 1
        else:
            break
    sys.stderr.write('co-occurrence calculate complete.\n')

    for word1 in co_occur_dict:
        for word2 in co_occur_dict[word1]:
            co_occur_dict[word1][word2] /= (df_dict[word1] * df_dict[word2])
            co_occur_dict[word1][word2] = math.log(co_occur_dict[word1][word2])
    sys.stderr.write('PMI calculate complete.\n')

    end = time.time()
    sys.stderr.write('Elapsed time: ' + str(end - start) + '\n')

    pmi_pre_cal_path = '/home/t-yuniu/xiaoice/yuniu/dataset/domain/Corpus/pmi_pre_cal.pkl'
    with open(pmi_pre_cal_path, 'wb') as pmi_pre_cal_file:
        pickle.dump(co_occur_dict, pmi_pre_cal_file)
