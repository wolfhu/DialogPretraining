# encoding: utf-8

import pickle
import sys
import jieba
import time
import math

# from util.pmi import pmi_sentence_to_keyword

user_dict_path = '/home/t-yuniu/xiaoice/yuniu/dataset/domain/Keywords/keywords.dict'
jieba.load_userdict(user_dict_path)

pmi_reverse_index_path = '/home/t-yuniu/xiaoice/yuniu/dataset/domain/Corpus/pmi_inverse_index.pkl'
with open(pmi_reverse_index_path, 'rb') as pmi_rev_index_file:
    rev_idx = pickle.load(pmi_rev_index_file)
sys.stderr.write('loaded pmi index.\n')
print(len(rev_idx))
string = 'c罗印象最深的是任意球，过人真心不怎么样，过人还是看大罗吧 啥意思'


def pmi_word_to_word(word1, word2):
    word_num = float(len(rev_idx))
    doc_list_1 = rev_idx[word1]
    doc_list_2 = rev_idx[word2]
    p1 = float(len(doc_list_1)) / word_num
    p2 = float(len(doc_list_2)) / word_num
    i = 0
    j = 0
    co_occur_count = 0.1
    while i < len(doc_list_1) and j < len(doc_list_2):
        if doc_list_1[i] == doc_list_2[j]:
            co_occur_count += 1
            i += 1
            j += 1
        elif doc_list_1[i] < doc_list_2[j]:
            i += 1
        else:
            j += 1
    p_co_occur = co_occur_count / (word_num**2)
    # print(p_co_occur)
    # print(p1)
    # print(p2)
    # print()
    return math.log(p_co_occur / (p1 * p2))


def pmi_sentence_to_keyword(sentence, keyword):
    return sum([pmi_word_to_word(word, keyword) for word in sentence]) / len(sentence)


seq = jieba.lcut(string)
print(seq)
start = time.time()

# print(pmi_sentence_to_keyword(seq, '印象'))
# print(pmi_sentence_to_keyword(seq, '任意球'))
# print(pmi_sentence_to_keyword(seq, '足球'))
# print(pmi_word_to_word('印象', '足球'))
pmi = 0.0
for word in seq:
    start1 = time.time()
    pmi += pmi_word_to_word(word, '印象')
    print('stage time: ' + str(time.time() - start1))
pmi /= len(seq)
print(pmi)

end = time.time()
print('Elapsed time: ' + str(end-start))
