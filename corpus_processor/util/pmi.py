# encoding: utf-8

import sys
import math
import pickle

import jieba


def build_reverse_index(file_path, index_path, userdict_path=None):
    rev_idx = {}
    if userdict_path:
        jieba.load_userdict(userdict_path)
    with open(file_path, 'r') as corpus_f, \
        open(index_path, 'wb') as index_f:
        doc_idx = 0
        while True:
            line = corpus_f.readline().strip()
            if line:
                for word in jieba.cut(line):
                    rev_idx.setdefault(word, [])
                    rev_idx[word].append(doc_idx)
                doc_idx += 1
            else:
                break
        pickle.dump(rev_idx, index_f)


def pmi_word_to_word(word1, word2, rev_idx):
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


def pmi_sentence_to_keyword(sentence, keyword, rev_idx):
    return sum([pmi_word_to_word(word, keyword, rev_idx) for word in sentence]) / len(sentence)


if __name__ == '__main__':
    file_path, index_path, userdict_path = sys.argv[1:4]
    build_reverse_index(file_path, index_path, userdict_path)
