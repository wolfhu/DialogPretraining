# encoding: utf-8

import sys
import pickle

import jieba
import jieba.posseg as pseg

kg_pkl_path = '/home/t-yuniu/xiaoice/yuniu/dataset/kg_data/kg.pkl'
pmi_pre_cal_path = '/home/t-yuniu/xiaoice/yuniu/dataset/domain/Corpus/pmi_pre_cal.pkl'

user_dict_path = '/home/t-yuniu/xiaoice/yuniu/dataset/domain/Keywords/keywords.dict'
jieba.load_userdict(user_dict_path)


def pmi_sentence_to_keyword(cutted_seq, keyword):
    pmi = 0.0
    discount = 0
    for word in cutted_seq:
        if word == keyword:
            pmi += len(cutted_seq) * 10.0
        elif word not in pmi_dict[keyword]:
            discount += 1
            continue
        else:
            pmi += pmi_dict[keyword][word]
    available_word_count = len(cutted_seq) - discount
    if available_word_count <= 0:
        return float('-inf')
    pmi /= available_word_count
    return pmi


if __name__ == '__main__':
    with open(kg_pkl_path, 'rb') as kg_pkl_file:
        _, entity_edge_dict = pickle.load(kg_pkl_file)
    sys.stderr.write('loaded kg.\n')

    with open(pmi_pre_cal_path, 'rb') as pmi_pre_cal_file:
        pmi_dict = pickle.load(pmi_pre_cal_file)
    sys.stderr.write('loaded pmi data.\n')

    while True:
        line = sys.stdin.readline().strip()
        if line:
            items = line.split('\t')
            if len(items) != 2:
                # sys.stdout.write(line + '\n')
                continue
            query, answer = items
            if len(query) > 50:
                continue
            cutted_sentence = jieba.lcut(' '.join((query, answer,)))
            # sys.stderr.write('line: ' + line+'\n')
            for word, tag in list(set(pseg.lcut(query))):
                if tag[0] == 'n' and word in entity_edge_dict:
                    # sys.stderr.write('word: %s, tag: %s\n' % (word, tag))
                    max_triple = None
                    max_pmi = float('-inf')
                    kg_list = list(set(entity_edge_dict[word]))
                    for _, relation, predict in kg_list:
                        curr_pmi = 0.0
                        relation_pmi = None
                        if relation in pmi_dict:
                            relation_pmi = pmi_sentence_to_keyword(cutted_sentence, relation)
                            # sys.stderr.write('relation: %s, pmi: %.2f\n' % (relation, relation_pmi))
                        predict_pmi = None
                        if predict in pmi_dict:
                            predict_pmi = pmi_sentence_to_keyword(cutted_sentence, predict)
                            # sys.stderr.write('predict: %s, pmi: %.2f\n' % (predict, predict_pmi))
                        if relation_pmi is None and predict_pmi is None:
                            curr_pmi = float('-inf')
                        elif relation_pmi is None:
                            curr_pmi = predict_pmi
                        elif predict_pmi is None:
                            curr_pmi = relation_pmi
                        else:
                            curr_pmi = (relation_pmi + predict_pmi) / 2
                        if curr_pmi > max_pmi:
                            max_triple = [word, relation, predict, '%.2f' % curr_pmi]
                            max_pmi = curr_pmi
                    if not max_triple:
                        # sys.stderr.write('not recalled: ' + line + '\n')
                        continue
                    print('\t'.join([' '.join(max_triple), query, answer]))
                    sys.stdout.flush()
        else:
            break
