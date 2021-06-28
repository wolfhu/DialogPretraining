# encoding: utf-8

import sys
import pickle

from util.trie import Trie

kg_pkl_path = '/home/t-yuniu/xiaoice/yuniu/dataset/ConceptNet/concept_net.pkl'


def detect_tag(sentence):
    """
    Judge if sentence contain as least a tag.
    :param sentence: query or answer
    :return: boolean, True if contain, False otherwise.
    """
    length = len(sentence)
    detected_tags = []
    for idx in range(length):
        node = entity_trie.lookup
        idx_tmp = idx
        while True:
            if idx_tmp >= length:
                break
            if sentence[idx_tmp] in node:
                node = node[sentence[idx_tmp]]
                idx_tmp += 1
                if Trie.END in node:
                    detected_tags.append(sentence[idx:idx_tmp])
            else:
                break
    return detected_tags


if __name__ == '__main__':
    with open(kg_pkl_path, 'rb') as kg_pkl_file:
        entity_trie, entity_edge_dict = pickle.load(kg_pkl_file)

    while True:
        line = sys.stdin.readline()
        if line:
            line = line.strip()
            items = line.split('\t')
            if len(items) != 2:
                # sys.stdout.write(line + '\n')
                continue
            query, answer = items
            if len(query) > 50:
                continue
            entities = detect_tag(query)
            if len(entities) == 0:
                continue
            edge = None
            for entity in entities:
                sub_graph = entity_edge_dict[entity]
                for _, relation, predict in sub_graph:
                    if predict in answer and len(answer) > 2*len(predict):
                        edge = (entity, relation, predict)
                        break
                if edge:
                    break
            if edge:
                print('\t'.join([' '.join(edge), query, answer]))
        else:
            break
