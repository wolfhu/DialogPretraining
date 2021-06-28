# encoding: utf-8

import sys
import pickle

from util.detect_tag import detect_tag

kg_pkl_path = '/home/t-yuniu/xiaoice/yuniu/dataset/kg_data/kg.pkl'


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
            entities = detect_tag(query, entity_trie)
            if len(entities) == 0:
                continue
            entities = sorted(entities, key=lambda item: len(item), reverse=True)
            edge = None
            for entity in entities:
                sub_graph = entity_edge_dict[entity]
                for _, relation, predict in sub_graph:
                    if predict in answer and len(answer) > 2*len(predict):
                        edge = (entity, relation, predict)
                        break
                    # if (relation in query and predict in answer) and len(answer) > len(predict):
                    #     edge = (entity, relation, predict)
                    #     break
                if edge:
                    break
            if edge:
                print('\t'.join([' '.join(edge), query, answer]))
        else:
            break
