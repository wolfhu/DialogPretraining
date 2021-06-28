# encoding: utf-8

import sys
import pickle

from util.trie import Trie

kg_file = "/home/t-yuniu/xiaoice/yuniu/dataset/kg_data/ownthink_v2.csv"
# Yinge KG
# kg_file = "/home/t-yuniu/xiaoice/yuniu/dataset/KG/EntityId_Predicate_Value_20170512.txt"

kg_pkl_path = '/home/t-yuniu/xiaoice/yuniu/dataset/kg_data/kg.pkl'

if __name__ == '__main__':
    entity_trie = Trie()
    entity_edge_dict = {}
    triple_count = 0
    with open(kg_file, 'r') as kg_f:
        kg_line_count = 0
        while True:
            line = kg_f.readline()
            if line:
                kg_line_count += 1
                items = line.strip().split(',')
                # Yinge KG
                # items = line.strip().split('|||')
                if len(items) != 3:
                    continue
                entity, relation, predict = items
                relation = relation.strip()
                predict = predict.strip()
                entity = entity.split('[')[0]
                entity = entity.strip()
                # Yinge KG
                # entity = entity.split('_')[1]
                # entity = entity.split('(')[0]
                # predict = predict.split('|')[0]
                if len(entity) == 1 or len(predict) == 1:
                    continue
                if entity.isdigit() or predict.isdigit():
                    continue
                if entity in predict or predict in entity\
                        or entity in relation or relation in entity:
                    continue
                entity_trie.insert(entity)
                entity_edge_dict.setdefault(entity, [])
                entity_edge_dict[entity].append((entity, relation, predict))
                triple_count += 1
            else:
                break
            if kg_line_count % 5000000 == 0:
                sys.stdout.write('KG processed %d lines.\n' % (kg_line_count,))
                sys.stdout.flush()
    sys.stdout.write('KG index building complete with %d triples.\n' % (triple_count,))

    with open(kg_pkl_path, 'wb') as kg_pkl_file:
        pickle.dump((entity_trie, entity_edge_dict), kg_pkl_file)
    sys.stdout.write('KG pickle saved.\n')
