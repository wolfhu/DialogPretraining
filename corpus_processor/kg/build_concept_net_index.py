# encoding: utf-8

import sys
import pickle

from util.trie import Trie

concept_net_kg_file = '/home/t-yuniu/xiaoice/yuniu/dataset/ConceptNet/' \
                      'zh_filtered_remove_repete_translated'

kg_pkl_path = '/home/t-yuniu/xiaoice/yuniu/dataset/ConceptNet/concept_net.pkl'

if __name__ == '__main__':
    entity_trie = Trie()
    entity_edge_dict = {}
    triple_count = 0
    with open(concept_net_kg_file, 'r') as kg_f:
        while True:
            line = kg_f.readline().strip()
            if line:
                relation, entity, predict = line.split('\t')
                if len(entity) == 1 or len(predict) == 1:
                    continue
                if entity in predict or predict in entity:
                    continue
                entity_trie.insert(entity)
                entity_edge_dict.setdefault(entity, [])
                entity_edge_dict[entity].append((entity, relation, predict))
                triple_count += 1
            else:
                break
    sys.stdout.write('KG index building complete with %d triples.\n' % (triple_count,))

    with open(kg_pkl_path, 'wb') as kg_pkl_file:
        pickle.dump((entity_trie, entity_edge_dict), kg_pkl_file)
    sys.stdout.write('KG pickle saved.\n')
