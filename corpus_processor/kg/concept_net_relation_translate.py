# encoding: utf-8

import sys

relation_en = 'kg/concept_net_relation_en'
relation_zh = 'kg/concept_net_relation_zh'

trans_table = {}
if __name__ == '__main__':
    with open(relation_en, 'r') as f_en, open(relation_zh, 'r') as f_zh:
        for en, zh in zip(f_en.readlines(), f_zh.readlines()):
            en = en.strip()
            zh = zh.strip()
            trans_table.setdefault(en, zh)

    while True:
        line = sys.stdin.readline().strip()
        if line:
            relation, entity, predict = line.split('\t')
            relation = trans_table[relation]
            print('\t'.join((relation, entity, predict)))
        else:
            break
