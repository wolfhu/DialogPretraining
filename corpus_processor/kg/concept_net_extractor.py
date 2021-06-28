# encoding: utf-8
"""
NOTE:
    存在重复的三元组，这里没有去掉。
    ConceptNet格式是relation, entity, predict，其他KG是entity, relation, predict
"""

import sys

if __name__ == '__main__':
    while True:
        line = sys.stdin.readline().strip()
        if line:
            items = line.split('\t')
            relation = items[1]
            entity = items[2]
            predict = items[3]
            relation = relation.split('/')[-1]
            entity_lang, entity = entity.split('/')[2:4]
            predict_lang, predict = predict.split('/')[2:4]
            if entity_lang == 'zh' or predict_lang == 'zh':
                print('\t'.join((relation, entity, predict,)))
        else:
            break
