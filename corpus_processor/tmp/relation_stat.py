# encoding: utf-8

import sys

if __name__ == '__main__':
    relation_dict = {}
    while True:
        line = sys.stdin.readline().strip()
        if line:
            relation = line.split('\t')[0]
            relation_dict.setdefault(relation, True)
        else:
            break
    for relation in relation_dict:
        print(relation)
