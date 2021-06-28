# encoding: utf-8

import sys
import re

import jieba

if __name__ == '__main__':
    while True:
        line = sys.stdin.readline()
        if line:
            line = line.strip()
            query, answer = line.split('\t')
            query = re.sub("\W+", "", query)
            query = query.replace('_', '')
            query = query.replace('\n', '')
            query = query.replace('\r', '')
            answer = re.sub("\W+", "", answer)
            answer = answer.replace('_', '')
            answer = answer.replace('\n', '')
            answer = answer.replace('\r', '')

            print(' '.join(jieba.cut(query)) + '\t' + ' '.join(jieba.cut(answer)))
        else:
            break
