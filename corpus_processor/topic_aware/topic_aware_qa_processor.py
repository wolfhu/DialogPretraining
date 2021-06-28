# encoding: utf-8
"""
从单轮douban语料生产带keywords单轮对话语料
用tf-idf抽取keywords
"""

import sys

import jieba.analyse

allowPos = ('n', 'nr', 'nr1', 'nr2', 'nrj', 'nrf', 'ns', 'nsf', 'nt', 'nz', 'nl', 'ng',
            'v', 'vd', 'vn', 'vshi', 'vyou', 'vf', 'vx', 'vi', 'vl', 'vg', 'a', 'ad',
            'an', 'ag', 'al')

if __name__ == '__main__':
    while True:
        line = sys.stdin.readline().strip()
        if line:
            try:
                (query, answer) = line.split('\t')
            except ValueError as e:
                sys.stderr.write('illegal line.\n')
                continue
            tmp_keywords = jieba.analyse.extract_tags(answer, allowPOS=allowPos)
            keywords = []
            for keyword in tmp_keywords:
                if len(keyword) > 1 and not keyword.encode('utf-8').isalpha():
                    keywords.append(keyword)
            if len(keywords) <= 0:
                continue
            print('\t'.join((' '.join(keywords), query, answer,)))
        else:
            break
