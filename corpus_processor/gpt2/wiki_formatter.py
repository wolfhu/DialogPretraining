# encoding: utf-8

import sys

from util.str_util import to_sentences
max_length = 1024
min_length = 512

while True:
    line = sys.stdin.readline()
    res = ''
    if line:
        line = line.strip()
        paragraph_list = line.split('\t')
        for paragraph in paragraph_list:
            for sent in to_sentences(paragraph):
                if len(res) + len(sent) <= max_length:
                    res += sent
                else:
                    if len(res) >= min_length:
                        print(res[:max_length])
                    res = ''
                    res += sent
            if len(res) > 0:
                res += '\t'
        res = res.strip()
        if len(res) >= min_length:
            print(res[:max_length])
    else:
        break
