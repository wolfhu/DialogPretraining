# encoding: utf-8

import sys
import argparse

from util.str_util import to_sentences

parser = argparse.ArgumentParser()

parser.add_argument("--min_len", default=512, type=int, help="minimum length of single sample")
parser.add_argument("--max_len", default=1024, type=int, help="maximum length of single sample")

args = parser.parse_args()

max_length = args.max_len
min_length = args.min_len
# DEBUG
# print("max length: %d" % max_length)
# print("min length: %d" % min_length)


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
