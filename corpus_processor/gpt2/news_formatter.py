# encoding: utf-8

import sys

from util.str_util import to_sentences

max_length = 1024
min_length = 512

if __name__ == '__main__':
    while True:
        line = sys.stdin.readline().strip()
        if line:
            sents = to_sentences(line)
            res = ''
            curr_length = 0
            for sent in sents:
                if curr_length + len(sent) <= max_length:
                    res += sent
                    curr_length += len(sent)
            if len(res) < min_length:
                continue
            else:
                print(res)
        else:
            break
