# encoding: utf-8

import sys

from util import str_util
if __name__ == '__main__':
    while True:
        line = sys.stdin.readline()
        if line:
            line = line.strip()
            key_words, question, answer = line.split('\t')
            question = str_util.remove_blank(question)
            answer = str_util.remove_blank(answer)
            print('\t'.join((key_words, question, answer)))
        else:
            break
