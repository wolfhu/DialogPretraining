# encoding: utf-8

import sys

if __name__ == '__main__':
    while True:
        line = sys.stdin.readline()
        if line:
            line = line.strip()
            turns = line.split('\t')
            for i in range(len(turns)-3):
                print('\t'.join(turns[i:i+4]))
        else:
            break
