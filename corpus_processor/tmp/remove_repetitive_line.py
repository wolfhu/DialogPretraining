# encoding: utf-8

import sys

if __name__ == '__main__':
    line_dict = {}
    while True:
        line = sys.stdin.readline().strip()
        if line:
            line_dict.setdefault(line, True)
        else:
            break
    for line in line_dict:
        print(line)
