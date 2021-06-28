# encoding: utf-8

import sys

from util import str_util

if __name__ == '__main__':
    for line in sys.stdin.readlines():
        if line:
            line = line.strip()
            line = line.replace('\r', '')
            line = line.replace('#R#', '')
            line = line.replace('#N#', '')
            line = line.replace('#TAB#', '')
            line = str_util.remove_html_tag(line)
            line = str_util.regexp(line)
            items = line.split('\t')
            if len(items) != 2:
                continue
            query, answer = items
            query = query.strip()
            answer = answer.strip()
            if len(query) == 0 or len(answer) == 0:
                continue
            print('\t'.join((query, answer,)))
