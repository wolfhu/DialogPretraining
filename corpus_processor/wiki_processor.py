# encoding: utf-8

import sys
import zhconv

from util import str_util

if __name__ == '__main__':
    for line in sys.stdin.readlines():
        line = line.strip()
        line = zhconv.convert(line, 'zh-cn')
        line = line.replace('\t', '')
        line = str_util.regexp(line)
        if len(line) <= 10:
            continue
        print(line)
