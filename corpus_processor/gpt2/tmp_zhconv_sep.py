# encoding: utf-8

import sys
import zhconv

while True:
    line = sys.stdin.readline()
    if line:
        line = line.strip()
        line = zhconv.convert(line, 'zh-cn')
        line = line.replace('\t', ' [SEP] ')
        print(line)
    else:
        break
