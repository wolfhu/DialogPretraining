# encoding: utf-8

import sys
import zhconv

from util import str_util


if __name__ == '__main__':
    for line in sys.stdin.readlines():
        line = line.strip()
        line = zhconv.convert(line, 'zh-cn')
        line = line.replace('*', '')
        line = line.replace('>', '')
        line = line.replace('_', '')
        line = line.replace('(ZhihuV1)', '')
        line = line.replace('(ZhihuV2)', '')
        line = str_util.regexp(line)
        if len(line) <= 10:
            continue
        line += "\t" + '2'
        print(line)
