# encoding: utf-8

import sys
import zhconv

from util import  str_util


def remove_blank(s):
    res = ""
    for ch in s:
        if ch != " ":
            res += ch
    return res


if __name__ == '__main__':
    for line in sys.stdin.readlines():
        line = line.strip()
        line = zhconv.convert(line, 'zh-cn')
        line = remove_blank(line)
        line = str_util.regexp(line)
        if len(line) <= 3:
            continue
        print(line)
