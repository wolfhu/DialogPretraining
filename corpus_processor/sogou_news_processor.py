# encoding: utf-8

import sys
import zhconv

from util import str_util


if __name__ == '__main__':
    while True:
      line = sys.stdin.readline()
      if line:
        line = line.strip()
        line = zhconv.convert(line, 'zh-cn')
        line = line.replace('<content>', '')
        line = line.replace('</content>', '')
        line = str_util.strQ2B(line)
        line = str_util.regexp(line)
        if len(line) <= 15:
          continue
        print(line)
      else:
        break
