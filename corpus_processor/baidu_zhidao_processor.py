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
      line = line.replace('#R#', '')
      line = line.replace('#N#', '')
      line = line.replace('#TAB#', '')
      line = str_util.regexp(line)
      items = line.split('\t')
      items = [item.strip() for item in items]
      line = '\t'.join(items)
      if len(line) <= 15:
        continue
      print(line)
    else:
      break
