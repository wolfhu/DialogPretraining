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
      try:
        query, answer = line.split('\t')
      except ValueError:
        sys.stderr.write('error\n')
        continue
      query = query.strip()
      answer = answer.strip()
      if len(answer) > 50:
        continue
      line = '\t'.join((query, answer))
      if len(line) <= 15:
        continue
      print(line)
    else:
      break
