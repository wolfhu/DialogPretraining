# encoding: utf-8

import sys
import zhconv
import langid

from util import  str_util

strings_to_skip_whole_line = ['暂无歌词', '纯音乐']

if __name__ == '__main__':
  while True:
    whether_to_skip = False
    line = sys.stdin.readline()
    if line:
      line = line.strip()
      if langid.classify(line)[0] != 'zh':
        continue
      if len(line) <= 10:
        continue
      for string in strings_to_skip_whole_line:
        if string in line:
          whether_to_skip = True
          break
      if not whether_to_skip:
        print(line)
    else:
      break
