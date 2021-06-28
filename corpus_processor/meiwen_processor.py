# encoding: utf-8

import sys
import zhconv
import json
import langid

from util import str_util


def remove_str_in_brackets(string, bracket_left, bracklet_right):
  res = ''
  in_bracket = False
  for ch in string:
    if ch == bracket_left:
      in_bracket = True
    elif ch == bracklet_right:
      in_bracket = False
    if not in_bracket and ch != bracklet_right:
      res += ch
  return res


strs_to_remove = ['文章阅读网：', '首发散文网：', '\n', '\r', '\t']

if __name__ == '__main__':
  while True:
    line = sys.stdin.readline()
    if line:
      line = line.strip()
      content = json.loads(line)['TextContent']
      content = str_util.regexp(content)
      for string in strs_to_remove:
        content = content.replace(string, '')
      if langid.classify(content)[0] != 'zh':
        continue
      content = str_util.remove_blank(content)
      content = zhconv.convert(content, 'zh-cn')
      content = remove_str_in_brackets(content, '(', ')')
      content = remove_str_in_brackets(content, '（', '）')
      content = remove_str_in_brackets(content, '【', '】')
      if len(content) <= 10:
        continue
      print(content)
    else:
      break
