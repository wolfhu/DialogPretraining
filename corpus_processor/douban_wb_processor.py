# encoding: utf-8

import sys
import zhconv
import re


def regexp(sentence):
  # url
  regexp_url = u"(https?|ftp|file|ttp)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]|www.(.*?).+(com|cn|org|htm|html)"
  sentence = re.sub(regexp_url, u'tgurltg', sentence)
  # email
  regexp_email = u'[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)+|[0-9]+qqcom|([0-9]+qq com)|([0-9]+ qq com)'
  sentence = re.sub(regexp_email, u'tgemailtg', sentence)
  # phone num
  regexp_phone = u"1[3|4|5|7|8][0-9]{9}|0\d{2}-\d{8}|0\d{3}-\d{7}0\d{2}\d{8}|0\d{3}\d{7}"
  sentence = re.sub(regexp_phone, u'tgphonetg', sentence)
  return sentence


def if_include_spec_word(line_q, line_a):
  # filter special data containing 楼主，顶楼，顶帖，豆油，豆邮，此楼，此帖
  filter_dict = {'楼主': 1, '顶楼': 1, '顶帖': 1, '豆油': 1, '豆邮': 1, '此帖': 1, '此楼': 1}
  for special_word in filter_dict.keys():
    if special_word in line_q or special_word in line_a:
      return True
  return False


if __name__ == '__main__':
  while True:
    line = sys.stdin.readline().strip()
    if line:
      line = zhconv.convert(line, 'zh-cn')
      line = line.split('\t')
      query = line[1]
      answer = line[2]
      query = zhconv.convert(query, 'zh-cn')
      answer = zhconv.convert(answer, 'zh-cn')
      query = regexp(query)
      answer = regexp(answer)
      if if_include_spec_word(query, answer):
        continue
      if len(query) + len(answer) <= 10:
        continue
      print('\t'.join([query, answer]))
    else:
      break
