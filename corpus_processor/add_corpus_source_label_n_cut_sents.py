# encoding: utf-8

"""
douban & weibo: 1
baidu_zhidao: 2
wiki & sogou: 3
"""

import sys
from util import str_util

corpus_source_label = sys.argv[1]  # 语料来源标签
sys.stderr.write("corpus_source_label: %s\n" % corpus_source_label)
if corpus_source_label != '2':
  split_to_multi_sents = False  # 当文本可以被切分为多行时，设为True，否则False
else:
  split_to_multi_sents = True
max_sent_len = 512


if __name__ == '__main__':
  while True:
    res = ''
    line = sys.stdin.readline()
    if line:
      line = line.strip()
      if split_to_multi_sents:
        sents = str_util.split_sentence(line)
        for sent in sents:
          res += sent
          if len(res) >= max_sent_len:
            print(corpus_source_label + '\t' + res)
            res = ''
        if len(res) > 0:
          print(corpus_source_label + '\t' + res)
      else:
        print(corpus_source_label + '\t' + line)
    else:
      break
