# encoding: utf-8

import sys, os
import zhconv
import re, random
from util.str_util import str_extreme_clean
from multiprocessing import Pool
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
  filter_dict = {'楼主': 1, '顶楼': 1, '顶帖': 1, '豆油': 1, '豆邮': 1, '此帖': 1, '此楼': 1, 'lz':1}
  for special_word in filter_dict.keys():
    if special_word in line_q or special_word in line_a:
      return True
  return False

def stdin_test():
  while True:
    line = sys.stdin.readline().strip()
    if line:
      line = zhconv.convert(line, 'zh-cn')
      line = line.split('\t')
      query = line[1]
      answer = line[2]
      query = zhconv.convert(query, 'zh-cn')
      answer = zhconv.convert(answer, 'zh-cn')
      query = str_extreme_clean(query)
      answer = str_extreme_clean(answer)
      if if_include_spec_word(query, answer):
        continue
      if len(query) + len(answer) <= 10:
        continue
      print('\t'.join([query, answer]))
    else:
      break


def file_test(filename):
  print("Working on "+ filename)
  file = open(filename,encoding="utf-8")
  fw = open(filename+"_fliter", "w",encoding="utf-8")
  for line in file:
      line = zhconv.convert(line, 'zh-cn')
      line = line.split('\t')
      query = line[1]
      answer = line[2]
      query = zhconv.convert(query, 'zh-cn')
      answer = zhconv.convert(answer, 'zh-cn')
      query =str_extreme_clean(query)
      answer = str_extreme_clean(answer)
      if if_include_spec_word(query, answer):
        continue
      if len(query) + len(answer) <= 10:
        continue
      fw.write('\t'.join([query, answer]))
      fw.write('\n')

def filelist_test():
  file_list = [r"D:\user\yuwu1\douban\wb\douban.talk.0.wb.id.tsv",
  r"D:\user\yuwu1\douban\wb\douban.talk.1.wb.id.tsv",
  r"D:\user\yuwu1\douban\wb\douban.talk.2.wb.id.tsv",
  r"D:\user\yuwu1\douban\wb\douban.talk.3.wb.id.tsv"]
  fw = open(r"D:\user\yuwu1\douban\wb\sample.txt","w",encoding="utf-8")
  for f in file_list:
    file = open(f,encoding="utf-8")
    for line in file:
        line = zhconv.convert(line, 'zh-cn')
        line = line.split('\t')
        query = line[1]
        answer = line[2]
        query = zhconv.convert(query, 'zh-cn')
        answer = zhconv.convert(answer, 'zh-cn')
        query =str_extreme_clean(query)
        answer = str_extreme_clean(answer)
        if if_include_spec_word(query, answer):
          continue
        if len(query) + len(answer) <= 10:
          continue
        fw.write('\t'.join([query, answer]))
        fw.write("\n")


def multi_process_clean(thread_count=1):
  """
  split data
  """
  base_folder = r"D:\user\yuwu1\douban\wb"
  base_tmp_folder = os.path.join(base_folder,"tmp")
  

  file_list = [r"D:\user\yuwu1\douban\wb\douban.talk.0.wb.id.tsv",
  r"D:\user\yuwu1\douban\wb\douban.talk.1.wb.id.tsv",
  r"D:\user\yuwu1\douban\wb\douban.talk.2.wb.id.tsv",
  r"D:\user\yuwu1\douban\wb\douban.talk.3.wb.id.tsv"]
  fw_list = []
  for c in range(thread_count):
    fw = open(os.path.join(base_tmp_folder,str(c)+".txt"),"w",encoding="utf-8")
    fw_list.append(fw)

  for f in file_list:
    file = open(f,encoding="utf-8")
    for line in file:
      file_to_write = random.randint(0,thread_count - 1) % thread_count
      fw_list[file_to_write].write(line)


  """
  pre-process data
  """
  p = Pool(thread_count)
  for i in range(thread_count):
    p.apply_async(file_test, (os.path.join(base_tmp_folder,str(i)+".txt"),))

  p.close()
  p.join()

  """
  merge data
  """
  fw = open(r"D:\user\yuwu1\douban\wb\merge.txt","w",encoding="utf-8")
  for i in range(thread_count):
    f = open(os.path.join(base_tmp_folder,str(i)+".txt" +"_fliter"),encoding="utf-8")
    for line in f:
      fw.write(line)
    #p.apply_async(file_test, (
if __name__ == '__main__':

  multi_process_clean(60)
  #filelist_test()
