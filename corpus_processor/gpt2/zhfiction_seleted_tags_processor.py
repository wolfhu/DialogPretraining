# encoding: utf-8

import sys
import os
import re

from util.str_util import str_extreme_clean, remove_str_in_square_brackets, remove_str_in_brackets,\
    is_chinese, is_alphabet, remove_book_title_mark
from util.zhfiction_util import contain_selected_tag

line_min_length = 10
min_chinese_rate = 0.5
# dir_path = '/home/t-yuniu/xiaoice/yuniu/dataset/zhfiction/sample/'
dir_path = '/home/t-yuniu/xiaoice/yuniu/dataset/zhfiction/Clean/'

text_block_patterns = [u'内容版权归作者所有',
                       u'全文：',
                       u'作者：',
                       u'书名：',
                       u'本书类别：{0,}',
                       u'更新时间：{0,}',
                       u'本章字数：{0,}',
                       u'潇湘书院',
                       u'正文完结',
                       u'文案：',
                       u'QQ：',
                       u'VIP',
                       u'阅读[0-9]{1,}人',
                       u'收藏[0-9]{1,}人',
                       u'新文开坑',
                       u'欢迎跳坑',
                       u'多多支持',
                       u'第一次写',
                       u'古代文',
                       u'小说网',]


def remove_text_block(string):
    for block_pattern in text_block_patterns:
        string = re.sub(block_pattern, u'', string)
    return string


def remove_chapter_name(string):
    pattern = u'第(零|一|二|三|四|五|六|七|八|九|十|百|千|万|){1,}(章|卷)'
    string = re.sub(pattern, u'', string)
    return string


def non_zh_or_alpha_count(string):
    count = 0
    for ch in string:
        if is_chinese(ch) or is_alphabet(ch):
            count += 1
    return len(string)-count


for filename in os.listdir(dir_path):
    if '.txt' in filename and contain_selected_tag(filename):
        with open(dir_path + filename, 'r') as f:
            res = []
            while True:
                line = f.readline()
                if line:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    # if len(line) < line_min_length:
                    #     continue
                    # if chinese_rate(line) < min_chinese_rate:
                    #     continue
                    line = line.replace('\t', '')
                    raw = line
                    line = str_extreme_clean(line)
                    line = remove_text_block(line)
                    line = remove_chapter_name(line)
                    if len(line) != len(raw):
                        continue
                    line = remove_str_in_square_brackets(line)
                    line = remove_str_in_brackets(line)
                    line = remove_book_title_mark(line)
                    illegal_ch_count = non_zh_or_alpha_count(line)
                    # illegal_ch_count = 0
                    if len(line)-illegal_ch_count <= float(len(raw))*0.6:
                        continue
                    # line = remove_str_in_brackets(line)
                    # if len(line) < line_min_length:
                    #     continue
                    res.append(line)
                else:
                    break
            if len(res) > 0:
                print('\t'.join(res))
