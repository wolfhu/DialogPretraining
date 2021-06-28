# encoding: utf-8

import sys
import os
import re
import zhconv

from util.str_util import to_sentences, is_punctuation
from util.zhfiction_util import contain_selected_tag

# dir_path = '/home/t-yuniu/xiaoice/yuniu/dataset/zhfiction/sample/'
dir_path = '/home/t-yuniu/xiaoice/yuniu/dataset/zhfiction/Clean/'

continuous_not_chat_limit = 10  # 连续出现多少次非chat行则截断
min_chat_length = 5


def get_chat(string):
    pattern = u"“.*?”"
    res = re.findall(pattern, string)
    pattern = u"\".*?\""
    res += re.findall(pattern, string)
    if not res:
        return ''
    # if sum([len(elem) for elem in res]) < min_chat_length and not is_punctuation(res[-1][-1]):
    if sum([len(elem[1:-1]) for elem in res]) < min_chat_length and not is_punctuation(res[-1][-2]):
        return ''
    return ''.join([sent[1:-1] for sent in res])


chapter_name_pattern_list = [u'第(零|一|二|三|四|五|六|七|八|九|十|百|千|万){1,}(章|卷)',
                             u'[0-9]{3,}',
                             u'[0-9]{1,}\.',
                             u'[0-9]{1,}、',
                             u'(零|一|二|三|四|五|六|七|八|九|十|百|千|万){1,}、']


def is_chapter_name(string):
    sents = to_sentences(string)
    if len(sents) != 1:
        return False
    for pattern in chapter_name_pattern_list:
        res = re.match(pattern, string)
        if res and res.start() == 0:
            return True
    return False


for filename in os.listdir(dir_path):
    if '.txt' in filename and contain_selected_tag(filename):
        with open(dir_path + filename, 'r') as f:
            res = []
            continuous_not_chat_count = 0  # 连续的非chat行计数
            while True:
                line = f.readline()
                if line:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    if is_chapter_name(line):
                        if len(res) > 0:
                            print('\t'.join(res))
                        res = []
                    line = zhconv.convert(line, 'zh-cn')
                    chat = get_chat(line)
                    if len(chat) == 0:
                        continuous_not_chat_count += 1
                    else:
                        if continuous_not_chat_count >= continuous_not_chat_limit:
                            if len(res) > 0:
                                print('\t'.join(res))
                            res = []
                            continuous_not_chat_count = 0
                        res.append(chat)
                else:
                    break
            if len(res) > 0:
                print('\t'.join(res))
