# encoding: utf-8

import sys
import json
import langid
import zhconv
import re

from util.str_util import str_extreme_clean, n_gram_overlap, remove_str_in_brackets, contain_blocked_str, \
    alpha_rate, digit_rate

ngram_overlap_limit = 1.0/2.0
alpha_rate_limit = 0.2
digit_rate_limit = 0.2

desc_block = ['rt']

black_list = ['楼上', '楼下', '楼主', '贴图', '图片', '照片', '帖图']


def remove_esc(string):
    string = string.replace('\r', '')
    string = string.replace('\n', '')
    string = string.replace('\t', '')
    string = re.sub(u"-{3,}", u"", string)
    return string


while True:
    line = sys.stdin.readline()
    if line:
        line = line.strip()
        entity = json.loads(line)
        title = entity['title']
        description = entity['desc']
        comment = entity['content']
        # Filter Query
        if contain_blocked_str(title) or contain_blocked_str(description):
            continue
        title = str_extreme_clean(title)
        title = remove_str_in_brackets(title)
        title = remove_esc(title)
        description = str_extreme_clean(description)
        description = remove_str_in_brackets(description)
        description = remove_esc(description)
        if description in desc_block:
            description = ''
        if len(title) < 5:
            title = ''
        if len(description) < 5:
            description = ''
        if min(len(title), len(description)) > 0 and n_gram_overlap(title, description, n=2) > ngram_overlap_limit:
            if len(title) > len(description):
                query = title
            else:
                query = description
        else:
            query = ' '.join([title, description])
        if len(query) > 1024 or len(query) < 12:
            continue
        query = zhconv.convert(query, 'zh-cn')
        if alpha_rate(query) > alpha_rate_limit or digit_rate(query) > digit_rate_limit:
            continue
        if '@' in query:  # 删除艾特了某个用户的case
            continue
        if len(query) == 0:
            continue
        # Filter Answer
        comment = str_extreme_clean(comment)
        comment = remove_str_in_brackets(comment)
        comment = remove_esc(comment)
        comment = zhconv.convert(comment, 'zh-cn')
        legal_flag = True
        for black in black_list:
            if black in comment:
                legal_flag = False
                break
        if not legal_flag:
            continue
        if '@' in comment:
            continue
        if contain_blocked_str(comment):
            continue
        if langid.classify(comment)[0] != 'zh':
            continue
        if len(comment) > 1024 or len(comment) < 10:
            continue
        if 'http://' in comment or 'https://' in comment or 'ftp://' in comment \
                or 'www.' in comment or '.com' in comment or '.net' in comment \
                or '.cn' in comment or '.org' in comment or '.gov' in comment \
                or '.htm' in comment or '.html' in comment:
            continue
        if re.match('[0-9]{7,100}', comment) is not None:
            continue
        if alpha_rate(comment) > alpha_rate_limit or digit_rate(comment) > digit_rate_limit:
            continue
        # Filter final corpus line
        corpus = '\t'.join([query, comment])
        if len(corpus) > 1024 or len(corpus) < 10:
            continue
        print(corpus)
    else:
        break
