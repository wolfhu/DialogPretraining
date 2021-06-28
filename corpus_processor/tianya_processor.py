# encoding: utf-8

import sys
import json
import langid
import zhconv
import re

from util.str_util import str_extreme_clean, n_gram_overlap, remove_str_in_brackets, contain_blocked_str, is_alphabet

message_repeat_limit = 100
ngram_overlap_limit = 1.0/2.0
message_clean_limit_in_instance = 1.0/3.0  # 最多去掉的message比例

message_count = {}

desc_block = ['rt']

black_list = ['楼上', '楼下', '楼主', '贴图', '图片', '照片', '帖图']


while True:
    line = sys.stdin.readline()
    if line:
        line = line.strip()
        entity = json.loads(line)
        title = entity['Title']
        description = entity['Description']
        comment_list_2d = entity['Comments']
        if contain_blocked_str(title) or contain_blocked_str(description):
            continue
        title = str_extreme_clean(title)
        title = remove_str_in_brackets(title)
        description = str_extreme_clean(description)
        description = remove_str_in_brackets(description)
        if description in desc_block:
            description = ''
        if min(len(title), len(description)) > 0 and n_gram_overlap(title, description, n=2) > ngram_overlap_limit:
            if len(title) > len(description):
                query = title
            else:
                query = description
        else:
            query = ' '.join([title, description])
        query = zhconv.convert(query, 'zh-cn')
        if '@' in query:  # 删除艾特了某个用户的case
            continue
        if len(query) == 0:
            continue
        corpus_in_instance = []
        clean_comment_count = 0.0
        total_comment_count = 0.0
        for comment_list in comment_list_2d:
            clean_comment_list = []
            total_comment_count += len(comment_list)
            for idx, comment in enumerate(comment_list):
                comment = str_extreme_clean(comment)
                comment = remove_str_in_brackets(comment)
                comment = zhconv.convert(comment, 'zh-cn')
                legal_flag = True
                for black in black_list:
                    if black in comment:
                        legal_flag = False
                        break
                if not legal_flag:
                    break
                if '@' in comment:
                    break
                if contain_blocked_str(comment):
                    break
                if langid.classify(comment)[0] != 'zh':
                    break
                if len(comment) > 128 or len(comment) < 10:
                    break
                if 'http://' in comment or 'https://' in comment or 'ftp://' in comment \
                    or 'www.' in comment or '.com' in comment or '.net' in comment \
                    or '.cn' in comment or '.org' in comment or '.gov' in comment \
                    or '.htm' in comment or '.html' in comment:
                    break
                if re.match('[0-9]{7,100}', comment) is not None:
                    break
                alpha_count = 0.0
                num_count = 0.0
                for ch in comment:
                    if is_alphabet(ch):
                        alpha_count += 1
                    if ch.isdigit():
                        num_count += 1
                if max(alpha_count, num_count) / len(comment) > 0.2:
                    break
                message_count.setdefault(comment, 0)
                if message_count[comment] >= message_repeat_limit:
                    break
                message_count[comment] += 1
                if idx >= 1:
                    max_ngram_overlap = max([n_gram_overlap(clean_comment_list[-1], comment, n=1),
                                             n_gram_overlap(clean_comment_list[-1], comment, n=2),
                                             n_gram_overlap(clean_comment_list[-1], comment, n=3)])
                    if max_ngram_overlap > ngram_overlap_limit:
                        break
                if idx >= 2:
                    max_ngram_overlap = max([n_gram_overlap(clean_comment_list[-2], comment, n=1),
                                             n_gram_overlap(clean_comment_list[-2], comment, n=2),
                                             n_gram_overlap(clean_comment_list[-2], comment, n=3)])
                    if max_ngram_overlap > ngram_overlap_limit:
                        break
                clean_comment_list.append(comment)
            if len(clean_comment_list) == 0:
                continue
            clean_comment_count += len(clean_comment_list)
            corpus_in_instance.append('\t'.join([query] + clean_comment_list))
        if (1 - clean_comment_count/total_comment_count) >= message_clean_limit_in_instance:
            continue
        for line in corpus_in_instance:
            print(line)
    else:
        break
