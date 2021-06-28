# encoding: utf-8

import sys
import langid
import zhconv

from util.str_util import str_extreme_clean, n_gram_overlap, remove_str_in_brackets, contain_blocked_str, is_alphabet

message_repeat_limit = 100
ngram_overlap_limit = 1.0/2.0
message_clean_limit_in_instance = 1.0/3.0  # 最多去掉的message比例

message_count = {}


def count_alpha(string):
    count = 0
    for ch in string:
        if is_alphabet(ch):
            count += 1
    return count


def count_digits(string):
    count = 0
    for ch in string:
        if ch.isdigit():
            count += 1
    return count


if __name__ == '__main__':
    while True:
        line = sys.stdin.readline()
        if line:
            line = line.strip()
            line = zhconv.convert(line, 'zh-cn')
            line = line.replace('\t', '')
            items = line.split('$$$')
            items = [item.strip() for item in items]
            res = []
            for idx, turn in enumerate(items):
                turn = str_extreme_clean(turn)
                turn = remove_str_in_brackets(turn)
                if contain_blocked_str(turn) or langid.classify(turn)[0] != 'zh':
                    break
                if len(turn) > 128 or len(turn) < 10:
                    break
                if '@' in turn:
                    break
                alpha_count = count_alpha(turn)
                digit_count = count_digits(turn)
                if float(max(alpha_count, digit_count)) / len(turn) >= 0.2:
                    break
                message_count.setdefault(turn, 0)
                if message_count[turn] >= message_repeat_limit:
                  break
                message_count[turn] += 1
                if idx >= 1:
                    max_ngram_overlap = max([n_gram_overlap(res[-1], turn, n=1),
                                             n_gram_overlap(res[-1], turn, n=2),
                                             n_gram_overlap(res[-1], turn, n=3)])
                    if max_ngram_overlap > ngram_overlap_limit:
                        break
                if idx >= 2:
                    max_ngram_overlap = max([n_gram_overlap(res[-2], turn, n=1),
                                             n_gram_overlap(res[-2], turn, n=2),
                                             n_gram_overlap(res[-2], turn, n=3)])
                    if max_ngram_overlap > ngram_overlap_limit:
                        break
                res.append(turn)
            if (1 - len(res)/float(len(items))) >= message_clean_limit_in_instance:
              continue
            if len(res) <= 1:
              continue
            line = '\t'.join(res)
            if len(line) <= 10:
                continue
            print(line)
        else:
            break
