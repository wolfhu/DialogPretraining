# encodign: utf-8
"""
domain chat语料存在空行，所以必须用 readlines() 全部读进来再迭代
"""

import sys

from util import str_util

# illegal_line_path = "/home/t-yuniu/xiaoice/yuniu/dataset/domain/Corpus/all_filtered/all.filtered.txt.illegal"


def remove_tieba_tag(sentence):
    """
    删除贴吧吧名标签
    如：query
    智子为什么说三体比人类更危险？_三体吧_
    :param sentence:
    :return:
    """
    if not sentence[-2:] == '吧_':
        return sentence
    reverse_sentence = sentence[-2::-1]
    another_idx = reverse_sentence.find('_')
    if another_idx == -1:
        return sentence
    return reverse_sentence[another_idx+1:][-1::-1]


if __name__ == '__main__':
    for line in sys.stdin.readlines():
        if line:
            line = line.strip()
            line = line.replace('\r', '')
            line = line.replace('#R#', '')
            line = line.replace('#N#', '')
            line = line.replace('#TAB#', '')
            line = str_util.remove_html_tag(line)
            line = str_util.regexp(line)
            items = line.split('\t')
            if len(items) != 2:
                continue
            query, answer = items
            len_answer = len(answer)
            if query[-len_answer:] == answer:
                continue
            query = remove_tieba_tag(query)
            query = query.replace('_', '')
            query = query.strip()
            answer = remove_tieba_tag(answer)
            answer = answer.replace('_', '')
            answer = answer.strip()
            if len(query) == 0 or len(answer) == 0:
                continue
            print('\t'.join((query, answer,)))
