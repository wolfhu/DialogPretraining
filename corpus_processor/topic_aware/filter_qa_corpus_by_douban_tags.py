# encoding: utf-8

import sys
import json

from util.trie import Trie

douban_tag_file_path = '/home/t-yuniu/xiaoice/yuniu/dataset/douban_title/douban_title.json'

tag_black_dict = {}
tag_black_dict.setdefault('游戏', True)

tag_trie = Trie()


def detect_tag(sentence):
    """
    Judge if sentence contain as least a tag.
    :param sentence: query or answer
    :return: boolean, True if contain, False otherwise.
    """
    length = len(sentence)
    detected_tags = []
    for idx in range(length):
        node = tag_trie.lookup
        idx_tmp = idx
        while True:
            if idx_tmp >= length:
                break
            if sentence[idx_tmp] in node:
                node = node[sentence[idx_tmp]]
                idx_tmp += 1
                if Trie.END in node:
                    detected_tags.append(sentence[idx:idx_tmp])
            else:
                break
    return detected_tags


if __name__ == '__main__':
    # build trie from tag file
    with open(douban_tag_file_path) as douban_tag_file:
        for line in douban_tag_file.readlines():
            line = line.strip()
            tags = json.loads(line)['Tag']
            for tag in tags:
                if len(tag) == 1 or tag in tag_black_dict:
                    continue
                tag_trie.insert(tag)
    # filter corpus contain tags
    while True:
        line = sys.stdin.readline().strip()
        if line:
            try:
                line = line.replace('#', '')
                query, answer = line.split('\t')[:2]
                detected_tags = detect_tag(query)
                detected_tags.extend(detect_tag(answer))
                if len(detected_tags) > 0:
                    print('\t'.join([' '.join(set(detected_tags)), query, answer]))
            except ValueError:
                sys.stdout.write('Illegal line.\n')
        else:
            break
