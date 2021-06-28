# encoding: utf-8

from util.trie import Trie


def detect_tag(sentence, tag_trie):
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


def detect_no_overlap_tag(sentence, tag_trie):
    """
    Judge if sentence contain as least a tag.
    重叠的tag会选择最长的。
    :param sentence: query or answer
    :return: boolean, True if contain, False otherwise.
    """
    length = len(sentence)
    detected_tags = []
    for idx in range(length):
        node = tag_trie.lookup
        idx_tmp = idx
        max_right_idx = idx
        while True:
            if idx_tmp >= length:
                break
            if sentence[idx_tmp] in node:
                node = node[sentence[idx_tmp]]
                idx_tmp += 1
                if Trie.END in node:
                    max_right_idx = idx_tmp
            else:
                break
        if max_right_idx > idx:
            detected_tags.append(sentence[idx:max_right_idx])
    return detected_tags
