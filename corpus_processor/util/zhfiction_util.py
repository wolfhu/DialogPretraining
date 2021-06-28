# encoding: utf-8

tags_file = 'gpt2/zhfiction_tags'

tag_set = set()
with open(tags_file, 'r') as f:
    for tag in f.readlines():
        tag = tag.strip()
        tag_set.add(tag)


def contain_selected_tag(string):
    """
    判断给定的字符串是否包含任意一个指定的tag
    tag由tags_file指定
    主要用于判断文件是否是小说
    """
    for tag in tag_set:
        if tag in string:
            return True
    return False
