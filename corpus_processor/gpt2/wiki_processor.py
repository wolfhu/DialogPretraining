# encoding: utf-8

import sys
import re
import zhconv

from util.str_util import str_extreme_clean

# in_file_path = '/home/t-yuniu/xiaoice/yuniu/dataset/wiki/zhwiki.txt'


def is_title(string):
    pattern1 = u"【.*?】"
    pattern2 = u"={1,}.*?={1,}"
    res = re.match(pattern1, string)
    if res is not None and res.end() - res.start() == len(string):
        return True
    res = re.match(pattern2, string)
    # print('res of pat2: ' + str(res))
    # print('string: %s' % string)
    # print('start: %d, end: %d' % (res.start(), res.end()))
    if res is not None and res.end() - res.start() == len(string):
        return True
    return False


def is_item(string):
    """
    去除如下的item列表
    ==电影制作==
    # 脚本发展
    # Pre-production
    # 电影制作

    == 电影种类 ==
    * 动作电影
    * 奇幻电影
    * 喜剧电影

    == 参见 ==
    ** 长镜头
    ** 电影特技
    **3D电影
    """
    if string[0] in ['*', '#']:
        return True
    else:
        return False


empty_line_count = 0
res = []
while True:
    line = sys.stdin.readline()
    if line:
        line = line.strip()
        if len(line) == 0:
            empty_line_count += 1
            if empty_line_count >= 3:
                print('\t'.join(res))
                res = []
            continue
        else:
            empty_line_count = 0
        # print('before is_title')
        # print(len(line))
        # print(line)
        # print('is_title: ' + str(is_title(line)))
        if is_title(line) or is_item(line) or len(line) <= 10:
            continue
        line = str_extreme_clean(line)
        line = line.replace('\t', '')
        line = zhconv.convert(line, 'zh-cn')
        if len(line) == 0:
            continue
        else:
            res.append(line)
    else:
        break
print('\t'.join(res))
