# encoding: utf-8

import sys
import zhconv
import json
import langid

from util import str_util

strs_to_remove = ['\n']

while True:
    line = sys.stdin.readline().strip()
    if line:
        description_txt = json.loads(line)['Description'].strip()
        description_txt = zhconv.convert(description_txt, 'zh-cn')
        if langid.classify(description_txt)[0] != 'zh':
            continue
        description_list = description_txt.split('\n')

        content_txt = json.loads(line)['Content'].strip()
        content_txt = zhconv.convert(content_txt, 'zh-cn')
        if langid.classify(content_txt)[0] != 'zh':
            continue
        content_list = content_txt.split('\n')

        description = ''
        for desc_line in description_list:
            desc_line = str_util.remove_blank(desc_line)
            if len(desc_line) <= 10:
                continue
            description += str_util.remove_blank(desc_line)
        description = zhconv.convert(description, 'zh-cn')

        content = ''
        for content_line in content_list:
            content_line = str_util.remove_blank(content_line)
            if len(content_line) <= 10:
                continue
            content += str_util.remove_blank(content_line)
        content = zhconv.convert(content, 'zh-cn')

        if len(description) > 10:
            print(description)
        if len(content) > 10:
            print(content)
    else:
        break
