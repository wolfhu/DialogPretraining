# encoding: utf-8

import sys
import zhconv
import json
import langid

from util import str_util

while True:
    line = sys.stdin.readline().strip()
    if line:
        obj = json.loads(line)
        content = obj['content'].strip()
        content = str_util.regexp(content)
        if langid.classify(content)[0] != 'zh':
            continue
        content = zhconv.convert(content, 'zh-cn')
        if len(content) <= 10:
            continue
        print(content)
    else:
        break
