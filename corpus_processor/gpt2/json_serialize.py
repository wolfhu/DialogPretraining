# encoding: utf-8

import sys
import json

while True:
    line = sys.stdin.readline()
    if line:
        line = line.strip()
        line = line.replace('\t', ' [SEP] ')
        res = json.dumps({'text':line})
        print(res)
    else:
        break
