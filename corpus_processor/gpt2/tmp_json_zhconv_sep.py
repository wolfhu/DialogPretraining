# encoding: utf-8

import sys
import zhconv
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default=None)
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()

with open(args.input, "r", encoding='utf-8') as f_in, \
    open(args.out, "w", encoding='utf-8') as f_out:
    while True:
        line = f_in.readline()
        if line:
            line = line.strip()
            line = json.loads(line)
            line['text'] = zhconv.convert(line['text'], 'zh-cn')
            line['text'] = line['text'].replace('\t', ' [SEP] ')
            line = json.dumps(line)
            f_out.write(line + '\n')
        else:
            break
