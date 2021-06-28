# encoding: utf-8

import sys
import json

if __name__ == '__main__':
    tag_stat = {}
    total_count = 0.0
    while True:
        line = sys.stdin.readline().strip()
        if line:
            tags = json.loads(line)['Tag']
            for tag in tags:
                tag_stat.setdefault(tag, 0.0)
                tag_stat[tag] += 1
                total_count += 1
        else:
            break
    tag_n_freq = sorted(tag_stat.items(), key=lambda elem: elem[1], reverse=True)
    for tag, freq in tag_n_freq:
        print('%s: %d, %.2f%%' % (tag, freq, freq/total_count*100))
