# encoding: utf-8

import sys

if __name__ == '__main__':
    tag_stat = {}
    total_count = 0.0
    while True:
        line = sys.stdin.readline().strip()
        if line:
            items = line.split('\t')
            assert len(items) == 3
            keywords = items[0].split(' ')
            for keyword in keywords:
                tag_stat.setdefault(keyword, 0.0)
                tag_stat[keyword] += 1
                total_count += 1
        else:
            break
    tag_n_freq = sorted(tag_stat.items(), key=lambda elem: elem[1], reverse=True)
    for tag, freq in tag_n_freq:
        print('%s: %d, %.2f%%' % (tag, freq, freq/total_count*100))