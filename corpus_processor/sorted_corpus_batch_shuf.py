# encoding: utf-8

import sys
import random

batch_size = 16

batch_dict = {}
curr_batch = []
batch_count = 0

if __name__ == '__main__':
    while True:
      line = sys.stdin.readline()
      if line:
        line = line.strip()
        curr_batch.append(line)
        if len(curr_batch) >= batch_size:
          batch_dict.setdefault(batch_count, curr_batch)
          batch_count += 1
          curr_batch = []
          if batch_count % 100000 == 0:
            sys.stderr.write('INFO: written batch: %d\n' % batch_count)
      else:
        break
    if len(curr_batch) > 0:
      batch_dict.setdefault(batch_count, curr_batch)
    idx_list = list(batch_dict.keys())
    random.shuffle(idx_list)
    for batch_idx in idx_list:
      for line in batch_dict[batch_idx]:
        print(line)
