# encoding:utf-8

import sys
import re

from util.str_util import remove_repeat_ch, is_alphabet

black_list = ['楼上', '楼下', '楼主', '贴图', '图片', '照片', '帖图']

if __name__ == '__main__':
    while True:
        line = sys.stdin.readline().strip()
        if line:
            legal_flag = True
            for black in black_list:
                if black in line:
                    legal_flag = False
                    break
            if not legal_flag:
                continue
            # print('legal')
            turns = []
            for turn in line.split('\t'):
                # print('turn')
                # print(turn)
                turn = turn.strip()
                if re.match('[0-9]{7,100}', turn) is not None:
                    # print('repeat num')
                    break
                alpha_count = 0.0
                num_count = 0.0
                for ch in turn:
                    if is_alphabet(ch):
                        alpha_count += 1
                    if ch.isdigit():
                        num_count += 1
                # print(alpha_count)
                # print(num_count)
                if max(alpha_count, num_count) / len(turn) > 0.2:
                    # print('too much numbers or characters')
                    break
                turn = remove_repeat_ch(turn)
                if 10 < len(turn) < 128:
                    turns.append(turn)
            if len(turns) > 1:
                print('\t'.join(turns))
        else:
            break
