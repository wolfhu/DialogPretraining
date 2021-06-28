# encoding: utf-8

import sys

query_answer_dict = {}

answer_num_limit = 1000

if __name__ == '__main__':
    for line in sys.stdin.readlines():
        line = line.strip()
        (query, answer) = line.split('\t')
        query_answer_dict.setdefault(query, {})
        query_answer_dict[query].setdefault(answer, 0)
        query_answer_dict[query][answer] += 1

    for query in query_answer_dict.keys():
        answer_dict = query_answer_dict[query]
        if len(answer_dict.keys()) > answer_num_limit:
            answer_freq_tuple = sorted(answer_dict.items(), reverse=True)
            for answer, freq in answer_freq_tuple:
                for _ in range(freq):
                    print('\t'.join((query, answer)))
        else:
            for answer in answer_dict.keys():
                for _ in range(answer_dict[answer]):
                    print('\t'.join((query, answer)))
