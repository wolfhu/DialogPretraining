# encoding: utf-8

import sys
import requests
import time
import threading
import queue
import urllib3

from util.str_util import remove_blank

thread_num = 100
mes_q = queue.Queue()
url = 'http://int-corechat.cloudapp.net/api/knowledge/KnowledgeInfer'


def produce(query, answer):
    mes_q.put((query, answer,))


def relation_predict():
    while True:
        try:
            query, answer = mes_q.get(False)
        except urllib3.exceptions.ProtocolError as e:
            sys.stderr.write(str(e) + '\n')
            continue
        except requests.exceptions.ConnectionError as e:
            sys.stderr.write(str(e) + '\n')
            continue
        except queue.Empty:
            return
        data = {
            "TargetDomain": "general",
            "Query": query
        }
        response = requests.post(url, data).json()
        if len(response) == 0:
            continue
        triple_list = []
        for item in response:
            entity = item['sbj']
            entity = remove_blank(entity)
            relation = item['predicate']
            predict = item['obj']
            triple_list.append(' ||| '.join((entity, relation, predict,)))
        sys.stdout.write('\t'.join((' $$$ '.join(triple_list), query, answer)) + '\n')


if __name__ == '__main__':
    while True:
        line = sys.stdin.readline().strip()
        if line:
            query, answer = line.split('\t')
            produce(query, answer)
        else:
            break

    sys.stderr.write('queue size: %d\n' % (mes_q.qsize(),))

    start = time.time()
    thread_list = []
    for i in range(thread_num):
        thread = threading.Thread(target=relation_predict, args=())
        thread.start()
        thread_list.append(thread)

    for thread in thread_list:
        thread.join()
    end = time.time()
    sys.stderr.write(str(end - start) + '\n')


# start = time.time()
# thread_list = []
# for i in range(thread_num):
#     query = '登巴巴回归，肯定是目前最好的选择'
#     thread = threading.Thread(target=relation_predict, args=(query,))
#     thread.start()
#     thread_list.append(thread)
#
# for i in range(thread_num):
#     thread_list[i].join()
#
# end = time.time()
# print('Elapsed time: ' + str(end - start))


# if __name__ == '__main__':
#     while True:
#         line = sys.stdin.readline().strip()
#         if line:
#             query, answer = line.split('\t')
#             data["Query"] = query
#             response = requests.post(url, data).json()
#             if len(response) == 0:
#                 continue
#             triple_list = []
#             for item in response:
#                 entity = item['sbj']
#                 relation = item['predicate']
#                 predict = item['obj']
#                 triple_list.append(' ||| '.join((entity, relation, predict,)))
#             print('\t'.join((' $$$ '.join(triple_list), query, answer)))
#         else:
#             break
