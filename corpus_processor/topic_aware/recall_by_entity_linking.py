# encoding: utf-8

import sys
import requests
import json
import queue
import urllib3
import time
import threading

from util.str_util import remove_blank

domain = 'travel'
el_url = 'http://int-corechat.trafficmanager.net/api/Knowledge/EntityLinking'

thread_num = 100
mes_q = queue.Queue()


def keyword_extract():
    el_body = {
        "Query": "我爱天安门",
        "TargetMention": "",
        "TargetDomain": domain
    }
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
        el_body["Query"] = answer
        response = requests.post(url=el_url, json=el_body)
        if response.status_code != 200:
            continue
        response = response.json()
        if len(response) == 0:
            continue
        keyword_list = set()
        for res in response:
            keyword = remove_blank(res['Mention'])
            keyword_list.add(keyword)
        keyword_list = list(keyword_list)
        keyword_str = ' '.join(keyword_list)
        sys.stdout.write('%s\n' % ('\t'.join([keyword_str, query, answer]),))


if __name__ == '__main__':
    while True:
        line = sys.stdin.readline().strip()
        if line:
            query, answer = line.split('\t')
            if query == '“该条回应已被删除”':
                continue
            mes_q.put((query, answer,))
        else:
            break

    start = time.time()
    thread_list = []
    for i in range(thread_num):
        thread = threading.Thread(target=keyword_extract, args=())
        thread.start()
        thread_list.append(thread)

    for thread in thread_list:
        thread.join()
    end = time.time()
    sys.stderr.write(str(end - start) + '\n')
