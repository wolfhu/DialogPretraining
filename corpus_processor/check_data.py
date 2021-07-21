import editdistance

def remove_dup():
    f = open(r"D:\user\yuwu1\douban\douban_weibo_tianya_multi_turn_big.train",encoding="utf-8")
    fw = open(r"D:\user\yuwu1\douban\douban_weibo_tianya_multi_turn_big.train.fliter2","w",encoding="utf-8")
    counter = 0
    for line in f:
        tmp = line.split("\t")
        flag = False
        for idx in range(len(tmp)-1):
            query = tmp[idx]
            answer = tmp[idx + 1]
            if len(query) >= 5:
                dis = editdistance.eval(query,answer)
                if dis / min(len(query),len(answer)) <= 0.2:
                    flag = True
                    break
                    print(query, answer)
        if not flag:
            fw.write(line)
        else:
            counter += 1
            #print(line)
    print(counter)


def count_message():
    d = {}
    f = open(r"D:\user\yuwu1\douban\douban_weibo_tianya_multi_turn_big.train",encoding="utf-8")
    for line in f:
        tmp = line.split("\t")
        flag = False
        for t in tmp:
            if t not in d:
                d[t] = 0
            d[t] += 1

    res = dict(sorted(d.items(), key=lambda item: item[1], reverse= True))
    for utt in res:
        print(utt.strip(), res[utt])
        if res[utt] < 1000:
            break

remove_dup()
#count_message()