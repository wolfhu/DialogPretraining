
import editdistance
def count_qasim():
    f = open(r"D:\user\yuwu1\douban\wb\merge_douban.txt.dedup",encoding="utf-8")
    response = {}

    for line in f:
        query, answer = line.split('\t')
        if len(query) >= 5:
            dis = editdistance.eval(query,answer)
            if dis / min(len(query),len(answer)) <= 0.2:
                print(query, answer)

        elif query.strip() == answer.strip():
            print(query, answer)
            
    counter = 0    
    res = dict(sorted(response.items(), key=lambda item: item[1], reverse= True))
    for utt in res:
        print("{0}\t{1}\n".format(utt.strip(), res[utt]))
        counter += 1
        if counter > 1000:
            break