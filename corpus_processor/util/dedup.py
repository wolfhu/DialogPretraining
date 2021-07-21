import editdistance


def dedup():
    exist_utt = set()
    f = open(r"D:\user\yuwu1\douban\wb\merge_douban.txt",encoding="utf-8")
    fw = open(r"D:\user\yuwu1\douban\wb\merge_douban.txt.dedup","w",encoding="utf-8")

    for line in f:
        exist_utt.add(line)
        if len(exist_utt) % 1000000 == 0:
            print(len(exist_utt))
    

    for utt in exist_utt:
        fw.write(utt)

def multi_process_dedup():
    '''
    use Hash to map utterances in smal files, then dedup in each file and merge them
    '''
    pass

def count_utt():
    f = open(r"D:\user\yuwu1\douban\wb\merge_douban.txt.dedup",encoding="utf-8")
    response = {}

    for line in f:
        query, answer = line.split('\t')
        if answer not in response:
            response[answer] = 1
        response[answer] += 1
        #if len(response) > 1000000:
        #    break
    
    res = dict(sorted(response.items(), key=lambda item: item[1], reverse= True))
    
    fw = open(r"D:\user\yuwu1\douban\wb\merge_douban.tt.dedup",'w',encoding="utf-8")
    counter = 0
    for utt in res:
        fw.write("{0}\t{1}\n".format(utt.strip(), res[utt]))
        print(utt.strip(), res[utt])
        counter += 1
        if counter > 1000:
            break


if __name__ == '__main__':
    #count_qasim()
    count_utt()
  #dedup()
  #filelist_test()