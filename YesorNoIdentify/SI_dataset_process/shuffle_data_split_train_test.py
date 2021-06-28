import pickle
from random import shuffle

if __name__ == '__main__':
    threshold = 20
    with open('./dataset_%s.pk' % (threshold), 'rb') as f:
        dataset = pickle.load(f)

    type_list = ['certain', 'neg', 'question', 'uncertain']
    with open('./type_list.pk', 'wb') as f:
        pickle.dump(type_list, f)

    result_data = []
    for t in type_list: # [type, seq], no seq len
        type_id = type_list.index(t)
        for d in dataset[t]['data']:
            result_data.append([type_id, d])
    shuffle(result_data)
    '''
    dataset:
        [type_id, seq_id]
    '''
    with open('./train_test_dataset_%s.pk' % (threshold), 'wb') as f:
        pickle.dump(result_data, f)

    test_ratio = 0.2
    train_data = result_data[int(len(result_data)*test_ratio):]
    with open('./train_dataset_%s.pk' % (threshold), 'wb') as f:
        pickle.dump(train_data, f)
    test_data = result_data[:int(len(result_data) * test_ratio)]
    with open('./test_dataset_%s.pk' % (threshold), 'wb') as f:
        pickle.dump(test_data, f)
    print(1)

