import os
import pickle as pk
import matplotlib.pyplot as plt

def collect_eval_result(input_dir, output_dir):
    file_list = os.listdir(input_dir)
    result = []
    for f_n in file_list:
        para = f_n.split('_')
        para = [d.split('-') for d in para]
        tmp_dict = dict()
        for k, v in para:
            tmp_dict[k] = v
        tmp_dict['eval'] = dict()

        with open(input_dir + f_n + '/eval.txt', 'r', encoding='utf-8') as f:
            data = [d[:-2] for d in f.readlines()]
        type_list = data[0].split('\t')[1:]
        data = [d.split('\t') for d in data[1:]]
        for d in data:
            tmp_dict['eval'][d[0]] = dict()
            for i, v in enumerate(d[1:]):
                tmp_dict['eval'][d[0]][type_list[i]] = float(v)
        result.append(tmp_dict)
    with open(output_dir + 'eval_result.pk', 'wb') as f:
        pk.dump(result, f)
    with open(output_dir + 'eval_result.pk', 'rb') as f:
        print(pk.load(f))



def draw_plots(save_dir):
    with open(save_dir + 'eval_result.pk', 'rb') as f:
        data = pk.load(f)
    # input_dim cores_len
    core_3 = [d for d in data if d['cores'] == '[1, 2, 3]' and d['outchannel'] == '5']
    core_4 = [d for d in data if d['cores'] == '[1, 2, 3, 4]' and d['outchannel'] == '5']
    core_5 = [d for d in data if d['cores'] == '[1, 2, 3, 4, 5]' and d['outchannel'] == '5']
    X_3 = [d['input'] for d in core_3]
    X_4 = [d['input'] for d in core_4]
    X_5 = [d['input'] for d in core_5]
    Y_3 = [round(sum([v for k, v in d['eval']['f1'].items()]) * 25, 2) for d in core_3]
    Y_4 = [round(sum([v for k, v in d['eval']['f1'].items()]) * 25, 2) for d in core_4]
    Y_5 = [round(sum([v for k, v in d['eval']['f1'].items()]) * 25, 2) for d in core_5]
    plt.xlabel('input_dim')
    plt.plot(X_3, Y_3, color='r', label='core:[1,2,3]')
    plt.plot(X_4, Y_4, color='b', label='core:[1,2,3,4]')
    plt.plot(X_5, Y_5, color='g', label='core:[1,2,3,4,5]')
    plt.ylabel('F1 average')
    plt.title('outchannel:5')
    plt.legend(loc='best')
    plt.savefig(save_dir+'dim.png')
    plt.show()
    # outchannel


    return


if __name__ == '__main__':
    input_dir = './eval/'
    output_dir = './analyis/'
    #collect_eval_result(input_dir, output_dir)
    draw_plots(output_dir)
    print(1)




