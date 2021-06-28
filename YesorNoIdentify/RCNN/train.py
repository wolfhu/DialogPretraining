import pickle
import random
import os
from tqdm import tqdm
from RCNN import RCNN_Classifer
from focalloss import *
import torch.utils.data

if __name__ == '__main__':
    threshold = 20
    is_test = False
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    print(device)
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 1000
    input_dim = 160
    core_lens = [1,2,3,4]
    output_channel = 2
    dir_path = './data/'
    file_name = 'train_dataset_%s.pk' % (threshold)
    with open(dir_path + file_name, 'rb') as f:
        data = pickle.load(f)
    test_file_name = 'test_dataset_%s.pk' % (threshold)
    with open(dir_path + test_file_name, 'rb') as f:
        test_data = pickle.load(f)
    wordlist_file_name = 'word_list_%s.txt' % (threshold)
    with open(dir_path + wordlist_file_name, 'r', encoding='utf-8') as f:
        word_list = [d[:-1] for d in f.readlines()]
    word_num = len(word_list)

    random.shuffle(data)
    train_len = int(len(data) * 0.8)
    train_data, valid_data = data[:train_len], data[train_len:]
    # dataset loader
    train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    drop_last=True)
    valid_data_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    drop_last=True)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   drop_last=True)

    total_train_batch = len(train_data_loader)
    total_valid_batch = len(valid_data_loader)
    total_test_batch = len(test_data_loader)
    print(total_train_batch)
    print(total_valid_batch)
    print(total_test_batch)

    model = RCNN_Classifer(word_num, input_dim, input_dim//2, core_lens, output_channel, device).to(device)
    #criterion = torch.nn.CrossEntropyLoss().to(device)  # Softmax is internally computed.
    criterion = FocalLoss(gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print('Learning started. It takes sometime.')
    pre_avg_acc = 0
    count = 0
    max_save_file_name = ''
    max_acc = -100
    save_file_dir = './save/' + 'threshold-' + str(threshold) + '_input-' + str(input_dim) \
                    + '_hidden-' + str(input_dim // 2) + '_cores-' + str(core_lens) + '_outchannel-' + str(output_channel) \
                    + '/'
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
    for epoch in range(training_epochs):
        model.train()# set the model to train mode (dropout=True)
        avg_cost = 0
        for label, X in tqdm(train_data_loader):
            #label = torch.cat(tuple([xxx.unsqueeze(1) for xxx in label]), dim=1)
            X = torch.cat(tuple([xxx.unsqueeze(1) for xxx in X]), dim=1)

            label = label.to(device)
            X = X.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)
            cost = criterion(hypothesis, label)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_train_batch
        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
        with torch.no_grad():
            model.eval()
            avg_acc = 0
            for label, X in tqdm(valid_data_loader):
                X = torch.cat(tuple([xxx.unsqueeze(1) for xxx in X]), dim=1)

                label = label.to(device)
                X = X.to(device)

                prediction = model(X)
                correct_prediction = torch.argmax(prediction, -1) == label

                avg_acc += correct_prediction.float().mean().item() / total_valid_batch
            print('Acc ', avg_acc)

        if avg_acc < pre_avg_acc:
            count += 1
            if count >= 1:
                learning_rate = learning_rate / 2
                count = 0
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                print('learing rate adjust: ', learning_rate)
        else:
            save_file_name = save_file_dir + str(epoch) + '_' + str(avg_acc)[:5]
            if epoch == 0:
                max_save_file_name = save_file_name
                max_acc = avg_acc
            elif epoch != 0 and avg_acc > max_acc:
                print('max changed')
                max_save_file_name = save_file_name
                max_acc = avg_acc
            torch.save(model, save_file_name)
            print('saveed..')
        count = 0
        pre_avg_acc = avg_acc

    print('Learning Finished!')

    print('use: ', max_save_file_name)
    with open(save_file_dir+'best_para.txt', 'w', encoding='utf-8') as f:
        f.writelines(max_save_file_name)
    model = torch.load(max_save_file_name)
    with torch.no_grad():
        model.eval()
        avg_acc = 0
        for label, X in tqdm(test_data_loader):
            X = torch.cat(tuple([xxx.unsqueeze(1) for xxx in X]), dim=1)
            label = label.to(device)
            X = X.to(device)
            prediction = model(X)
            correct_prediction = torch.argmax(prediction, -1) == label
            avg_acc += correct_prediction.float().mean().item() / total_test_batch
        print('Acc ', avg_acc)




