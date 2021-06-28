# -*- coding: utf-8 -*-

import jieba, random, fileinput, numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import xiaoice_jieba


def load_data_simple(name):
	tokenizer = Tokenizer(filters='', lower=True, split=' ', oov_token=1)
	texts = [' '.join(jieba.cut(' '.join([x.strip() for x in line.split('\t')[:2]]))) \
			 for line in open('data/{}/{}.txt'.format(name, name), 'r', encoding='utf-8'
							  ).read().strip().split('\n')]
	tokenizer.fit_on_texts(texts)

	x_a_train,x_b_train, y_train = [], [], []
	text_train = []
	for line in open(r'F:\ruijie\data\chat\train.data.pre.txt', 'r', encoding='utf-8').read().strip().split('\n'):
		a, text, label = line.split('\t')
		if label != '0':
			label = 1
		text_train.append(text.strip())
		x_a_train.append(' '.join(list(xiaoice_jieba.cut_with_dict(a.strip()))))
		x_b_train.append(' '.join(list(xiaoice_jieba.cut_with_dict(text.strip()))))
		y_train.append(int(label))

	x_a_dev, x_b_dev, y_dev = [], [], []
	text_dev = []
	for line in open(r'F:\ruijie\data\chat\test.data.pre.txt' , 'r', encoding='utf-8').read().strip().split('\n'):
		a, text, label = line.split('\t')
		if label != '0':
			label = 1
		text_dev.append(text.strip())
		x_a_dev.append(' '.join(list(xiaoice_jieba.cut_with_dict(a.strip()))))
		x_b_dev.append(' '.join(list(xiaoice_jieba.cut_with_dict(text.strip()))))
		y_dev.append(int(label))

	x_a_test, x_b_test, y_test = [], [], []
	text_test = []
	for line in open(r'F:\ruijie\data\chat\test.data.pre.txt' , 'r', encoding='utf-8').read().strip().split('\n'):
		a, text, label = line.split('\t')
		if label != '0':
			label = 1
		text_test.append(text.strip())
		x_a_test.append(' '.join(list(xiaoice_jieba.cut_with_dict(a.strip()))))
		x_b_test.append(' '.join(list(xiaoice_jieba.cut_with_dict(text.strip()))))
		y_test.append(int(label))

	return (x_a_train, x_b_train,y_train,text_train),\
		   (x_a_dev, x_b_dev, y_dev,text_dev),\
		   (x_a_test, x_b_test, y_test, text_test), \
			tokenizer
