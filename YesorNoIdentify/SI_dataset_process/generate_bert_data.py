from random import shuffle

def read_data(input_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return [d[:-1] for d in data]


def save_file(save_file_path, data):
    with open(save_file_path, 'w', encoding='utf-8') as f:
        f.writelines([d+'\n' for d in data])


def get_data(input_file_path, type_name):
	data = read_data(input_file_path)
	return [type_name+'\t'+d for d in data]

if __name__ == '__main__':
	results = []

	results.extend(get_data('./new_certain.txt', 'certain'))
	results.extend(get_data('./new_neg.txt', 'neg'))
	results.extend(get_data('./new_question.txt', 'question'))
	results.extend(get_data('./new_uncertain.txt', 'uncertain'))

	shuffle(results)
	save_file('./bert/train.tsv', results[:int(len(results)*0.8)])
	save_file('./bert/dev.tsv', results[int(len(results)*0.8):int(len(results)*0.9)])
	save_file('./bert/test.tsv', results[int(len(results)*0.9):])

