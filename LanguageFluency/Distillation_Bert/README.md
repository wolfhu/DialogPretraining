##	使用Transformer进行三层BERT蒸馏

##	缺失文件

位于 **/stcvm-i10/D/t-dach/LanguageFluency/Distillation_Bert/** 

###	./data/

蒸馏数据与Roberta使用的训练数据不同，数据源自两部分：

	20160101_20181231.log.query.txt

	tracelog.fromSR.20200301-20200520.txt

构造后的通顺和不通顺数据分别对应为：

	correct_data_log.txt

	correct_data_sr.txt

	error_data_log.txt

	error_data_sr.txt

数据格式为：句子\t通顺value\t不通顺value

最终数据汇总encode为：all_dataset_ids.tsv

###	GPT2

encoder和decoder的单词表


###	model

保存蒸馏后的模型




