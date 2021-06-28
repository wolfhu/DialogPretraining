##	数据预处理

将原始语料进行encode，分为以下几步：

*	get_wordlist_convert2id.py：将原始数据生成词表并encode，保存为pickle文件
*	shuffle_data_split_train_test.py：将数据混洗并分割为train和dev
*	generate_bert_data.py：转为bert类型数据（BERT与RCNN效果相差不大，没有使用BERT）


缺失数据文件位于 **/stcvm-i10/D/t-dach/YesorNoIdentify/SI_dataset_process/** 

