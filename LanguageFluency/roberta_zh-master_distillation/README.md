##  目的
通过对以训练好模型Roberta，获取对如下蒸馏数据集的输出value

  20160101_20181231.log.query.txt

  tracelog.fromSR.20200301-20200520.txt

由于正常的BERT类型模型以增添Softmax层，因此此模型输出去掉了这层以确保能够进行接下来的蒸馏


##	缺失文件
位于 **/stcvm-i10/D/t-dach/LanguageFluency/roberta_zh-master_distillation/** 

chinese_roberta_wwm_ext_L-12_H-768_A-12：存放RoBERTa预训练模型

data：存放待预测的数据，分别为对应log.query和sr

output：存放Roberta训练好的模型参数，以及预测数据的value结果



使用：中文预训练RoBERTa模型 
-------------------------------------------------
RoBERTa是BERT的改进版，通过改进训练任务和数据生成方式、训练更久、使用更大批次、使用更多数据等获得了State of The Art的效果；可以用Bert直接加载。

