**LanguageFluency** 数据构造，数据，训练模型及蒸馏模型

##	项目介绍
本项目是一个判定对话句子是否通顺的模型。模型的构造一共使用四步：
* 	默认原始数据为通顺数据（原始数据约5%的不通顺），使用人工构造方法，构造出不通顺的句子。再通过模型训练-再分类-提取更通顺数据的几轮操作，使得通顺数据纯度和通顺度得以提升。再以此为最终的通顺数据，人工构造不通顺数据，得到最终的训练数据。
* 	使用Roberta模型对训练数据进行训练并得到结果
* 	将训练好的12层Roberta模型蒸馏为同等其他参数下3层BERT模型，以及RCNN模型。
*	将蒸馏好的三层BERT模型转换为onnx

##	注意
本项目的数据及模型参数位于stcvm-i10机器上，在下文介绍时用‘\*’标注

其中模型输出中：label=0为通顺，label=1为不通顺

##	项目文件介绍

Dataset：最终使用的数据集

roberta_zh-master：使用roberta哈工大中文预训练模型进行fintune

roberta_zh-master_zhengliu：使用20160101_20181231.log.query.txt和tracelog.fromSR.20200301-20200520.txt预测，获取其未softmax的分布

Distillation_RCNN：使用RCNN对Roberta模型进行蒸馏

Distillation_Bert：使用三层BERT对Roberta模型进行蒸馏

Trans_Distillation_Bert_to_onnx：训练的蒸馏BERT模型，pytorch转onnx

具体的项目介绍位于各个文件夹内部的README.md中
