##	RCNN模型

字粒度RCNN模型，采用原始数据集频率超过20的字

###	参数描述

测试了参数区间input-dim:[20:500:20], core:[1,2,3] [1,2,3,4] [1,2,3,4,5]

测试结果位于./analyis/1-core-dim.png所示

可知在input_dim=140左右, core:[1,2,3,4]效果较好

再测试input-dim:[120:160:20], outchannel:[2:12:1]

测试结果./analyis/2-dim-channel.png所示

最终选定input_dim=140, core=[1,2,3,4], outchannel=5

###	阈值选择

阈值的调试结果位于./analyis/threshold_\*.txt

可以看出，模型对于类别预测已经置信率已经很高。。。几乎没有在两个类别之间摇摆不定的情况。。。

因此卡阈值提升r值并不明显

###	其他缺失文件

位于 **/stcvm-i10/D/t-dach/YesorNoIdentify/RCNN/** 

数据训练数据集位于 **/stcvm-i10/D/t-dach/YesorNoIdentify/dataset/** 
