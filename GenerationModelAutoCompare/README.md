比较模型还未完全完成，因为判定效果并不是很好，模型数据中的生成数据没有和真实问答数据建立完全对应关系，导致数据集各个标签的比重失衡，效果不佳（eval：p/r=0.65）

对于同一句上文，判定两个下一句回复哪一个更好，label值含义如下：

*	-1：前面比后面好
*	0：一致
*	+1：后面比前面好

采用真实问答数据集，与模型通过beam search生成的生成数据集作为比较对

同样数据集对的值为0

真实回复与生成恢复值为+1

生成恢复值与真实回复为+1

具体数据位于 **/stcvm-i10/D/t-dach/GenerationModelAutoCompare/bert-pytorch-multi-gpu/data/** 

模型结果采用BERT分类结构，输入格式为：[cls] 问1 [sep] 答1 [sep] 问1 [sep] 答2