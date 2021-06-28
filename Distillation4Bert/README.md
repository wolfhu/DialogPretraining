**Chat ML models** 蒸馏框架
##	项目介绍
本项目是一个模型蒸馏/压缩的框架，主要使用Bert作为base 模型。由于不同模型压缩方法和结构不同，该项目主要将压缩结构分成了两种：
* 	Teacher /student 同时训练（之后简称“框架1”）， 例如Theseus Bert
* 	Teacher /student 分开训练（之后简称“框架2”）， 例如TinyBert、DistillerBert、PKD 等

##	使用方法
由于压缩结构不同，“框架1” 和“框架2”的训练方法也不同，下面分开介绍。

### Teacher /student 同时训练（“框架1”）
      该部分的模型主要包括 Theseus-Bert 于DCBert 两个。这两个模型都是在具体NLP 下游task的finetue阶段实施的，而不是直接作用于预训练阶段。

#### 数据/模型准备
进行压缩时，需要有：
a)	已经finetune 好的teacher 模型 t_model
b)	Teacher 模型结构（通常以.config 文件进行表达）
c)	进行压缩需要的数据
d)	Student模型结构（通常以.config 文件进行表达）
e)	进行压缩的其他参数设置 (在config/distill.example.1.json 中进行设置)

#### 进行压缩
Step1. 将model_type、数据地址、输出地址填入config/distill.example.1.json

```shell
"task_name": "chat",
"data_dir": "data",
"output_dir": "output_test",
"do_train" : true,
"do_eval" : true,
"configuration_file": "must feed this value when the distill type is not in the exist map",
"per_gpu_train_batch_size" : 8,
"num_train_epochs": 2,
```

Step2. python run_glue.py

#### 其他参数设置

DCBert: 

```shell
"dc_bert_model_config":{
  "note":"this is only for DCbert, must init when distilling with dc-bert. Otherwise, ignore it",
  "last_layer_model_type": "transformer",
  "copy_from_bert_layer_num": -1
},
```

可以选择最后一层的结构类型，如Linear/Transformer, 如需其他类型，可以在models/dc_bert.py自行配置; 可选择dc bert layer 使用teacher_model的哪个层进行初始化，默认是最后一层
Theseus-bert：

```shell
"theseus_scheduler" :{
  "note":"this is only for Theseus bert, must init when distilling with theseus. Otherwise, ignore it",
  "replacing_rate": 0.3,
  "scheduler_type": "linear",
  "scheduler_linear_k": 0.0006
},
```

 TheseusBert 会在训练时将student_model 的某个module 平行替换teacher_model 中对应的module。替换策略有两种:  linear: LinearReplacementScheduler; None:ConstantReplacementScheduler.

### Teacher /student 分开训练（“框架2”）
      该部分的模型主要是常规的模型蒸馏方法，包括TinyBert/DistillerBert/PKD 等。模型在压缩是包括模型预训练阶段压缩与task 相关的压缩。

#### 数据/模型准备
进行压缩时，需要有：
a)	预训练好的模型
b)	预训练好的模型结构(通常以.config 文件进行表达）
c)	已经finetune 好的teacher 模型 t_model
d)	Teacher 模型结构（通常以.config 文件进行表达）
e)	进行压缩需要的数据
f)	Student模型结构（通常以.config 文件进行表达）
g)	进行压缩的其他参数设置 (在config/distill.example.1.json 中进行设置)

#### 进行压缩
Step1. 将model_type、数据地址、输出地址填入config/distill.example.1.json
Step2.  python run_glue.py

#### RNN based 模型压缩
	
现在支持Elmo-based 模型压缩以及 RNN-based 的模型压缩。

Elmo-based模型压缩。 Elmo的模型文件放在了 \\gcrnfsw2-xiaoic\xiaoicechatexp\ruizha\elmo， 使用前需要将这个目录下的两个文件先拷贝到自己的训练目录。然后直接运行 elmo/elmo_distill.py 即可。
 
LTD-based 模型压缩。由于取IDF等操作需要进行分词，所以需要预先安装 xiaoice_jieba 分词工具，然后直接运行 ltd-bert/distill.py 即可。

####一些模型说明
DistillerBert,  先进性预训练的压缩，然后利用压缩的结果直接进行task-finue，本框架只进行了压缩，finetune 部分可利用HuggingFace 的框架直接finetune
TinyBert, 包括预训练压缩、task压缩，task 压缩又分成了 HiddenLayer&Attention layer压缩，predictLayer 压缩。在config/models_config 已经将相关的配置全部写好。在原始TinyBert 论文中还有DataAugumention 过程，可直接使用data_utils/data_augmentation.py 进行数据增强操作。

#### 其他
如果需要压缩的模型配置没有预先配置好，可以 在config/models_config 下面添加相应的模型配置文件，并且填写config/distill.example.1.json 的  configuration_file 字段。

```shell
"configuration_file": "must feed this value when the distill type is not in the exist map",
```


##	自定义模型

### 自定义数据
现在框架里实现了QA 数据(即task:ChatQA) 以及Bert_glue 的数据处理格式。如果需要处理的数据格式不在已有的方法内，可以参考Bert 添加自己的processor 即可，并且定义好相应的task_name。

### 自定义模型结构
现在框架主要实现了QA 类型以及BertForSequenceClassification， 如果又其他自定义的数据模型，可以按照Bert 添加自己的模型即可

##	环境设置
docker/Dockerfile  已经设置好了相应的环境配置，可以直接安装docker 进行训练，也可按照Dockerfile   的内容配置自己的环境
##	其他
如果有其他问题，可以联系ruizha@microsoft.com
