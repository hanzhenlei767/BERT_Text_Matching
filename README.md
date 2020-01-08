# BERT_Text_Matching
BERT在文本匹配问题的应用

* 1.此版本是我公司项目用到的模型，使用的是BERT fine-tuning，可根据不用的应用任务修改模型。
* 2.数据放到data目录下，数据文件夹用任务名。
* 3.中文或者英文任务需要把对应的BERT预训练参数、bert_config.json、vocab.txt拷贝到/bert/english_model文件夹下。
* 4.参数可以设置的不多，最大句子长度max_seq_length，train_batch_size，num_train_epochs。
* 5.我项目中用到最大句子长度是380，单GPU情况12G显存情况下支持最大batch是20,4块GPU情况下用的Batch是64。效果略有提升。
* 6.执行时，仅需要有python bert_model.py即可。

