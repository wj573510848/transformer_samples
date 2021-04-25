使用transformer库跑模型

1. 中文模型的下载
https://github.com/ymcui/Chinese-BERT-wwm 提供的模型，使用BertTokenizer以及BertModel加载
https://huggingface.co/hfl 在transformers可以加载的模型

* chinese-roberta-wwm-ext(tf)

* macbert(tf)
https://zhuanlan.zhihu.com/p/333202482 原理

* albert_chinese_small(torch)



# 2. 感情分类模型，三分类
emotion_detect

* bert(使用tensorflow)
训练
```
python train_emotion_classify.py
```

测试
```
python test_emotion_classify.py
```

* albert(使用pytorch)

```python
python train_emotion_classify_albert.py
```

| 模型 | train_acc | val_acc | test_acc |
| ---- | ---- | ---- | ---- |
|chinese-roberta-wwm-ext|0.9527|0.8434|0.8422|
|macbert|0.9424|0.8331|0.8336|
|albert|0.9225|0.8106|0.8079|


# 3. ner

使用数据：https://github.com/CLUEbenchmark/CLUENER2020 中的数据

使用了两种方法，综合下来看，加了crf效果会略好

注意点：

* crf使用 tensorflow_addons，目前decode存在bug， tf-serving及metric不能正常使用
* train_step函数可自定义训练步骤
* serving_output函数可定义tf-serving的输出数据



 模型 | train_acc | val_acc | test_acc |
| ---- | ---- | ---- | ---- |
|chinese-roberta-wwm-ext|0.9541|0.9275|-|
|chinese-roberta-wwm-ext crf |0.9665|0.9235|
|macbert|0.9542|0.9248|-|
|macbert-crf|0.966|0.9232


* lstm的实现方式 https://github.com/saiwaiyanyu/bi-lstm-crf-ner-tf2.0/blob/master/model.py

# 4. generate

```
https://huggingface.co/mymusise/CPM-GPT2 # gpt2中文开源模型
https://github.com/mymusise/CPM-TF2Transformer/ # github，转化为tf
https://github.com/TsinghuaAI/CPM-Generate # 原版地址
https://github.com/TsinghuaAI/CPM-Finetune # finetune
```