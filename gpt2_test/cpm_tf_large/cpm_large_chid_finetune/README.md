前面试了下[CPM-模型]((https://github.com/TsinghuaAI/CPM-Generate))的零样本学习效果，今天打算看下finetune模型的实现方式。

# 数据处理

zero Learning是将候选成语带入原句，分别计算loss，以loss最小的那一个成语作为预测值。

finetune则将句子、候选成语及答案拼接为一句话：

```
上下文: predix <mask> postfix <eod> 选项0: I_0 <sep> 选项1: I_1 <sep> ... 选项9: I_9 <sep> <mask> 答案是: L <eod>
```

举个例子：

```python
{"candidates": ["齐头并进", "借花献佛", "味同嚼蜡", "势不可当", "脍炙人口", "各个击破", "各自为战", "高歌猛进", "改过自新", "一网打尽"],
"content": ["老李带领球队一路#idiom600177#。","关键一点是大家看到光进铜退#idiom600179#。”(秀倩)", "据业内人士消息，"]}
```

组合之后：

```python
"上下文:老李带领球队一路<mask>。<eod>选项0:齐头并进<sep> 选项1:借花献佛<sep> ... 选项9:一网打尽<sep>答案是:7<eod>"
```

# 结果

基本上能跑通，模型太大了，没跑完...
