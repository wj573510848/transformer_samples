GPT2是这几年比较火的一个模型，以前没有发现好的中文预训练模型，最近看到了中文的GPT2开源模型，[CPM-Generate](https://github.com/TsinghuaAI/CPM-Generate)。感谢作者的分享：
```
为了促进中文自然语言处理研究的发展，本项目提供了 CPM-LM (2.6B) 模型的文本生成代码，可用于文本生成的本地测试，并以此为基础进一步研究零次学习/少次学习等场景。
```

# 零样本学习

通常预训练模型使用finetune的方式应用于下游任务，零样本学习则不需要finetune，直接使用模型进行预测，[CMP-Finetune](https://github.com/TsinghuaAI/CPM-Finetune)提供了ChiD与STC两个数据集的处理。此次探索了下chid的零样本学习过程。

# chid数据集

chid数据集为一个成语完形填空的数据集，数据集分为train、dev、test三大类，每一类数据集的格式是相同的，以test为例，为两个文件`test.json`,`test_answer.json`，`test.json`中，其每一条原始数据为：

```python
{"candidates": ["齐头并进", "借花献佛", "味同嚼蜡", "势不可当", "脍炙人口", "各个击破", "各自为战", "高歌猛进", "改过自新", "一网打尽"],
"content": ["老李带领球队一路#idiom600177#。","关键一点是大家看到光进铜退#idiom600179#。”(秀倩)", "据业内人士消息，"]}
```

candidates：候选的成语集合，固定为10个

content：需要预测成语的句子，有多条，由于原文比较长，这里截取了部分，成语使用`#idiom600177#`类似的格式标识。

`test_answer.json`为一个字典：
```python
{
    "#idiom600168#": 6,
    "#idiom600169#": 3,
    "#idiom600170#": 4,
    "#idiom600171#": 8,
    "#idiom600172#": 0,
    "#idiom600177#": 7,}
```
标识了content中的成语与candidates中成语的对应关系。如`#idiom600177#`对应为7，那么对应candidates中第8个成语“高歌猛进”。

# 模型的任务

模型的任务可以归纳为：

输入： 带有成语记号的句子，如 “老李带领球队一路#idiom600177#” ，候选集合十个成语

输出：正确的成语

# 零样本学习模型建立方式

零样本学习如果直接当做一个分类任务，那么其效果会不好。阅读[CPM-Generate](https://github.com/TsinghuaAI/CPM-Generate)的源代码之后，作者采用的是，将十个成语依次填入到原句之中，计算句子的loss，以loss最低的句子作为正确句子。简化过程如下：

1. 制作十个备选句子，前面为例：

```python
# 十个备选句子
"老李带领球队一路齐头并进"
"老李带领球队一路借花献佛"
"老李带领球队一路味同嚼蜡"
# ...
"老李带领球队一路一网打尽"
```

2. 使用模型，计算loss

3. 选取loss最小的模型进行预测

# 代码实现

CPM是使用pytorch实现的，由于一直使用的是tf，所以，使用transformers中的tf模型做了尝试。

transformers安装使用：`https://huggingface.co/transformers/`

CPM tf2 模型（基于transformers）： `https://huggingface.co/mymusise/CPM-GPT2`

## GPT2预测模型

transformer中的`TFGPT2LMHeadModel`计算loss的方式与本任务有些出入，因此，对计算loss进行了改写，为了标识出label的mask，label使用`-100`进行padding。

```python
# models.py

        if labels is not None:
            mask = tf.not_equal(labels, -100)
            
            labels = tf.cast(labels,tf.int32) * tf.cast(mask,tf.int32) 
            raw_loss = loss_fun(labels,logits)

            mask = tf.cast(mask,tf.float32)
            losses = tf.cast(raw_loss,tf.float32) * mask
            losses = tf.reduce_sum(losses,axis=-1) / tf.reduce_sum(mask,axis=-1)
        
        return losses

def loss_fun(labels, logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    return loss_fn(labels, logits)

```

## 最大长度零样本学习

与[CPM-Generate](https://github.com/TsinghuaAI/CPM-Generate)相同，每句话使用了全部的文字：

```python
# preprocess_data.py
def process_one_sent_eval(tokenizer, sent, answers, candidates):
    '''
    tokenizer: 分词器
    sent: 带有成语标识的句子，"老李带领球队一路#idiom600177#。"
    answers: dict，答案#idiom600177#类似对应的答案
    candidates: list,候选成语
    '''
    pattern = re.compile(r"#idiom\d+#")
    L = []
    # 正则找到含有成语标识的位置
    find_iter = pattern.finditer(sent)
    # 将所有的候选成语填充到句子之中
    for m in find_iter:
        if m:
            L.append({'cands': [], 'truth': answers[m.group()]})
            for idm in candidates:
                cand = sent[:m.start()] + idm + sent[m.end():]
                cand = pattern.sub('', cand)
                ids = tokenizer.encode(cand)
                L[-1]['cands'].append(ids)
    return L
```

## 固定长度零样本学习

最大长度进行零样本学习，最大的问题就是计算速度太慢了，是否可以使用固定长度呢？思路是，成语之前截取XX长度，不足的以成语之后的文字填充

```python
# preprocess_data_fixed_length.py
def process_one_sent_eval(tokenizer, sent, answers, candidates, max_length):
    pattern = re.compile(r"#idiom\d+#")
    L = []
    find_iter = pattern.finditer(sent)
    # 将所有的候选成语填充到句子之中
    for m in find_iter:
        if m:
            L.append({'cands': [], 'truth': answers[m.group()]})
            for idm in candidates:
                pre_sent = sent[:m.start()] + idm
                pre_sent = pattern.sub("", pre_sent)
                post_sent = sent[m.end():]
                post_sent = pattern.sub('', post_sent)

                pre_ids = tokenizer.encode(pre_sent)[:-2]
                post_ids = tokenizer.encode(post_sent)[:-2]

                pre_start = len(pre_ids) - max_length
                post_end = max_length - len(pre_ids)

                if len(pre_ids) >= max_length:
                    ids = pre_ids[pre_start:]
                else:
                    ids = pre_ids + post_ids[:post_end]

                assert len(ids) <= max_length
                L[-1]['cands'].append(ids)
    return L
```

# 结果

|长度|准确率|
| --- | --- |
|最大长度|0.6792|
|64|0.5310|

# 备注

个人代码：`https://github.com/wj573510848/transformer_samples/tree/master/gpt2_test/cpm_tf_large/cpm_large_chid_zero`

需在`config.py`中指定预训练模型位置，预训练模下载地址：`https://huggingface.co/mymusise/CPM-GPT2`。

需在`config.py`中指定chid数据集位置，如果使用固定长度，需要指定固定长度大小。

