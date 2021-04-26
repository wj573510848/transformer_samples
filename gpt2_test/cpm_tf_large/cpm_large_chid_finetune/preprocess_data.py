from transformers import XLNetTokenizer
import jieba
import os
import json
from tqdm import tqdm
import re
import random

from config import basic_config

# 将chid数据组合为：上下文: predix <mask> postfix <eod> 选项0: I_0 <sep> 选项1: I_1 <sep> ... 选项9: I_9 <sep> <mask> 答案是: L <eod>


class XLNetTokenizer(XLNetTokenizer):
    translator = str.maketrans(" \n", "\u2582\u2583")

    def _tokenize(self, text, *args, **kwargs):
        text = [
            x.translate(self.translator)
            for x in jieba.cut(text, cut_all=False)
        ]
        text = " ".join(text)
        return super()._tokenize(text, *args, **kwargs)

    def _decode(self, *args, **kwargs):
        text = super()._decode(*args, **kwargs)
        text = text.replace(' ', '').replace('\u2582',
                                             ' ').replace('\u2583', '\n')
        return text


def process_one_sent(tokenizer, sent, answers, candidates, max_length):
    '''
    tokenizer: 分词器
    sent: 带有成语标识的句子
    answer:dict,key: 成语标识 valu: 候选集对应的序号
    candidates: 候选集，10个候选成语
    max_length: 最大长度
    将chid数据组合为：上下文: predix <mask> postfix <eod> 选项0: I_0 <sep> 选项1: I_1 <sep> ... 选项9: I_9 <sep> <mask> 答案是: L <eod>
    '''
    
    pattern = re.compile(r"#idiom\d+#")

    eod_id = tokenizer.encode('<eod>')[0]
    sep_id = tokenizer.encode('<sep>')[0]
    mask_id = tokenizer.encode('<mask>')[0]

    cand_string = ''
    for i in range(len(candidates)):
        cand_string += "选项{}:{}<sep>".format(i,candidates[i])
    cand_string += "答案是:"
    cand_ids = tokenizer.encode(cand_string)[:-2] # len: 95
    start_ids = tokenizer.encode("上下文:")[:-2] # len: 4
    
    sent_max_length = max_length - len(cand_ids) - len(start_ids) - 4
    assert sent_max_length > 0

    matches = pattern.finditer(sent)
    for m in matches:
        prefix = pattern.sub('', sent[:m.start()])
        postfix = pattern.sub('', sent[m.end():])
        prefix_ids = tokenizer.encode(prefix)[:-2]
        post_ids = tokenizer.encode(postfix)[:-2]
        if len(prefix_ids) >= sent_max_length:
            prefix_ids = prefix_ids[len(prefix_ids)-sent_max_length:]
            post_ids = []
        else:
            post_ids = post_ids[:sent_max_length-len(prefix_ids)]
        ans_string = str(answers[m.group()]) + "<eod>"
        ans_ids = tokenizer.encode(ans_string)[:-2] 
        ids = start_ids + prefix_ids + [mask_id] + post_ids + [eod_id] + cand_ids + ans_ids
        assert len(ids) <= max_length
        # print(len(ids))
        # print(tokenizer.decode(ids))
        yield  ids

def preprocess(config=None):
    if config is None:
        config = basic_config()
    tokenizer = XLNetTokenizer.from_pretrained(config.pretrained_model_path)
    for split in ['test', 'dev', 'train']:
        raw_data_file = os.path.join(config.raw_data_dir,
                                     '{}.json'.format(split))
        raw_ans_file = os.path.join(config.raw_data_dir,
                                    '{}_answer.json'.format(split))
        save_file = os.path.join(config.data_save_path,
                                 '{}.json'.format(split))

        with open(raw_data_file, 'r') as f:
            raw_data_lines = f.readlines()
        with open(raw_ans_file, 'r') as f:
            ans_d = json.load(f)

        with open(save_file, 'w') as f:
            for line in tqdm(raw_data_lines,
                             desc="Preprocessing {}".format(split)):
                line_json = json.loads(line)
                contents = line_json['content']
                candidates = line_json['candidates']
                for sent in contents:
                    results = process_one_sent(tokenizer=tokenizer,
                                               sent=sent,
                                               answers=ans_d,
                                               candidates=candidates,
                                               max_length=config.max_length)
                    for ids in results:
                        f.write(json.dumps(ids))
                        f.write("\n")
if __name__=="__main__":
    preprocess()
