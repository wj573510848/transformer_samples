from transformers import XLNetTokenizer
import jieba
import os
import json
from tqdm import tqdm
import re
import random

from config import config_cpm_large_zero


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


def load_idiom_dict(path):
    with open(path, 'r', encoding='utf8') as f:
        idiom_dict = json.load(f)


def process(config=None):
    if config is None:
        config = config_cpm_large_zero()
    tokenizer = XLNetTokenizer.from_pretrained(config.pretrained_model_path)
    idiom_dict = load_idiom_dict(
        os.path.join(config.raw_data_dir, 'idiomDict.json'))
    # 零样本的场景，仅测试测试集
    for split in ['test']:
        raw_file = os.path.join(config.raw_data_dir, '{}.json'.format(split))
        ans_file = os.path.join(config.raw_data_dir,
                                '{}_answer.json'.format(split))
        with open(raw_file, 'r') as f:
            lines = f.readlines()
        with open(ans_file, 'r') as f:
            ans_d = json.load(f)
        all_data = {'contents': [], 'sids': [], 'labels': [], 'cids': []}
        sid = 0
        for line in tqdm(lines, desc="Preprocessing {}".format(split)):
            jobj = json.loads(line)
            for sent in jobj['content']:
                processed_results = process_one_sent_eval(
                    tokenizer=tokenizer,
                    sent=sent,
                    answers=ans_d,
                    candidates=jobj['candidates'])
                for sample in processed_results:
                    all_data['contents'].extend(sample['cands'])
                    all_data['sids'].extend([sid]*len(sample['cands']))
                    all_data['cids'].extend(list(range(len(sample['cands']))))
                    all_data['labels'].append(sample['truth'])
                    sid += 1
        for key in all_data:
            print("{}:{}".format(key,len(all_data[key])))
        cand_ids = random.choice(all_data['contents'])
        # print(tokenizer.decode(cand_ids))
        # print(random.choice(all_data['sids']))
        # print(random.choice(all_data['cids']))
        # print(random.choice(all_data['labels']))
        save_file = os.path.join(config.data_save_path,'{}.json'.format(split))
        with open(save_file,'w') as f:
            json.dump(all_data,f,indent=4,ensure_ascii=False)
    save_file = os.path.join(config.data_save_path,"idioms.json")
    with open(save_file,'w') as f:
        json.dump(idiom_dict,f)

if __name__ == '__main__':
    process()

# tokenizer = XLNetTokenizer.from_pretrained(config.pretrained_model_path)

# text = '这是个测试'

# a = tokenizer(text)
# b = tokenizer.decode(a['input_ids'])
# print(a)
# print(b)
