from transformers import XLNetTokenizer
import jieba
import os
import json
from tqdm import tqdm
import re
import random

from config import config_cpm_large_zero_fixed_length


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


def load_idiom_dict(path):
    with open(path, 'r', encoding='utf8') as f:
        idiom_dict = json.load(f)


def process(config=None):
    if config is None:
        config = config_cpm_large_zero_fixed_length()
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
                    candidates=jobj['candidates'],
                    max_length=config.max_length)
                for sample in processed_results:
                    all_data['contents'].extend(sample['cands'])
                    all_data['sids'].extend([sid] * len(sample['cands']))
                    all_data['cids'].extend(list(range(len(sample['cands']))))
                    all_data['labels'].append(sample['truth'])
                    sid += 1
        for key in all_data:
            print("{}:{}".format(key, len(all_data[key])))
        cand_ids = random.choice(all_data['contents'])
        print(len(cand_ids))
        print(tokenizer.decode(cand_ids))
        print(random.choice(all_data['sids']))
        print(random.choice(all_data['cids']))
        print(random.choice(all_data['labels']))
        save_file = os.path.join(config.data_save_path,
                                 '{}_{}.json'.format(split,config.max_length))
        with open(save_file, 'w') as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
    save_file = os.path.join(config.data_save_path, "idioms.json")
    with open(save_file, 'w') as f:
        json.dump(idiom_dict, f)


if __name__ == '__main__':
    process()

# tokenizer = XLNetTokenizer.from_pretrained(config.pretrained_model_path)

# text = '这是个测试'

# a = tokenizer(text)
# b = tokenizer.decode(a['input_ids'])
# print(a)
# print(b)
