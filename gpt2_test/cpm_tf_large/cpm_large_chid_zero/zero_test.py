from models import ChidGPT2Model
import os
import numpy as np
import json
import pickle
from tqdm import tqdm
import pandas as pd

from config import config_cpm_large_zero
from preprocess_data import process

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def load_data(config=None):
    if not config:
        config = config_cpm_large_zero()
    test_file = os.path.join(config.data_save_path,'test.json')
    if not os.path.isfile(test_file):
        process(config)
    with open(test_file,'r') as f:
        raw_data = json.load(f)
    eod_id = 7
    pad_id = 0
    label_pad_id = -100
    sids = raw_data['sids']
    cids = raw_data['cids']
    label_ids = raw_data['labels']
    cand_ids = raw_data['contents']
    batch_size = config.batch_size
    
    def batch_gen():
        start_index = 0
        while start_index < len(cand_ids):
            input_ids = cand_ids[start_index:start_index+batch_size]
            max_length = max([len(i) for i in input_ids]) - 2
            batch_inputs = []
            batch_labels = []
            for ids in input_ids:
                ids = ids[:-2]
                label = ids[1:] + [eod_id]
                assert len(ids) == len(label)
                if len(ids) < max_length:
                    ids = ids + [pad_id] * (max_length - len(ids))
                    label = label + [label_pad_id] * (max_length - len(label))
                batch_inputs.append(ids)
                batch_labels.append(label)
                assert len(label) == max_length
            batch_inputs = np.array(batch_inputs,dtype=np.int32)
            batch_labels = np.array(batch_labels,dtype=np.int32)
            yield batch_inputs, batch_labels
            start_index += batch_size
    return sids,cids,label_ids,batch_gen

def test():
    config = config_cpm_large_zero()

    loss_file = os.path.join(config.data_save_path,'loss_1.pkl')
    if not os.path.isfile(loss_file):
        sids,cids,true_labels,batch_gen = load_data(config)
        model = ChidGPT2Model.from_pretrained(config.pretrained_model_path)
        total_loss = []
        
        with tqdm(total=int(len(sids)/config.batch_size)) as pbar:
            for input_ids, label_ids in batch_gen():
                losses = model(input_ids=input_ids,labels=label_ids)
                total_loss.append(losses.numpy())
                pbar.update(1)
                
        print("save!")
        with open(loss_file,'wb') as f:
            pickle.dump((sids,cids,true_labels,total_loss),f,-1)
    with open(loss_file,'rb') as f:
        sids,cids,true_labels,total_loss = pickle.load(f)
    sids = np.array(sids)
    cids = np.array(cids)
    true_labels = np.array(true_labels)
    all_loss = []
    for i in total_loss:
        all_loss.extend(i)
    df = pd.DataFrame({'sids':sids,"cids":cids,'loss':all_loss})
    df['min_loss'] = df.groupby('sids')['loss'].transform('min')
    df1 = df[df['loss']==df['min_loss']]
    df1['true_labels'] = true_labels
    rate = df1[df1['true_labels']==df1['cids']].shape[0]/df1.shape[0]
    print(rate) # 0.6792

if __name__=="__main__":
    test()