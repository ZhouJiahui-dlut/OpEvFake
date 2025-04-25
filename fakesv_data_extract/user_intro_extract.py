
# BERT
from transformers import BertTokenizer, BertModel, BertConfig
# model = RobertaModel.from_pretrained('roberta-large')
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import time
import pickle


def read_json(path):
    data = []
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            data.append(json.loads(line))
    return data

def str2num(str_x):
    if isinstance(str_x, float):
        return str_x
    elif str_x.isdigit():
        return int(str_x)
    elif 'w' in str_x:
        return float(str_x[:-1])*10000
    elif '亿' in str_x:
        return float(str_x[:-1])*100000000
    else:
        print ("error")
        print (str_x)

def pad_sequence(seq_len,video, emb):
    if isinstance(video, list):
        video = torch.stack(video)
    ori_len=video.shape[0]
    if ori_len == 0:
        video = torch.zeros([seq_len,emb],dtype=torch.long)
    elif ori_len>=seq_len:
        if emb == 200:
            video=torch.FloatTensor(video[:seq_len])
        else:
            video=torch.LongTensor(video[:seq_len])
    else:
        video=torch.cat([video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.long)],dim=0)
        if emb == 200:
            video=torch.FloatTensor(video)
        else:
            video=torch.LongTensor(video)
    result = video
    return result


class Config():
    def __init__(self):
        self.pad_size = 50
        self.batch_size = 8
        self.epochs = 1
        self.PTM = '/bert-base-chinese-local'

        self.device = 'cuda:0'
        # self.warmup_ratio = 0.1

config = Config()

bert = BertModel.from_pretrained(config.PTM)
bert_config = BertConfig.from_pretrained(config.PTM)

# if item['is_author_verified'] == 1:
intro_1 = "个人认证"
# elif item['is_author_verified'] == 2:
intro_2 = "机构认证"
# elif item['is_author_verified'] == 0:
intro_3 = "未认证"
# else:
intro_4 = "认证状态未知"

# 第一步，提取用户信息的特征
tokenizer = BertTokenizer.from_pretrained(config.PTM)

intro_tokens = tokenizer(intro_1, max_length=50, padding='max_length', truncation=True)
intro_inputid = torch.LongTensor(intro_tokens['input_ids']).unsqueeze(0)
intro_mask = torch.LongTensor(intro_tokens['attention_mask']).unsqueeze(0)
user_intro_1 = bert(input_ids=intro_inputid, attention_mask=intro_mask)[1] # [batch_size, 1024]最后一个隐藏层cls的输出

intro_tokens = tokenizer(intro_2, max_length=50, padding='max_length', truncation=True)
intro_inputid = torch.LongTensor(intro_tokens['input_ids']).unsqueeze(0)
intro_mask = torch.LongTensor(intro_tokens['attention_mask']).unsqueeze(0)
user_intro_2 = bert(input_ids=intro_inputid, attention_mask=intro_mask)[1]

intro_tokens = tokenizer(intro_3, max_length=50, padding='max_length', truncation=True)
intro_inputid = torch.LongTensor(intro_tokens['input_ids']).unsqueeze(0)
intro_mask = torch.LongTensor(intro_tokens['attention_mask']).unsqueeze(0)
user_intro_3 = bert(input_ids=intro_inputid, attention_mask=intro_mask)[1]

intro_tokens = tokenizer(intro_4, max_length=50, padding='max_length', truncation=True)
intro_inputid = torch.LongTensor(intro_tokens['input_ids']).unsqueeze(0)
intro_mask = torch.LongTensor(intro_tokens['attention_mask']).unsqueeze(0)
user_intro_4 = bert(input_ids=intro_inputid, attention_mask=intro_mask)[1]

user_intro = torch.cat([user_intro_1, user_intro_2, user_intro_3, user_intro_4], dim=0)
with open('data\\user_intro.pkl', 'wb') as f:
    pickle.dump(user_intro, f)
    print("保存成功")



# 第二步，遍历数据，填充用户信息，划分train/val/test
user_intro = {}
with open('data\\user_intro.pkl', 'rb') as f:
    intro_list = pickle.load(f)
with open('data\\data_train_list.pkl', 'rb') as f:
    fakesv_data_new = pickle.load(f)
for item in fakesv_data_new:
    if item['is_author_verified'] == 1:
        intro = intro_list[0]
    elif item['is_author_verified'] == 2:
        intro = intro_list[1]
    elif item['is_author_verified'] == 0:
        intro = intro_list[2]
    else:
        intro = intro_list[3]
    user_intro[item['video_id']] = intro.unsqueeze(0)
with open('data\\user_intro_3624\\user_intro_train.pkl', 'wb') as f:
    pickle.dump(user_intro, f)
    print("保存成功")
