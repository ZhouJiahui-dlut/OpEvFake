
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

class BaselineData(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.pad_size = config.pad_size
        self.max_sentence_len = 0

        self.config = config
        self.bert = BertModel.from_pretrained(config.PTM)
        self.bert_config = BertConfig.from_pretrained(config.PTM)

        self.key_list = list(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key = self.key_list[idx]
        sentence = self.data[key]

        input_ids, attention_mask = self.__convert_to_id__(sentence)

        return torch.tensor(input_ids), torch.tensor(attention_mask), key

    def __convert_to_id__(self, sentence):
        ids = self.tokenizer.encode_plus(sentence, max_length=config.pad_size, truncation=True)
        input_ids = self.__padding__(ids['input_ids'])
        attention_mask = self.__padding__(ids['attention_mask'])

        return input_ids, attention_mask

    def __padding__(self, sentence):
        if self.max_sentence_len < len(sentence):
            self.max_sentence_len = len(sentence)
            print(self.max_sentence_len)
        sentence = sentence[:self.pad_size]
        sentence = sentence + [0] * (self.pad_size - len(sentence))
        return sentence

class Config():
    def __init__(self):
        self.pad_size = 512
        self.batch_size = 8
        self.epochs = 1
        self.PTM = '/bert-base-chinese-local'

        self.device = 'cuda:0'


config = Config()

vid_list = []
with open('/implicit_opinion_train.pkl', 'rb') as f: # 读取LLM生成的隐藏观点
    fakesv_data_new = pickle.load(f)


tokenizer = BertTokenizer.from_pretrained(config.PTM)

dataloader = DataLoader(BaselineData(fakesv_data_new, tokenizer, config), batch_size=config.batch_size)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.PTM)
        self.bert_config = BertConfig.from_pretrained(config.PTM)

    def forward(self, x):
        x_bert = self.bert(input_ids=x[0], attention_mask=x[1]).last_hidden_state # [batch_size, 100, 1024]最后一个隐藏层的序列的输出

        return x_bert, x[2]

Model = Model(config)

text_data = {}
Model.eval()
with torch.no_grad():
    for idx, data in enumerate(dataloader):
        y, vid = Model(data)

        for idx, data_vid in enumerate(vid):
            text_data[data_vid] = y[idx]
with open('/data/gpt_description_3624/gpt_description_train', 'wb') as f:
    pickle.dump(text_data, f)
    print("保存成功")
