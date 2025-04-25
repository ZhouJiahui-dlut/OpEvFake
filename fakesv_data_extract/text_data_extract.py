
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
        self.pad_size = config.pad_size  #510
        self.max_sentence_len = 0

        self.config = config
        self.bert = BertModel.from_pretrained(config.PTM)
        self.bert_config = BertConfig.from_pretrained(config.PTM)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        comments_like = []
        for num in self.data[idx]['count_comment_like']:
            num_like = num.split(" ")[0]
            comments_like.append(str2num(num_like))

        comments_inputid = []
        comments_mask = []

        for comment in self.data[idx]['comments']:
            comment_tokens = self.__convert_to_id__(comment)
            comments_inputid.append(comment_tokens[0])
            comments_mask.append(comment_tokens[1])
        comments_inputid = torch.LongTensor(np.array(comments_inputid))
        comments_mask = torch.LongTensor(np.array(comments_mask))


        num_comments = 23
        comments_inputid = pad_sequence(num_comments, comments_inputid, 250)
        comments_mask = pad_sequence(num_comments, comments_mask, 250)
        if len(comments_like) >= num_comments:
            comments_like = torch.tensor(comments_like[:num_comments])
        else:
            comments_like = torch.tensor(comments_like + [0] * (num_comments - len(comments_like)))


        return {
            'comments_inputid': comments_inputid,
            'comments_mask': comments_mask,
            'comments_like': comments_like,
            'video_id': self.data[idx]['video_id']
        }
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
        self.pad_size = 250
        self.batch_size = 8
        self.epochs = 1
        self.PTM = '/bert-base-chinese-local'

        self.device = 'cuda:0'


config = Config()

vid_list = []
with open('data\\temporal_new\\vid_time3_val_new_546_273.txt', 'r', encoding='utf-8') as f:
    for vid in f.readlines():
        vid = vid.strip('\n')       #去除文本中的换行符
        vid_list.append(vid)
with open('data\\data_val_list.pkl', 'rb') as f: # 根据VID在fakeSV提供的完整数据data.json中筛选出的有效数据
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
        # x = self.bert(input_ids=x[0], attention_mask=x[1]).last_hidden_state # [batch_size, 100, 1024]最后一个隐藏层的序列的输出
        # x = self.bert(input_ids=x[0], attention_mask=x[1]).pooler_output # [batch_size, 1024]最后一个隐藏层cls的输出
        comments_feature = torch.empty(size=(0,23,768), dtype=torch.float32)
        for i in range(x['comments_inputid'].shape[0]):
            bert_fea = self.bert(input_ids=x['comments_inputid'][i], attention_mask=x['comments_mask'][i])[1]
            bert_fea = bert_fea.unsqueeze(0)
            comments_feature = torch.cat([comments_feature, bert_fea])
        # comments_feature = torch.stack(bert_fea)
        fea_comments = torch.empty(size=(0,768), dtype=torch.float32)
        for v in range(x['comments_like'].shape[0]):
            comments_weight = torch.stack(
                [torch.true_divide((i + 1), (x['comments_like'][v].shape[0] + x['comments_like'][v].sum())) for i in
                 x['comments_like'][v]])
            # test1 = comments_feature[v].transpose(2,0)
            # test2 = comments_weight.reshape(1, comments_weight.shape[0])
            comments_fea_reweight = torch.sum(comments_feature[v] * (comments_weight.reshape(comments_weight.shape[0],1)), dim=0)
            # fea_comments.append(comments_fea_reweight.transpose(1,0))
            comments_fea_reweight = comments_fea_reweight.unsqueeze(0)
            fea_comments = torch.cat([fea_comments, comments_fea_reweight])
        # fea_comments = torch.stack(fea_comments)
        return fea_comments, x['video_id']

Model = Model(config)
# text_data = np.empty(shape=(0,250,768))
text_data = {}
Model.eval()
with torch.no_grad():
    for idx, data in enumerate(dataloader):
    # for data in fakesv_data_new:

        y, vid = Model(data)

        for idx, data_vid in enumerate(vid):
            text_data[data_vid] = y[idx]
with open('E:\\fakesv数据集\\new\\comments_3624\\comments_val.pkl', 'wb') as f:
    pickle.dump(text_data, f)
    print("保存成功")

