import math
import os
import pickle

import h5py
import jieba
import jieba.analyse as analyse
import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import BertTokenizer
from torchvision import models
from transformers import BertModel, BertTokenizer
# from src.models import MULTModel
# from src.main import hyp_params
# avgpool = models.vgg19(pretrained=True).avgpool.cuda()
# classifier = models.vgg19(pretrained=True).classifier[:4].cuda()
import argparse
# from src.utils import *
from torch.utils.data import DataLoader
# from src import train


# 得到一个视频对应的所有数据
class SVFENDDataset(Dataset):

    def __init__(self, datamode='title+ocr', train_or_test='train'):  #标题+转录

        # 读取各模态特征
        #音频特征vggish
        with open(os.path.join('/data/audio', 'audio_'+train_or_test+'.pkl'), "rb") as fr:
            audio = pickle.load(fr)

        # 文本特征
        if datamode == 'title':
            with open(os.path.join('/data/text_title_temporal', 'text_title_lhs_'+train_or_test+'.pkl'), "rb") as fr:
                self.text = pickle.load(fr)
        elif datamode == 'title+ocr':
            with open(os.path.join('/data/text_title_ocr_temporal', 'text_title_ocr_lhs_'+train_or_test+'.pkl'), "rb") as fr:
                self.text = pickle.load(fr)

        with open(os.path.join('/data/comments', 'comments_' + train_or_test + '.pkl'), "rb") as fr:
            self.comments = pickle.load(fr)

        # gpt生成的文本分析
        with open(os.path.join('/data/gpt_description', 'gpt_description_' + train_or_test + '.pkl'), "rb") as fr:
            self.gpt_description = pickle.load(fr)

        # label
        with open(os.path.join('/data/label', 'label_'+train_or_test+'.pkl'), "rb") as fr:
            self.label = pickle.load(fr)

        #vgg9视频帧特征
        with open(os.path.join('/data/video', 'video_'+train_or_test+'.pkl'), "rb") as fr:
            self.video = pickle.load(fr)

        # user_intro
        with open(os.path.join('/data/user_intro', 'user_intro_'+train_or_test+'.pkl'), "rb") as fr:
            self.user_intro = pickle.load(fr)

        # vid
        with open(os.path.join('/data/vid', 'vid_'+train_or_test+'.pkl'), "rb") as fr:
            self.vid = pickle.load(fr)

        # c3d
        with open(os.path.join('/data/c3d', 'c3d_'+train_or_test+'.pkl'), "rb") as fr:
            self.c3d = pickle.load(fr)

        self.audio = dict(filter(lambda item: item[0] in self.vid, audio.items()))


        
    def __len__(self):
        return len(self.label)
     
    def __getitem__(self, idx):
        # item = self.data.iloc[idx] #根据索引idx从数据集中获取对应的样本
        vid = self.vid[idx]

        text = torch.tensor(self.text[idx], dtype=torch.float32)
        # comments = torch.tensor(self.comments[vid], dtype=torch.float32)
        comments = self.comments[vid]
        gpt_description = self.gpt_description[vid]

        audio = self.audio[vid]

        video = self.video[vid]

        c3d = self.c3d[vid]

        label = self.label[vid]
        label = torch.tensor(label)

        user_intro = self.user_intro[vid]

        audio = torch.tensor(audio, dtype=torch.float32)
        video = torch.tensor(video, dtype=torch.float32)
        c3d = torch.tensor(c3d, dtype=torch.float32)


        return {
            'label': label,  # 标签
            'text': text,
            'audioframes': audio,  # 音频帧
            'frames': video,  # 帧
            'comments': comments, # 评论
            'c3d': c3d,  # C3D特征
            'user_intro': user_intro,
            'gpt_description': gpt_description # gpt生成的文本辅助分析
        }

def pad_sequence(seq_len,lst, emb):
    result=[]
    for video in lst:
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
        result.append(video)
    return torch.stack(result)

def pad_frame_sequence(seq_len,lst):
    attention_masks = []
    result=[]
    for video in lst:
        # video=torch.FloatTensor(video)
        ori_len=video.shape[0]
        video = video.squeeze()
        if ori_len>=seq_len:
            gap=ori_len//seq_len
            video=video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video=torch.cat((video, torch.zeros([seq_len-ori_len, video.shape[1]], dtype=torch.float32)), dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len-ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)

def SVFEND_collate_fn(batch):
    # num_comments = 23
    num_frames = 83
    num_audioframes = 50


    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence(num_frames, frames)
    frames = frames.squeeze()

    audioframes = [item['audioframes'] for item in batch]
    audioframes, audioframes_masks = pad_frame_sequence(num_audioframes, audioframes)

    comments = [item['comments'] for item in batch]
    comments = torch.stack(comments)

    gpt_description = [item['description'] for item in batch]
    gpt_description = torch.stack(gpt_description)

    user_intro = [item['user_intro'] for item in batch]
    user_intro = torch.stack(user_intro)

    c3d = [item['c3d'] for item in batch]
    c3d, c3d_masks = pad_frame_sequence(num_frames, c3d)

    label = [item['label'] for item in batch]
    text = [item['text'] for item in batch]
    text = torch.tensor([item.cpu().detach().numpy() for item in text])

    return {
        'label': torch.stack(label),
        'text': text,
        'audioframes': audioframes,
        'audioframes_masks': audioframes_masks,
        'frames': frames,
        'frames_masks': frames_masks,
        'comments': comments,
        'c3d': c3d,
        'c3d_masks': c3d_masks,
        'user_intro': user_intro,
        'gpt_description': gpt_description,
    }

def _init_fn(worker_id):
    np.random.seed(2022)

def get_dataloader(modelConfig,data_type='SVFEND'):
    collate_fn=None

    if data_type == 'SVFEND':
        dataset_train = SVFENDDataset(datamode=modelConfig["datamode"], train_or_test='train')
        dataset_val = SVFENDDataset(datamode=modelConfig["datamode"], train_or_test='val')
        dataset_test = SVFENDDataset(datamode=modelConfig["datamode"], train_or_test='test')
        collate_fn=SVFEND_collate_fn

    train_dataloader = DataLoader(dataset_train, batch_size=modelConfig["batch_size"],
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=_init_fn,
        collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset_val, batch_size=modelConfig["batch_size"],
                                 num_workers=0,
                                 pin_memory=True,
                                 shuffle=False,
                                 worker_init_fn=_init_fn,
                                 collate_fn=collate_fn)
    test_dataloader=DataLoader(dataset_test, batch_size=modelConfig["batch_size"],
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        worker_init_fn=_init_fn,
        collate_fn=collate_fn)

    dataloaders = dict(zip(['train', 'val', 'test'], [train_dataloader, val_dataloader, test_dataloader]))

    return dataloaders

def split_word(df):  #去除停用词
    title = df['description'].values
    comments = df['comments'].apply(lambda x:' '.join(x)).values
    text = np.concatenate([title, comments],axis=0)
    analyse.set_stop_words('./data/stopwords.txt')
    all_word = [analyse.extract_tags(txt) for txt in text.tolist()]
    corpus = [' '.join(word) for word in all_word]
    return corpus