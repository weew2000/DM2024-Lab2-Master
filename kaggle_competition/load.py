#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:32:08 2024

@author: r12528053
"""

import os
import pandas as pd
import json
import numpy as np

sample_path   = "./sampleSubmission.csv"
emotion_path  = "./emotion.csv"
identify_path = "./data_identification.csv"

tweet_path = "./tweets_DM.json"

sample_df   = pd.read_csv(sample_path)
emotion_df  = pd.read_csv(emotion_path)
identify_df = pd.read_csv(identify_path)

with open(tweet_path, 'r') as f:
    tweet = [json.loads(line) for line in f]
    
id_list, text_list = [], []

for i in range(len(tweet)):
    id = tweet[i]["_source"]["tweet"]["tweet_id"]
    text = tweet[i]["_source"]["tweet"]["text"]
    
    id_list.append(id)
    text_list.append(text)

#Check whether the sample_df contains all test texts
sample_id  = sample_df['id'].tolist()
sample_emotion = sample_df['emotion'].tolist()

emotion_id = emotion_df['tweet_id'].tolist()
emotion_emotion = emotion_df['emotion'].tolist()

Train = {'emotions':[], 'text':[]}
Test  = {'id': [], 'emotions': [], 'text': []}

for i in range(len(id_list)):
    print (i)
    id   = id_list[i]
    text = text_list[i]
    
    if id in sample_id:
        idx = sample_id.index(id)
        emotion = sample_emotion[idx]
        
        Test['id'].append(id)
        Test['emotions'].append(emotion)
        Test['text'].append(text)
        
    elif id in emotion_id:
        idx = emotion_id.index(id)
        emotion = emotion_emotion[idx]
        
        Train['emotions'].append(emotion)
        Train['text'].append(text)
        
    else: print (id)

# Convert to DataFrame
train_df = pd.DataFrame(Train)
train_df.to_excel("/mnt/NFS2/r12528053/dmhw_2/train_1.xlsx")

test_df = pd.DataFrame(Test)

test_df.to_excel("/mnt/NFS2/r12528053/dmhw_2/test_1.xlsx")
