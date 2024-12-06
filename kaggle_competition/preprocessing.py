#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:15:18 2024

@author: r12528053
"""

import os
import re
import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt

## ignore warnings
import warnings
warnings.filterwarnings('ignore')

def plot_dis(df) :
    #Visualize the distribution of emotion
    labels = df['emotions'].unique()
    total_length = len(df['emotions'])

    df_1 = df.groupby(['emotions']).count()['text']
    df_1 = df_1.apply(lambda x : round(x * 100/ total_length, 3))
    
    fig, ax = plt.subplots(figsize=(10,5))
    plt.bar(df_1.index, df_1.values)

    plt.ylabel('% of instances')
    plt.xlabel('Emotion')
    plt.title('Emotion distribution')
    plt.grid(True)
    plt.show()


def remove_emoji_and_lh(text):
    # Regular expression for emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        "\U00002500-\U00002BEF"  # Chinese characters
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "\U0001F926-\U0001F937"  # Supplementary symbols
        "\U00010000-\U0010FFFF"  # Unicode emojis
        "\u200d"                 # Zero-width joiners
        "\u2640-\u2642"          # Gender symbols
        "\u2600-\u2B55"          # Misc symbols
        "\u23cf"                 # Misc technical
        "\u23e9"                 # Misc arrows
        "\u231a"                 # Watch
        "\ufe0f"                 # Dingbats
        "\u3030"                 # Misc symbols
        "]+", 
        flags=re.UNICODE
    )

    # Remove emojis
    text = emoji_pattern.sub(r'', text)

    # Remove '<LH>'
    text = re.sub(r'<LH>', '', text)

    # Remove extra spaces left
    return text.strip()

train_df_1 = pd.read_excel("./train_1_1.xlsx")
train_df_2 = pd.read_excel("./train_1_2.xlsx")
train_df   = pd.concat([train_df_1, train_df_2])
plot_dis(train_df)

print (train_df.isnull().sum())
clean_train_df = train_df.dropna()

clean_train_df['text_clean'] = [remove_emoji_and_lh(sentence) for sentence in clean_train_df['text']]

clean_train_df['text_tokenized'] = clean_train_df['text_clean'].apply(lambda x: nltk.word_tokenize(x))
print(clean_train_df.head())

training_corpus = clean_train_df['text_tokenized'].values
print (training_corpus[:3])

clean_train_df.to_csv("./final_train.csv")

test_df = pd.read_excel("./test_1.xlsx")
print (test_df.isnull().sum())

test_id_list = test_df['id'].tolist()
index_1      = test_id_list.index('0x2172bc')
index_2      = test_id_list.index('0x33b84f')
index_3      = test_id_list.index('0x38c4eb')

test_df["text"][index_1] = " #IDWP  P E A C E ~ PEACE is a PERFECT-WORK of  MERCY!  PEACE is also RADIANCE shown forth as LOVING KINDNESS!  BE #RADIANT BE <LH> #BE"
test_df["text"][index_2] = " Infringement Of/On My Human Rights. #Hey #Headturns"
test_df["text"][index_3] = " He Is my Refuge and Fortress I Will in Him "

print ("After process", test_df.isnull().sum())

test_df['text_clean'] = [remove_emoji_and_lh(sentence) for sentence in test_df['text']]
print (test_df.head())

test_df.to_csv("./final_test.csv")