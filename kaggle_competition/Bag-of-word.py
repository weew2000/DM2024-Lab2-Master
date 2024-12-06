#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:24:21 2024

@author: r12528053
"""

import os
import pandas as pd
import nltk
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
    
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_df   = pd.read_csv("./final_train.csv")
print (train_df.isnull().sum())
clean_train_df = train_df.dropna()
print (clean_train_df.isnull().sum())

test_df    = pd.read_excel("./test_1.xlsx")
print (test_df.isnull().sum())

test_id_list = test_df['id'].tolist()
index_1      = test_id_list.index('0x2172bc')
index_2      = test_id_list.index('0x33b84f')
index_3      = test_id_list.index('0x38c4eb')

test_df["text"][index_1] = " #IDWP  P E A C E ~ PEACE is a PERFECT-WORK of  MERCY!  PEACE is also RADIANCE shown forth as LOVING KINDNESS!  BE #RADIANT BE <LH> #BE"
test_df["text"][index_2] = " Infringement Of/On My Human Rights. #Hey #Headturns"
test_df["text"][index_3] = " He Is my Refuge and Fortress I Will in Him "

print ("After process", test_df.isnull().sum())

stopwords_nltk = nltk.corpus.stopwords
stop_words = stopwords_nltk.words('english')
print(stop_words[:5])

#Make the text into the vector using sklearn countvectorizer
BOW_5000 = CountVectorizer(max_features = 500, tokenizer=nltk.word_tokenize)

BOW_5000.fit(clean_train_df['text_clean'])

train_data_BOW_features = BOW_5000.transform(clean_train_df['text_clean'])
test_data_BOW_features  = BOW_5000.transform(test_df['text'])

print (train_data_BOW_features.shape)

x_train = train_data_BOW_features
y_train = clean_train_df['emotions']

x_test = test_data_BOW_features
y_test = test_df['emotions']

print ("x_train shape", x_train.shape, "y_train shape", y_train.shape)
print ("x_test shape", x_test.shape, "y_test shape", y_test.shape)


#Decision tree
DT_model = DecisionTreeClassifier(random_state = 53)
DT_model = DT_model.fit(x_train, y_train)

y_test_pred = DT_model.predict(x_test)
test_df['Decision tree'] = y_test_pred


#Random forest
RF_model = RandomForestClassifier(random_state=53)
RF_model = RF_model.fit(x_train, y_train)

RF_test_pred = RF_model.predict(x_test)
test_df['Random forest'] = RF_test_pred


#Naive bayes classifier
NB_model = MultinomialNB()
NB_model.fit(x_train, y_train)

NB_test_pred = NB_model.predict(x_test)
test_df['NB'] = NB_test_pred


#Deal with categorical variable
def encode(le, labels):
    enc = le.transform(labels)
    return tf.keras.utils.to_categorical(enc)

def decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis = 1)
    return le.inverse_transform(dec)

label_encoder = LabelEncoder()
label_encoder.fit(y_train)

y_train_one_hot = encode(label_encoder, y_train)

print ("one hot y_train shape", y_train_one_hot.shape)

#ANN using tensorflow
input_shape = x_train.shape[1]
output_shape = len(label_encoder.classes_)
print ("input shape", input_shape, "output shape", output_shape)

input = tf.keras.layers.Input(shape = (input_shape, ))
x     = tf.keras.layers.Dense(64, activation = 'relu')(input)
x     = tf.keras.layers.Dense(64, activation = 'relu')(x)
output= tf.keras.layers.Dense(output_shape, activation = 'softmax')(x) 

tf_model = tf.keras.Model(inputs=input, outputs = output)    
tf_model.compile(optimizer = "adam",
                 loss      = "categorical_crossentropy",
                 metrics   = "accuracy")

csv_logger = tf.keras.callbacks.CSVLogger('./training_log_1201.csv')

history = tf_model.fit(x_train, y_train_one_hot,
                       epochs     = 80,
                       batch_size = 128, 
                       callbacks = [csv_logger],)

pred_result   = tf_model.predict(x_test, batch_size = 128)
tf_pred_result= decode(label_encoder, pred_result)

test_df['tf_model'] = tf_pred_result

log_df = pd.read_csv('./training_log_1201.csv')

train_loss = log_df['loss'].tolist()
train_acc  = log_df['accuracy'].tolist()

plt.figure(figsize = (10,5))
plt.title("Training loss")
plt.plot(train_loss, label="train")
plt.xlabel("Epoch")
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize = (10,5))
plt.title("Training acc")
plt.plot(train_acc, label="train")
plt.xlabel("Epoch")
plt.ylabel('Acc')
plt.legend()
plt.show()



#%%
sample_df = pd.read_csv("./sampleSubmission.csv")

id_list   = test_df['id'].tolist()
pred_list_dt = test_df['Decision tree'].tolist()
pred_list_rf = test_df['Random forest']
pred_list_nb = test_df['NB'].tolist()
pred_list_tf = test_df['tf_model'].tolist()

sample_dt, sample_rf, sample_nb, sample_tf, sample_ensemble = [], [], [], [], []

for i in range(len(sample_df['id'])):
    pred = []
    
    id = sample_df['id'][i]
    print (i, id)
    
    index        = id_list.index(id)
    pred_emotion_dt = pred_list_dt[index]
    pred_emotion_rf = pred_list_rf[index]
    pred_emotion_nb = pred_list_nb[index]
    pred_emotion_tf = pred_list_tf[index]
    
    sample_dt.append(pred_emotion_dt)
    sample_rf.append(pred_emotion_rf)
    sample_nb.append(pred_emotion_nb)
    sample_tf.append(pred_emotion_tf)
    
    pred.append(pred_emotion_dt)
    pred.append(pred_emotion_rf)
    pred.append(pred_emotion_nb)
    pred.append(pred_emotion_tf)
    
    counter = Counter(pred)
    most_common = counter.most_common(1)
    
    most_common = most_common[0][0]    
    
sample_df['dt'] = sample_dt
sample_df['nb'] = sample_nb
sample_df['tf'] = sample_tf

sample_df.to_csv("pred_nltk_BOW_500.csv")
