#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 00:18:53 2024

@author: r12528053
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers import RobertaTokenizerFast, RobertaModel
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_masks': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


def tokenize_bert(data,max_len) :
    input_ids = []
    attention_masks = []
    for i in range(len(data)):
        encoded = tokenizer_bert.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)


train_df   = pd.read_csv("./final_train.csv")
print (train_df.isnull().sum())
clean_train_df = train_df.dropna()
print (clean_train_df.isnull().sum())

test_df    = pd.read_csv("./final_test.csv")
print (test_df.isnull().sum())

x = clean_train_df['text_clean'].values
y = clean_train_df['emotions'].values
x_train, x_valid, y_train, y_valid = train_test_split(x, y, stratify = y, random_state = 53)

x_test = test_df['text_clean'].values

print ("x_train shape", x_train.shape, "y_train shape", y_train.shape)
print ("x_valid shape", x_valid.shape, "y_valid shape", y_valid.shape)
print ("x_test shape", x_test.shape)

ohe = OneHotEncoder()
y_train = ohe.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
y_valid = ohe.fit_transform(np.array(y_valid).reshape(-1, 1)).toarray()

tokenizer_bert = RobertaTokenizerFast.from_pretrained("roberta-base")

token_lens = []

for txt in x_train:
    tokens = tokenizer_bert.encode(txt, max_length=512, truncation=True)
    token_lens.append(len(tokens))
max_length=np.max(token_lens)

MAX_LEN=max_length

# Data preparation (placeholders for the actual data)
train_input_ids, train_attention_masks = tokenize_bert(x_train, MAX_LEN)
val_input_ids, val_attention_masks = tokenize_bert(x_valid, MAX_LEN)

train_dataset = CustomDataset(train_input_ids, train_attention_masks, y_train)
val_dataset = CustomDataset(val_input_ids, val_attention_masks, y_valid)

train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=30, shuffle=False)

#%%
x_test = test_df['text'].values

token_lens = []

for txt in x_test:
    tokens = tokenizer_bert.encode(txt, max_length=512, truncation=True)
    token_lens.append(len(tokens))
max_length_test=np.max(token_lens)

MAX_LEN_test=max_length_test

test_input_ids, test_attention_masks = tokenize_bert(x_test, MAX_LEN_test)
test_dataset = CustomDataset(test_input_ids, test_attention_masks, np.zeros(len(test_input_ids)))
test_loader = DataLoader(test_dataset, batch_size = 30)

#%%
class RobertaClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(RobertaClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_masks):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_masks, return_dict=False)
        output = self.dropout(pooled_output)
        return self.fc(output)

# Model setup
roberta_model = RobertaModel.from_pretrained('roberta-base')
num_classes = 8
model = RobertaClassifier(roberta_model, num_classes)

#criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-7)

# Training loop
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 5
for epoch in range(epochs):
    print (epoch)
    
    model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_masks'].to(device)
        labels = batch['labels'].to(device)
        labels = torch.argmax(labels, dim=1) 

        outputs = model(input_ids, attention_masks)
        
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader)}")
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_masks'].to(device)
            labels = batch['labels'].to(device)
            labels = torch.argmax(labels, dim=1)

            outputs = model(input_ids, attention_masks)
            vloss = criterion(outputs, labels)     
    print(f"Epoch {epoch + 1}, Validation Loss:", vloss)
    
    if epoch == 0:
        best_vloss = vloss
        
    if vloss < best_vloss:
        best_vloss = vloss
        model_path = './model_{}'.format(epoch)
        torch.save(model.state_dict(), model_path)


# Validation and prediction
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_masks'].to(device)

        outputs = model(input_ids, attention_masks)
        predictions.append(outputs.cpu().numpy())

pred_result = np.concatenate(predictions, axis=0)
result_bert = ohe.inverse_transform(pred_result)
test_df['bert'] = result_bert

sample_df = pd.read_csv("./sampleSubmission.csv")

id_list   = test_df['id'].tolist()
pred_list_bert = test_df['bert'].tolist()

sample_rb = []

for i in range(len(sample_df['id'])):
    pred = []
    
    id = sample_df['id'][i]
    print (i, id)
    
    index             = id_list.index(id)
    print (index)
    pred_emotion_bert = pred_list_bert[index]
    
    sample_rb.append(pred_emotion_bert)
    
sample_df['bert'] = sample_rb

sample_df.to_csv("pred_nltk_BOW_bert_5.csv")