"""
Authored by: Bhuvan Chennoju
Created on: 21st July 2024

Kudos to:
    - Karpathy's: https://github.com/karpathy/build-nanogpt?tab=readme-ov-file
    - Video: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=784s

This file contains the bigram model implementation for the character level language model.
I have followed the same approach as in the video and the code is inspired from the above mentioned sources, and 
added few modifications to make it work for the character level language model.

The bigram model is a simple language model that predicts the next character based on the previous character.
The model is trained on the text data and the probabilities of the next character are calculated based on the
frequency of the characters in the text data.




"""

import os
import numpy as np
import sys
import torch

from data import get_data
from tokenizer import simpleTokenizer
from dataloader import train_valid_split, Batcher
from model import  BigramLanguageModel
from train import train

# setting the seed for reproducibility
seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)

# hyperparameters
block_size = 8 # maximum length of the sequence for prediction
batch_size = 64 # batch size for the model
max_iters = 1000 
lr = 0.001 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# directories
WORK_dir = ''
DATA_dir =  os.path.join(WORK_dir, 'data')
SRC_dir = os.path.join(WORK_dir, 'src')
input_file_path = os.path.join(DATA_dir,'shakespeare', 'input.txt')

# get the data
text = get_data(input_file_path)

# create the tokenizer
tokenizer = simpleTokenizer(text)
vocab_size = tokenizer.get_vocab_size()

# encode the text data
encoded_text = tokenizer.encode(text)
print(f'vocab size: {vocab_size}')  
print(f'encoded text: {encoded_text[:10]}')

# create the train, valid, test splits
data = torch.tensor(encoded_text)
data_splitter = train_valid_split(data,split_ratio = [0.8,0.2,0.0])
train_data = data_splitter.train_data
valid_data = data_splitter.valid_data

# create the batcher
data = {'train':train_data,'valid':valid_data}
data_batcher = Batcher(data,block_size,batch_size)

# create the model
model =  BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# training loop
loss_track = train(model,optimizer,max_iters,data_batcher,eval_iters)

# generate the new text
context = torch.zeros((1,1),dtype = torch.long,device = device)
generate_idx = model.generate(context, max_new_tokens = 100)
generate_text = tokenizer.decode(generate_idx[0].cpu().numpy())
print(generate_text)


