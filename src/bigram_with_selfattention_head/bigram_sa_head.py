"""
Authored by: Bhuvan Chennoju
Created on: 23st July 2024

Kudos to:
    - Karpathy's: https://github.com/karpathy/build-nanogpt?tab=readme-ov-file
    - Video: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=784s



The bigram model with self attention head is a simple language model that predicts the next character based on the previous character.




"""

import os
import sys
from pathlib import Path
import numpy as np
import logging
import matplotlib.pyplot as plt
import torch


sys.path.append(str(Path.cwd()))
from src.bigram.data import get_data
from src.bigram.tokenizer import simpleTokenizer
from src.bigram.dataloader import train_valid_split, Batcher
from src.bigram.train import train
from src.bigram_with_selfattention_head.model import BigramLanguageModel


seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)

######################################### CONFIG  #########################################
WORK_dir = ''
DATA_dir =  os.path.join(WORK_dir, 'data')
SRC_dir = os.path.join(WORK_dir, 'src')
input_file_path = os.path.join(DATA_dir,'shakespeare', 'input.txt')
logs_dir = os.path.join(WORK_dir,'logs')
fig_dir = os.path.join(WORK_dir,'assets','images')


exp_name = 'bigram_with_sa_head_loss'

# hyperparameters
split_ratio = [0.8,0.2,0.0]
block_size = 8 # maximum length of the sequence for prediction
batch_size = 64 # batch size for the model
max_iters = 5000 
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
max_tokens = 200


##########################################################################################



# logging setting 
logging.basicConfig(filename = os.path.join(logs_dir,f'{exp_name}.log'), level = logging.INFO)

# get the data
text = get_data(input_file_path)
logging.info(f'text data: {text[:100]}')

# create the tokenizer
tokenizer = simpleTokenizer(text)
vocab_size = tokenizer.get_vocab_size()

# encode the text data
encoded_text = tokenizer.encode(text)
logging.info(f'vocab size: {vocab_size}')
logging.info(f'encoded text: {encoded_text[:10]}')

# train, valid, test splits
data = torch.tensor(encoded_text)
data_splitter = train_valid_split(data,split_ratio = split_ratio)
train_data = data_splitter.train_data
valid_data = data_splitter.valid_data
logging.info(f'train data: {train_data.shape}')
logging.info(f'valid data: {valid_data.shape}')

# batcher
data = {'train':train_data,'valid':valid_data}
data_batcher = Batcher(data,block_size,batch_size)

# model and optimizer
model = BigramLanguageModel(vocab_size,n_embed,block_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# train
loss_track = train(model,optimizer,max_iters,data_batcher,eval_iters)

# loss to disk
np.save(os.path.join(logs_dir,f'{exp_name}.npy'),loss_track)


# generate the new text
context = torch.zeros((1,1),dtype = torch.long,device = device)
generate_idx = model.generate(context, max_new_tokens = max_tokens)
generate_text = tokenizer.decode(generate_idx[0].cpu().numpy())
logging.info(f'generated text: {generate_text}')



# plotting the train,val loss
loss_track = np.load(os.path.join(logs_dir,f'{exp_name}.npy'),allow_pickle = True).item()
plt.plot(loss_track['train'],label = 'train_loss')
plt.plot(loss_track['valid'],label = 'valid_loss')
plt.legend()
plt.savefig(os.path.join(fig_dir,f'{exp_name}.png'))



