"""
Authored by: Bhuvan Chennoju
Created on: 21st July 2024

Kudos to:
    - Karpathy's: https://github.com/karpathy/build-nanogpt?tab=readme-ov-file
    - Video: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=784s

I have created a simple dataloader to create the train,valid splits,
and batcher to generate the batches of data for the bigram model.

"""
import numpy as np
import random 
from torch import tensor,stack 

class train_valid_split:
    '''
    train valid test spliter for the data.
    args:
    data: list of data
    split_ratio: list of split ratios [train,valid,test]
    return:
    train, valid, test data
    '''
    def __init__(self,data,split_ratio = [0.8,0.2,0.0]):
        self.data = data
        self.split_ratio = split_ratio
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self._split_data()
    
    def _split_data(self):
        train_len = int(len(self.data)*self.split_ratio[0])
        valid_len = int(len(self.data)*self.split_ratio[1])
        self.train_data = self.data[:train_len]
        self.valid_data = self.data[train_len:train_len+valid_len]
        self.test_data = self.data[train_len+valid_len:]
        return self.train_data, self.valid_data, self.test_data


# def get_batch(split):
#     data  = train_data if split == 'train' else val_data
#     start_idx = torch.randint(0, data.size(0) - block_size, (batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in start_idx]) # (batch_size, block_size) expected shape and expected in tuple of tensors or list of tensors
#     y = torch.stack([data[i+1:i+block_size+1] for i in start_idx])
#     return x,y
    
class Batcher:
    '''
    create the batches of x,y data for the model;
    its an iterator that returns the x,y batch data on each iteration.
    args:
    data: dict of data{train,valid,test}
    block_size: block size of the data
    batch_size: batch size of the data
    return:
    x,y batch data
    

    '''
    def __init__(self,data,block_size,batch_size):
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
    
    def get_batch(self,split):
        data = self.data[split]
        start_idx = random.sample(range(len(data) - self.block_size), self.batch_size)
        x = stack([tensor(data[i:i+self.block_size]) for i in start_idx])
        y = stack([tensor(data[i+1:i+self.block_size+1]) for i in start_idx])
        return x,y
     

    

    # def __iter__(self):
    #     self.current_idx = 0
    #     random.shuffle(self.indices)
    #     return self
    
    # def __next__(self):
    #     if self.current_idx >= len(self.indices):
    #         raise StopIteration
    #     batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
    #     self.current_idx += self.batch_size
    #     x = stack([tensor(self.data[i:i+self.block_size]) for i in batch_indices])
    #     y = stack([tensor(self.data[i+1:i+self.block_size+1]) for i in batch_indices])
    #     return x,y
    


