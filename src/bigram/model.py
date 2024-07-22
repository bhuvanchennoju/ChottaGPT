"""
Authored by: Bhuvan Chennoju
Created on: 21st July 2024

Kudos to:
    - Karpathy's: https://github.com/karpathy/build-nanogpt?tab=readme-ov-file
    - Video: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=784s

a simple bigram model to predict the next character based on
last character in the text data. No hidden layers, no activation functions, just a simple bigram model.


"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self,idx,targets = None):

        logits = self.token_embedding_table(idx) # --> (B,T,C) - (batch_size, block_size, vocab_size) 
        # loss = F.cross_entropy(logits, targets) # --> (batch_size, block_size) expected target shape but this will run into error because of pytorch expectation of the
        #                                         # multidimentional is (B*T,C) not (B,T,C)
        # so converting logits --> (B*T,C) and targets --> (B*T,) 
        # refer: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self,idx, max_new_tokens = 100):
        '''
        function to generate new tokens based on the given context
        idx : tensor of shape (B,T) - batch_size, block_size
        returns: tensor of shape (B,T) - batch_size, block_size
        '''

        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx)
            # taking only last layer logits
            logits = logits[:,-1,:] # (B,C)
            # applying softmax to get the probabilities
            probs = F.softmax(logits, dim = -1) # (B,C) ---> in this bigram model, we care only for the last token but the attention models will take all the previous tokens into account 
            # sampling from the probabilities
            idx_next = torch.multinomial(probs, num_samples = 1) # (B,1) # refer (2) below
            # append the sampled index to the running sequence
            idx = torch.cat([idx,idx_next],dim = 1)
        return idx
