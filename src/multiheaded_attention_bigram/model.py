"""
Authored by: Bhuvan Chennoju
Created on: 21st July 2024

Kudos to:
    - Karpathy's: https://github.com/karpathy/build-nanogpt?tab=readme-ov-file
    - Video: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=784s

multiheaded attention model implementation for the character level language model.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):

    def __init__(self, n_embed,head_size,block_size):
        super().__init__()
        self.n_embed = n_embed
        self.head_size = head_size
        self.key = nn.Linear(n_embed,head_size,bias = False)
        self.query = nn.Linear(n_embed,head_size,bias = False)
        self.value = nn.Linear(n_embed,head_size,bias = False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C = x.shape
        # every token gives a key, query
        # query -- what I am interested in
        # key --  here is what i have
        # value --- if you find me interesting, here is what i will communicate to you.
        k = self.key(x) # (B,T,emb_size) --> (B,T,head_size)
        q = self.query(x) # (B,T,emb_size) --> (B,T,head_size)

        # computing attention scores
        wei = q @ k.transpose(-2,-1) * C**(-0.5) # (B,T,head_size) @ (B,head_size,T) --> (B,T,T) 
        wei = wei.masked_fill(self.tril[:T,:T] ==0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei,dim = -1)  #(B,T,T)
        v = self.value(x) #(B,T,head_size)
        out = wei @ v #(B,T,T) @ (B,T,head_size) --> (B,T,C)
        return out
    

class MultiHeadAttention(nn.Module):

    def __init__(self,nheads,head_size,n_embed,block_size):
        super().__init__()
        self.nheads = nheads
        self.head_size = head_size
        self.heads = nn.ModuleList([Head(n_embed,head_size,block_size) for _ in range(nheads)])
    
    def forward(self,x):
        return torch.cat([h(x) for h in self.heads],dim = -1) # (B,T,C) --> (B,T,C*nheads)
    
# feed forwad network
    
class FFN(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,n_embed*4),
            nn.ReLU(),
            nn.Linear(n_embed*4,n_embed)
        )

    def forward(self,x):
        return self.net(x)
    
# block layer
class Block(nn.Module):
    def __init__(self,n_embed,head_size,block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mha = MultiHeadAttention(4,n_embed//4,n_embed,block_size)
        self.ffn = FFN(n_embed)
    
    def forward(self,x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    



class BigramLanguageModel(nn.Module):
    
    def __init__(self,vocab_size,n_embed,block_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size
        # self.head_size = head_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # --> (vocab_size, n_embed) this is more like an intermediate representation of the token
        self.position_embedding_table = nn.Embedding(block_size,n_embed)

        # # self.sa_head = Head(n_embed,n_embed,block_size)
        # self.sa_head = MultiHeadAttention(4,n_embed//4,n_embed,block_size) # --> 4 heads with n_embed//4 size each head so let say emb_size = 32, then each head will have 8 size
        # # this is like a grouped convolutions, instead of one big attention head, ew just split the head into 4 heads and each head will have 8 size.
        # self.lm_head = nn.Linear(n_embed,vocab_size)

        self.sa_head = Block(n_embed,n_embed,block_size)
        self.lm_head = nn.Linear(n_embed,vocab_size)





    def forward(self,idx,targets = None):

        # logits = self.token_embedding_table(idx) # --> (B,T,C) - (batch_size, block_size, vocab_size) 
        tok_emb = self.token_embedding_table(idx) # --> (B,T,C) - (batch_size, block_size, n_embed)
        pos_emb = self.position_embedding_table(torch.arange(idx.shape[1],device = idx.device)) # --> (T,C) - (block_size, n_embed)
        x  = tok_emb + pos_emb # --> (B,T,C) - (batch_size, block_size, n_embed)
        x = self.sa_head(x) # --> (B,T,C) - (batch_size, block_size, n_embed)
        logits = self.lm_head(x) # --> (B,T,C) - (batch_size, block_size, vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape  # (batch_size, block_size) expected target shape but this will run into error because of pytorch expectation of the
                                  # multidimentional is (B*T,C) not (B,T,C) # so converting logits --> (B*T,C) and targets --> (B*T,) 
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
            

            # changes to reflect the new model
            
            # crop idx to the last block_size tokens
            idx_crop = idx[:,-self.block_size:] # (B,T) --> (B,block_size) we cant have more than block_size tokens in the context, because our position embedding is limited to block_size and run out of scope if we have more than block_size tokens.

            logits, _ = self.forward(idx_crop)

            # taking only last layer logits
            logits = logits[:,-1,:] # (B,C)
            # applying softmax to get the probabilities
            probs = F.softmax(logits, dim = -1) # (B,C) ---> in this bigram model, we care only for the last token but the attention models will take all the previous tokens into account 
            # sampling from the probabilities
            idx_next = torch.multinomial(probs, num_samples = 1) # (B,1) # refer (2) below
            # append the sampled index to the running sequence
            idx = torch.cat([idx,idx_next],dim = 1)
        return idx
