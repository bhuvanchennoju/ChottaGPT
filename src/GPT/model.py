"""
Authored by: Bhuvan Chennoju
Created on: 21st July 2024

Kudos to:
    - Karpathy's: https://github.com/karpathy/build-nanogpt?tab=readme-ov-file
    - Video: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=784s

GPT model implementation for the character level language model.


"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):

    def __init__(self, n_embed,head_size,block_size,dropout = 0.2):
        super().__init__()
        self.n_embed = n_embed
        self.head_size = head_size
        self.key = nn.Linear(n_embed,head_size,bias = False)
        self.query = nn.Linear(n_embed,head_size,bias = False)
        self.value = nn.Linear(n_embed,head_size,bias = False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,emb_size) --> (B,T,head_size)
        q = self.query(x) # (B,T,emb_size) --> (B,T,head_size)

        # computing attention scores
        wei = q @ k.transpose(-2,-1) * C**(-0.5) # (B,T,head_size) @ (B,head_size,T) --> (B,T,T) 
        wei = wei.masked_fill(self.tril[:T,:T] ==0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei,dim = -1)  #(B,T,T)
        wei = self.dropout(wei)
        v = self.value(x) #(B,T,head_size)
        out = wei @ v #(B,T,T) @ (B,T,head_size) --> (B,T,C)
        return out
    

class MultiHeadAttention(nn.Module):

    def __init__(self,nheads,head_size,n_embed,block_size,dropout = 0.2):
        super().__init__()
        self.nheads = nheads
        self.head_size = head_size
        self.heads = nn.ModuleList([Head(n_embed,head_size,block_size,dropout) for _ in range(nheads)])
        self.proj = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim = -1) # (B,T,C) --> (B,T,C*nheads)
        out = self.proj(out) # (B,T,C*nheads) --> (B,T,C)
        return self.dropout(out)
    

class FFN(nn.Module):
    def __init__(self,n_embed,dropout = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,n_embed*4),
            nn.ReLU(),
            nn.Linear(n_embed*4,n_embed),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self,n_embed,block_size,n_heads,dropout = 0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mha = MultiHeadAttention(n_heads,n_embed//n_heads,n_embed,block_size,dropout)
        self.ffn = FFN(n_embed,dropout = 0.2)
    
    def forward(self,x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    


class Transformer(nn.Module):
    
    def __init__(self,vocab_size,n_embed,block_size,n_layers,n_heads,dropout = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # --> (vocab_size, n_embed) this is more like an intermediate representation of the token
        self.position_embedding_table = nn.Embedding(block_size,n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed,block_size,n_heads,dropout) for _ in range(n_layers)])
        self.lm_head = nn.Linear(n_embed,vocab_size)
        self.ln_f = nn.LayerNorm(n_embed) #final layer





    def forward(self,idx,targets = None):

        # logits = self.token_embedding_table(idx) # --> (B,T,C) - (batch_size, block_size, vocab_size) 
        tok_emb = self.token_embedding_table(idx) # --> (B,T,C) - (batch_size, block_size, n_embed)
        pos_emb = self.position_embedding_table(torch.arange(idx.shape[1],device = idx.device)) # --> (T,C) - (block_size, n_embed)
        x  = tok_emb + pos_emb # --> (B,T,C) - (batch_size, block_size, n_embed)
        
        x = self.blocks(x) # --> (B,T,C) - (batch_size, block_size, n_embed)
        x = self.ln_f(x) # --> (B,T,C) - (batch_size, block_size, n_embed)
        
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
