"""
Authored by: Bhuvan Chennoju
Created on: 21st July 2024

Kudos to:
    - Karpathy's: https://github.com/karpathy/build-nanogpt?tab=readme-ov-file
    - Video: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=784s

character level mapping based tokenizer to encode decode the text data 
based on the character level mapping.

"""


class simpleTokenizer:

    '''
    Simple tokenizer that encode decode character level tokenization.
    Alternatives could be google sentencepiece, huggingface tokenizers, openai Tiktoken library.
    
    '''

    def __init__(self,text):
        self.unique_chars = sorted(list(set(text)))
        self.vocab_size = len(self.unique_chars)
        # strign to index mapping
        self.str_to_idx = {chr: idx for idx,chr in enumerate(self.unique_chars)}
        # index to string mapping
        self.idx_to_str = {idx: chr for idx,chr in enumerate(self.unique_chars)}

        # adding padding and unknown token
        self.str_to_idx['<PAD>'] = self.vocab_size
        self.idx_to_str[self.vocab_size] = '<PAD>'
        self.unique_chars.append('<PAD>')

        self.str_to_idx['<UNK>'] = self.vocab_size + 1
        self.idx_to_str[self.vocab_size + 1] = '<UNK>'
        self.unique_chars.append('<UNK>')

        self.vocab_size += 2

    def encode(self, text):
        '''
        this function takes a string and look up in the string to index mapping, and return the list of indices.
        '''
        return [self.str_to_idx[chr] if chr in self.unique_chars else self.str_to_idx['<UNK>'] for chr in text]
        
        
    def decode(self, indices):
        '''
        this function takes a list of indices and look up in the index to string mapping, and return the string.
        '''
        return ''.join([self.idx_to_str[idx] for idx in indices]) 
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_unique_chars(self):
        return self.unique_chars