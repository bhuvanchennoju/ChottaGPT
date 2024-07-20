"""
Authored by: Bhuvan Chennoju
Created on: 19th July 2024

kudos to @andrejkarpathy for the original code in his repo:
https://github.com/karpathy/nanoGPT/tree/master/data/openwebtext

in this code I have modified the original code to use huggingface datasets library for smaller datasets of openwebtext.
"""

import os
import numpy as np
from datasets import load_dataset  
from tqdm import tqdm
import tiktoken
import argparse
import sys
import json
import logging



def download_data(dataset_name, num_proc=1):
    if dataset_name == "openwebtext":
        dataset = load_dataset( "openwebtext", num_proc=num_proc)
    elif dataset_name == "openwebtext-10k":
        dataset = load_dataset("stas/openwebtext-10k", num_proc=num_proc)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} error, valid options are ['openwebtext', 'openwebtext-10k']")
    return dataset

def get_encoder(encoder_name):
    # in the original implementation gpt-2 encder is used and it is hardcoded, 
    # but I have made it dynamic so that we can use any encoder that could be replaced gpu-2 

    # "gpt2": gpt2,
    # "r50k_base": r50k_base,
    # "p50k_base": p50k_base,
    # "p50k_edit": p50k_edit,
    # "cl100k_base": cl100k_base,
    # "o200k_base": o200k_base,
    try:
        enc = tiktoken.get_encoding(encoder_name)
    except:
        raise NotImplementedError(f"Encoder {encoder_name} not found")
    return enc


def split_dataset(dataset, test_size=0.0005, seed=2024):
    # in hugging face if the data is not given in train,val, test format all the data would be stored in train
    split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') 
    return split_dataset


def tokenize_dataset(split_dataset, enc, num_proc=1):

    def process(example):
        ids = enc.encode_ordinary(example['text']) 
        ids.append(enc.eot_token) 
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    return tokenized

def write_to_bin(tokenized, dtype=np.uint16, total_batches=1024,save_dir = None):

    ## if the numpy version is greater than 1.x this will throw an error, so ensure to downgrade the numpy version to 1.x
    # I had an error with numpy version 2.x , so I downgraded to 1.x. apparently this is issue with datasets library, hard to debug lol.
    for split,dset in tokenized.items():
        logging.info(f"Writing {split} dataset to bin file")
        arr_len = np.sum(tokenized[split]['len'], dtype=np.uint64)
    
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.join(save_dir, f'{split}.bin')
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,)) # creating the memory array to store the data : this step ensures that we are not loading the entire data into memory
        idx = 0
        dataset_size = len(dset)
        if total_batches > dataset_size:
            total_batches = dataset_size # this is through me under the bus firsttime, so just making sure total_batches in limit
        
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            try:
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                batch_ids = batch['ids']
                arr_batch = np.concatenate(batch_ids)
                arr[idx: idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            except IndexError as e:
                logging.error(f"error {e}")
                break 

        arr.flush()
        del arr
        


def display_samples(tokenized_dataset,enc,num_samples = 2):
    for split in ['train','val']:
        logging.info(f"SAMPLES FROM {split} DATASET")
        logging.info(f"Displaying {num_samples} samples from {split} dataset")
        for i in range(num_samples):
            token_ids = tokenized_dataset[split][i]['ids']
            decode_text = enc.decode(token_ids) # this includes the eot token I just want to see if the text is decoded correctly with the eot token
            logging.info(f"Sample {i+1}: {decode_text[:100]} {decode_text[-100:]} \n")
       
        
def main(args):
    logging.info(f"Downloading {args.dataset} dataset")
    dataset = download_data(args.dataset, args.num_proc)

    logging.info(f"Tokenizing {args.dataset} dataset")
    enc = get_encoder(args.encoder)

    logging.info(f"Splitting {args.dataset} dataset")
    split_dataset_ = split_dataset(dataset, args.test_size, args.seed)

    logging.info(f"Tokenizing {args.dataset} dataset")
    tokenized_dataset = tokenize_dataset(split_dataset_, enc, args.num_proc)
    if args.samples:
        logging.info(f"Displaying {args.samples} samples from the dataset")
        logging.info("********************************************")
        display_samples(tokenized_dataset,enc,args.samples)

    logging.info(f"Writing {args.dataset} dataset to bin files")
    write_to_bin(tokenized_dataset, save_dir=args.save_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess and tokenize datasets")
    parser.add_argument("--dataset", type=str, default="openwebtext-10k", help="dataset name")
    parser.add_argument("--encoder", type=str, default="gpt2", help="encoder name")
    parser.add_argument("--num_proc", type=int, default=1, help="number of processes")
    parser.add_argument("--test_size", type=float, default=0.0005, help="test size")
    parser.add_argument("--seed", type=int, default=2024, help="seed")
    parser.add_argument("--samples", type=int, default=2, help="number of samples to display")
    parser.add_argument("--save_dir", type=str, default=None, help="directory to save the bin files")
    parser.add_argument("--verbose", action="store_true", help="verbose mode")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    main(args)



