"""
Authored by: Bhuvan Chennoju
Created on: 19th July 2024

This file downloads the data for this project, and save it to disk in binary format.
based on karpthy's implementation of nanoGPTs I am writing this code to download, and process the openwebtext, in full, or sample 10k datasets.

https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
https://huggingface.co/datasets/stas/openwebtext-10k



"""

import os
import numpy as np
from datasets import load_dataset  

def download_openwebtext_10k():
    pass
def download_openwebtext_full():
    pass

def download_data():
    pass

def process_data():
    pass

def main():
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="openwebtext-10k", help="Dataset to download and process")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save the data")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the existing data")
    args = parser.parse_args()

    main(args)
