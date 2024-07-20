## Openwebtext  

Breif about the dataset: OpenWebText is an open-source recreation of the WebText corpus. The text is web content extracted from URLs shared on Reddit with at least three upvotes. (38GB).

This file downloads the data for this project, and save it to disk in binary format.
based on karpthy's implementation of nanoGPTs I am writing this code to download, and process the openwebtext, in full, or sample 10k datasets.

https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
https://huggingface.co/datasets/stas/openwebtext-10k



For this dataset preparation I have used the orginal code but moduluraized a bit for the flexibility with the tokenization, and I am using cl100k_base to endode this data. 

```
OpenAI provide a Python library for doing this called tiktoken.

If you dig around inside the library you’ll find it currently includes five different tokenization schemes: r50k_base, p50k_base, p50k_edit, cl100k_base and gpt2.

Of these cl100k_base is the most relevant, being the tokenizer for both GPT-4 and the inexpensive gpt-3.5-turbo model used by current ChatGPT.

Source: 
https://simonwillison.net/2023/Jun/8/gpt-tokenizers/

```

Note: if the you are using hugging faces datasets version 2.20.0, and numpy > 1.xx.x , this will cause a numpy array duplication error when stroring the tokeneized data into bytes. To ensure the data preparation just downgrade the numpy to 1.x