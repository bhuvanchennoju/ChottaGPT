
## ChottaLLM 


###  Kerneal optimization with tricks
1) torch.compile: This perform the kernel fusion, and optimiezes kernal by reducging the python overhead. If we dont have torch.complie we would do the operations multiple times on the chip and save them to gpu memory (HBM). With torch.complie all these overheads (like calculationg an activations like GELU) would be perfomed only one time when the tensors on the gpu chip, and stores everying to memory(HBM) in one time. This is called kernel fusion. 
   
   ```
   Speedup mainly comes from reducing Python overhead and GPU read/writes, and so the observed speedup may vary on factors such as model architecture and batch size. For example, if a model’s architecture is simple and the amount of data is large, then the bottleneck would be GPU compute and the observed speedup may be less significant.
   
   ```
   Source: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
Torch complie is amazing in optimizing kernels but there are certain operations that torch.complie will not find, and one such example is flash attention. 

2) Flash Attention: This is kernal fusion algorithm, that fuses critical attention steps including matmul, dropout,softmax,mask,matmul into one fused kernal called flashattention. The reason it cannot find by the torch.compile is because it requires algorithmic rewrite. Flash attention is 7.6 x faster than traditionl attention because it memory concious, yet it have higher FLOP than traditional algorithm. Under the hood, instead of calcualating a big chunck of matrix calucation in (1), in the paper( online normalizer calcuation for softmax) they rewrote the way to calculate softmax in incremental tiling fasion, this make the physical existentce of (1) become redundent, and step will be done on fly. 
   ```python 
   # before 
   att = (q @ k.transpose(-2.-1)) * (1.0 / math.sqrt(k.size(-1))) #--->(1) 
   att = att.masked_fill(self.bias[:,:,:T,:T] == 0 , float('-inf'))
   att = F.softmax(att, dim = -1)
   y = att @ v # dims (B,nh, T, T) x (B, nh, T,hs) --> (B, nh, T, hs)

   # after
   y = F.scaled_dot_product_attention(q,k,v,is_causal = True)
   
   ```
   Source: https://github.com/Dao-AILab/flash-attention

3) Use powers of 2: Just a hacky way to optimize. In the cuda so many kernals use block tailes, and these usual in chunks of power 2, when the desired calculation does not fit in these blocks, all the operations will be done in two or three phases. This tasks more time, and by changing the numbers to nice powers of 2, we are removing the boundary chunks that need the second phase of calcualtion, and this optimize the run time. 

### Algorithmic Optimization: hyper parameters
1) Looking into GPT-3 paper to optimize the GPT-2 hyperparameters:
   * AdamW optimizer with beta1 = 0.9, and beta2 = 0.95, eps =10^-8
   * Cliping the gradients to 1 after backward. The reason behind this implemntation is sometimes we might get unlucky in batch, we end up with high loss, cause the high gradient and that would shock the model. This will help model to be more resilient to such unexpected shocks, and this help in stability of model.

  ```python

  norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

  ```
    * Cosine decay learning rate instead of fixed learning reate. and there is a linear warmup as well.
  ```python

  # in training loop
  for step in range(max_steps):
    # code 
    # code
    # backward, and norm
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  
    optimizer.step()

   max_lr = 3e-4
   min_lr = max_lr * 0.1
   warmup_steps = 10
   max_steps = 50
   def get_lr(step):
    # 1) linear warmup for warmup_iters steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # 2) if step > lr_decay_iters, return min learning rate
    if step > max_steps:
        return min_lr

    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0<= decay_ratio <= 1
    coerr = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
  ```
  * linear increase of the batch size. 
  * Data sampling without replacement during training to minimize overfitting
  * implement weightdecay of 0.1 to provide a min of regularization. In the implementation code, configure_optimizers, first all the parameters are splitted into parameters that needed to weight decayed, and parameters that are not supposed to weight decayed. Any one dim tensors (layer norm, scales, bias) are not supposed to weight decayed. Fused in adamw - faster in cuda. This get rid of lot of overhead in the AdamW and fuse all the kernals instead of for loops inside.
* 