# Toynetwork
This is a repo for building NNs from scratch.
## RNN
### Advantages
- Can handle variable length sequences.
- Weights are shared across time steps.
- Can take historical information into account.
- Model size won't increase with the change of size of input.
### Disadvantages
- Computation is slow.
- Can't do with long-term dependencies.
- Can't predict the future.

## Transformer
![img.png](./img/attention.png)
### Encoder part
Step1: input x1 into 3 different parts: query, key, value, output y1.  
Step2: res(y1 and x1 )into a normalization layer, get y2.  
Step3: y2 into a feed forward network, get y3.  
Step4: res(y2 and y3) into a normalization layer, get y4, this y4 is the output of a encoder layer.

Transformer architecture is permutation invariant, which means the order of input doesn't matter, which is the
reason why they add a positional encoding to the input to make sure the order does matter.

<img src="http://chart.googleapis.com/chart?cht=tx&chl= Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})" style="border:none;">  

### Decoder part
The difference in decoder transformer block is the value and key are from the encoder part, and the query is from the previous layer of decoder part. 

The first attention in encoder part is masked-attention.