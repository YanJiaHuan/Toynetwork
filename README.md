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
The first attention in encoder part is masked-attention, need to matmul to a triangular zero matrix.  

Step1: input x1 into 3 different parts: query, key, value, output y1.  
Step2: res(y1 and x1 )into a normalization layer, get y2.  
Step3: attention(encoder_output_key, encoder_output_value, y2) get y3.  
Step4: res(y2 and y3) into a normalization layer, get y4.  
Step5: y4 into a feed forward network, get y5.  
Step6: res(y4 and y5) into a normalization layer, get y6, this y6 is the output of a decoder layer.  

### Positional encoding
Nothing but y = x + pos_encoidng(seq_length), only to embedd the position information into the input.

### Num_layers
The real implemtation of Transformers may need to build Transformer_block and Decoder_block first, beacuse the num_layers
determine how many blocks needed to build a Transformer.  
The code is super easy, just call a layer fucntion and call the blocks in a loop.

### Num_heads
In transformer, embed_size = head_size * num_heads, and the matrix of query, key, value are consist of mutiple [head_size, head_size] matrix.  
With the **increase** of num_heads, the computation loop **increases**, which made the total computation cost **increases**. But the
good part is **more** iteration of computation can be introduced, which means **more** aspects of attention is considered.  
From my perspective, **increase** num_heads --> **increase** computation cost --> **decrease** attention block --> make model more focus on details.

#### For example:
Bert-base has 12 layers, 12 heads, 768 embed_size.  
Bert-large has 24 layers, 16 heads, 1024 embed_size.  
Bert-large: 340M parameters, Bert-base: 110M parameters.  
x = Bert-base.para, y = Bert-large.para
y = (24/12) * (1024/768) * x = 2.67x   
110M * 2.67 = 293.7M, which is close to 340M, normally there will be some other parameters inside.
