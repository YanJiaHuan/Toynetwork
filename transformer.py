### lib ###
import torch
import torch.nn as nn

### custom ###

class SelfAttention(nn.Module):
    def __init__(self,embed_size, heads):
        super(SelfAttention,self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "embed_size 要是 heads 的整數倍"
        # 不满足括号后的条件，就会报错
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # nn.Linear(in_features, out_features, bias=True/False)
        # out_feature = weight @ in_feature + bias, @ 指矩阵乘法
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self,values,keys,queries,mask):
        N = queries.shape[0] # N: batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        # Split embedding into self.heads pieces
        values = values.reshape(N,value_len,self.heads,self.head_dim)
        keys = keys.reshape(N,key_len,self.heads,self.head_dim)
        queries = queries.reshape(N,query_len,self.heads,self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk",[queries,keys])
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)
        if mask is not None:
            energy = energy.masked_fill(mask==0,float("-1e20"))
        # 实际上是用一个很小的值去填充成一个三角矩阵，最开始是全mask，到最后是只mask最后一个词









