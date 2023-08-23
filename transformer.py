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
        attention = torch.softmax(energy/(self.embed_size**(1/2)),dim=3)
        # softmax(dim=3) 指对第三维度进行softmax-->all elements in (0,1) and sum to 1
        # 为什么是对第三维度进行softmax？因为这个纬度的数量=单词(element)的数量，我的注意力要分配的是每个单词的权重，所以我要对每个单词进行softmax,因为这个energy tensor，dim0
        # 是batch size，dim1是head，dim2是query_len，dim3是key_len, 实际上query_len = key_len
        out = torch.einsum("nhql,nlhd->nqhd",[attention,values])
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out shape: (N, query_len, heads, heads_dim)
        out = out.reshape(N,query_len,self.heads*self.head_dim) # concat heads
        out = self.fc_out(out) # map the embed_size to embed_size
        return out

class TransformerBlock(nn.Module):
    def __int__(self,embed_size,heads,dropout,forward_expansion):
        super(TransformerBlock,self).__init__()
        self.attention = SelfAttention(embed_size,heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # 假如只定义一个norm1会怎么样？技术上可以，但由于nomalization layer 里也有一点可学习的参数，但两处地方，显然这个参数是需要不同的

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size,embed_size)
        )
        # 顶级💩










