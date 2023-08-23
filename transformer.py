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
        self.dropout = nn.Dropout(dropout)

    def forward(self,value,key,query,mask):
        attention = self.attention(value,key,query,mask)
        x = self.norm1(attention + query) # residual connection of attention and query, with a normalization layer
        x = self.dropout(x)
        x2 = self.feed_forward(x) # x after feed forward block
        out = self.norm2(x2 + x) # residual connection of x and x2, with a normalization layer
        out = self.dropout(out)
        return out

class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,# 词典大小
            embed_size, # 词向量维度 相当于用多大的向量去表示每个单词
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
    ):
        super(Encoder,self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size,embed_size)
        # nn.embedding 是一个index->vector的映射，这里的index是单词的index，vector是词向量
        self.position_embedding = nn.Embedding(max_length,embed_size)
        # 这个应该很合理，就是让模型学会 第一，第二...第max_length个单词的位置所代表的意义
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        N,seq_length = x.shape
        potisions = torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
        # what is .expand here for? torch.arange(0,N) 只是一个一维的向量，输入会有个N的batch纬度
        out = self.dropout(self.word_embedding(x) + self.position_embedding(potisions)) # word embedding + position embedding(redidual) and dropout
        for layer in self.layers:
            out = layer(out,out,out,mask)
        return out

class DecoderBlock(nn.Module):
    def __init__(self,embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock,self).__init__()
        self.attention = SelfAttention(embed_size,heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size,heads,dropout,forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        # src_mask: mask of source sentence, to prevent attention to padding token
        # trg_mask: mask of target sentence, to prevent decoder to attend future token (when predict the Nth, can only attend the first N-1)
        attention = self.attention(x,x,x,trg_mask)
        query_out = self.dropout(self.norm(attention + x)) # 那个单独出来的query 后面要跟着encoder 出来的value和key，进正常的transformer block
        out = self.transformer_block(value,key,query_out,src_mask)
        # value, key: encoder output
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
    ):
        super(Decoder,self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size,embed_size)
        self.position_embedding = nn.Embedding(max_length,embed_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size,heads,forward_expansion,dropout,device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size,trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):























