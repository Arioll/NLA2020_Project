import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        self.pe = pe.unsqueeze(0)
        pe.requires_grad = False
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * np.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:,:seq_len]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        
        assert d_model % heads == 0
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    def attention(self, q, k, v, d_k, mask=None, dropout=None):
    
        scores = torch.matmul(q, k.transpose(-2, -1)) /  np.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(3)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
    
        if dropout is not None:
            scores = dropout(scores)
        
        output = torch.matmul(scores, v)
        return output
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        sl = q.size(1)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, sl, self.h, self.d_k)
        q = self.q_linear(q).view(bs, sl, self.h, self.d_k)
        v = self.v_linear(v).view(bs, sl, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x = x + self.dropout(self.attn(x, x, x, mask))
        x = self.norm_1(x)
        x = x + self.dropout(self.ff(x))
        x = self.norm_2(x)
        return x

class TransformerEmbedding(nn.Module):
    
    def __init__(self, vocab_size, sequence_len, d_model, heads, layers, dropout=0.1):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, sequence_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, dropout) for _ in range(layers)])
        self.norm = Norm(d_model)
        self.padding_id = 0
        self.embedding_dim = d_model
        
    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.pe(x)
        for l in self.layers:
            x = l(x, mask)
        return self.norm(x)