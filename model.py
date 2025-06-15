import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        pe = torch.zeros((max_seq_length, d_model))
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "Vetor de embedding precisa ser divisivel pelo número de cabeças da camada de atenção!"
        self.head_dim = d_model // num_heads
        self.d_model, self.num_heads = d_model, num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, encoder_output=None):
        # Entra Q, K, V com dimensão (batch_size, sequence_length, d_model)
        # Reshape para (batch_size, sequence_length, num_heads, d_model)
        # Reordering para (batch_size, num_heads, sequence_length, d_model)
        if encoder_output is None:
            x = torch.reshape(x, shape=(x.shape[0], x.shape[1], self.num_heads, self.head_dim)) #.contiguous()
            x = x.permute(0, 2, 1, 3)
        else:
            raise NotImplementedError("Modelo ainda não compatível com Encoder.")
        return x

    def compute_attention_scores(self, q_linear_out, k_linear_out, v_linear_out, mask=None):
        qk_dot_product = torch.matmul(q_linear_out, k_linear_out.transpose(2, 3)) / self.head_dim ** 0.5

        if mask is not None:
            qk_dot_product = qk_dot_product.masked_fill(mask == 0, float('-inf'))

        attn_scores = nn.functional.softmax(qk_dot_product, dim=-1)
        attn_weighted_v = torch.matmul(attn_scores, v_linear_out)

        return attn_weighted_v


    def combine_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        return torch.reshape(x, shape=(x.shape[0], x.shape[1], int(x.shape[2] * x.shape[3])))

    def forward(self, x, mask):
        q_linear_out = self.split_heads(self.q(x))
        k_linear_out = self.split_heads(self.k(x))
        v_linear_out = self.split_heads(self.v(x))
        
        attn_weighted_v = self.compute_attention_scores(q_linear_out, k_linear_out, v_linear_out, mask=mask)
        attn_weighted_v = self.combine_heads(attn_weighted_v)
        return self.output_linear(attn_weighted_v)


class FeedForwardSubLayer(nn.Module):
    def __init__(self, d_model, hidden_size):
        super().__init__()
        self.ff_1 = nn.Linear(d_model, hidden_size)
        self.ff_2 = nn.Linear(hidden_size, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.ff_2(self.relu(self.ff_1(x)))
    

class DecoderBlock(nn.Module):
    def __init__(self, d_model, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.feed_forward = FeedForwardSubLayer(d_model, hidden_size)
        self.mha = MultiHeadAttention(d_model, num_heads) # nn.MultiheadAttention()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_mask):
        x = self.norm_1(x + self.dropout(self.mha(x, mask=tgt_mask)))
        x = self.norm_2(x + self.dropout(self.feed_forward(x)))
        return x
    

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_sequence_length, n_layers, hidden_size, num_heads, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model, max_sequence_length)
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, hidden_size, num_heads, dropout) for _ in range(n_layers)]
        )
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, tgt_mask):
        x = self.embedding(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, tgt_mask)
        out = self.output_layer(x)
        return out
