import torch
import torch.nn as nn

from linformer_pytorch.linformer_pytorch.linformer_pytorch import LinformerLM
from performer_pytorch.performer_pytorch.performer_pytorch import PerformerLM
from reformer_pytorch.reformer_pytorch.reformer_pytorch import ReformerLM
from transformer_pytorch.transformer_pytorch import TransformerEmbedding

class GenearalTransformer(nn.Module):

    def __init__(self, model, vocab_size, sequence_len, d_model, heads, layers):
        super().__init__()

        if model not in ['transformer', 'linformer', 'performer', 'reformer']:
            raise Exception(f'Unknown model type {model}. Model type must be one of transformer, linformer, performer, reformer')

        self.model_type = model

        if model == 'transformer':
            self.encoder = TransformerEmbedding(vocab_size, sequence_len, d_model, heads, layers)
        elif model == 'linformer':
            self.encoder = LinformerLM(num_tokens=vocab_size, input_size=sequence_len, channels=d_model, 
                                       nhead=heads, depth=layers)
        elif model == 'performer':
            self.encoder = PerformerLM(num_tokens=vocab_size, max_seq_len=sequence_len, dim=d_model, 
                                       heads=heads, depth=layers)
        elif model == 'reformer':
            self.encoder = ReformerLM(num_tokens=vocab_size, max_seq_len=sequence_len, dim=d_model, 
                                       heads=heads, depth=layers)
        else:
            pass

    def forward(self, X, mask):
        if self.model_type == 'transformer':
            return self.encoder(X, mask)
        elif self.model_type == 'performer':
            return self.encoder(X, return_encodings=True)
        elif self.model_type == 'linformer':
            return self.encoder(X, input_mask=mask, embedding_mask=mask)
        else:
            return self.encoder(X, input_mask=mask, context_mask=mask)