import torch
import torch.nn as nn
import numpy as np

from general_transformer import GenearalTransformer

parameter_dict_linformer = {
    'num_tokens': 10000, # Number of tokens in the LM
    'input_size': 512, # Dimension 1 of the input
    'channels': 64, # Dimension 2 of the input
    'dim_d': None, # Overwrites the inner dim of the attention heads. If None, sticks with the recommended channels // nhead, as in the "Attention is all you need" paper
    'dim_k': 128, # The second dimension of the P_bar matrix from the paper
    'dim_ff': 128, # Dimension in the feed forward network
    'dropout_ff': 0.15, # Dropout for feed forward network
    'nhead': 4, # Number of attention heads
    'depth': 2, # How many times to run the model
    'dropout': 0.1, # How much dropout to apply to P_bar after softmax
    'activation': "gelu", # What activation to use. Currently, only gelu and relu supported, and only on ff network.
    'use_pos_emb': True, # Whether or not to use positional embeddings
    'checkpoint_level': "C0", # What checkpoint level to use. For more information, see below.
    'parameter_sharing': "layerwise", # What level of parameter sharing to use. For more information, see below.
    'k_reduce_by_layer': 0, # Going down `depth`, how much to reduce `dim_k` by, for the `E` and `F` matrices. Will have a minimum value of 1.
    'full_attention': False, # Use full attention instead, for O(n^2) time and space complexity. Included here just for comparison
    'include_ff': True, # Whether or not to include the Feed Forward layer
    'w_o_intermediate_dim': None, # If not None, have 2 w_o matrices, such that instead of `dim*nead,channels`, you have `dim*nhead,w_o_int`, and `w_o_int,channels`
    'emb_dim': 128, # If you want the embedding dimension to be different than the channels for the Linformer
    'causal': False, # If you want this to be a causal Linformer, where the upper right of the P_bar matrix is masked out.
    'method': "learnable", # The method of how to perform the projection. Supported methods are 'convolution', 'learnable', and 'no_params'
    'ff_intermediate': None, # See the section below for more information
}

parameter_dict_performer = {
    'num_tokens': 20000,
    'max_seq_len': 2048,             # max sequence length
    'dim': 512,                      # dimension
    'depth': 12,                     # layers
    'heads': 8,                      # heads
    'causal': False,                 # auto-regressive or not
    'nb_features': 256,              # number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head
    'feature_redraw_interval': 1000, # how frequently to redraw the projection matrix, the more frequent, the slower the training
    'generalized_attention': False,  # defaults to softmax approximation, but can be set to True for generalized attention
    'kernel_fn': nn.ReLU(),          # the kernel function to be used, if generalized attention is turned on, defaults to Relu
    'reversible': True,              # reversible layers, from Reformer paper
    'ff_chunks': 10,                 # chunk feedforward layer, from Reformer paper
    'use_scalenorm': False,          # use scale norm, from 'Transformers without Tears' paper
    'use_rezero': False,             # use rezero, from 'Rezero is all you need' paper
    'tie_embedding': False,          # multiply final embeddings with token weights for logits, like gpt decoder
    'ff_glu': True,                  # use GLU variant for feedforward
    'emb_dropout': 0.1,              # embedding dropout
    'ff_dropout': 0.1,               # feedforward dropout
    'attn_dropout': 0.1,             # post-attn dropout
    'local_attn_heads': 4,           # 4 heads are local attention, 4 others are global performers
    'local_window_size': 256         # window size of local attention
}

parameter_dict_reformer = {
    'num_tokens': 20000,
    'dim': 1024,
    'depth': 12,
    'max_seq_len': 8192,
    'heads': 8,
    'lsh_dropout': 0.1,
    'ff_dropout': 0.1,
    'post_attn_dropout': 0.1,
    'layer_dropout': 0.1,  # layer dropout from 'Reducing Transformer Depth on Demand' paper
    'causal': True,        # auto-regressive or not
    'bucket_size': 64,     # average size of qk per bucket, 64 was recommended in paper
    'n_hashes': 4,         # 4 is permissible per author, 8 is the best but slower
    'mb_dim': 128,        # embedding factorization for further memory savings
    'im_head': 64,        # be able to fix the dimension of each head, making it independent of the embedding dimension and the number of heads
    'f_chunks': 200,      # number of chunks for feedforward layer, make higher if there are memory issues
    'ttn_chunks': 8,      # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
    'um_mem_kv': 128,       # persistent learned memory key values, from all-attention paper
    'win_attention': False, # both branches of the reversible network will be attention
    'ull_attn_thres': 1024, # use full attention if context length is less than set value
    'everse_thres': 1024,   # turn off reversibility for 2x speed for sequence lengths shorter or equal to the designated value
    'se_scale_norm': False,  # use scale norm from 'Transformers without tears' paper
    'se_rezero': False,      # remove normalization and use rezero from 'ReZero is All You Need'
    'ne_value_head': False,  # use one set of values for all heads from 'One Write-Head Is All You Need'
    'eight_tie': False,           # tie parameters of each layer for no memory per additional depth
    'eight_tie_embedding': False, # use token embedding for projection of output, some papers report better results
    '_local_attn_heads': 2,       # many papers suggest mixing local attention heads aids specialization and improves on certain tasks
    'km_layers': (4,7),           # specify layers to use product key memory. paper shows 1 or 2 modules near the middle of the transformer is best
    'km_num_keys': 128,           # defaults to 128, but can be increased to 256 or 512 as memory allows
    'se_full_attn': False    # only turn on this flag to override and turn on full attention for all sequence lengths. for comparison with LSH to show that it is working
}

parameter_dict_transformer = {
    'vocab_size': 50000, # num_tokens in other models
    'sequence_len': 100, # max_sequance_len in other models
    'd_model': 300, # dim in other models
    'heads': 8, # d_model % heads == 0 - required
    'layers': 2, # Depth in other models
    'dropout': 0.1
}

if __name__ == '__main__':

    seq_len = 128
    batch_size = 200
    vocab_size = 5000

    data = np.random.randint(0, vocab_size, (batch_size, seq_len))
    masks = np.zeros((batch_size, seq_len))
    for i in range(batch_size):
        sent_len = np.random.randint(10, seq_len, 1)[0]
        masks[i, :seq_len-sent_len] = 1

    data = torch.from_numpy(data).type(torch.LongTensor)
    masks = torch.from_numpy(masks).type(torch.BoolTensor)

    model = GenearalTransformer('transformer', 5000, seq_len, 128, 8, 1)
    output = model(data, masks)
    print(output.shape)

    model = GenearalTransformer('linformer', 5000, seq_len, 128, 8, 1)
    output = model(data, masks)
    print(output.shape)

    model = GenearalTransformer('reformer', 5000, seq_len, 128, 8, 1)
    output = model(data, masks)
    print(output.shape)

    model = GenearalTransformer('performer', 5000, seq_len, 128, 8, 1)
    output = model(data, masks)
    print(output.shape)