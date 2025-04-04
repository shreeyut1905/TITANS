from __future__ import annotations
from typing import Callable 

from math import ceil 
from copy import deepcopy 
from functools import partial 
from collections import namedtuple 

import tqdm 
import torch 
from torch import nn,stack,cat 
import torch.nn.functional as F 
from torch.nn import Module,ModuleList,Linear 

flex_attention = None 

try:
    from torch.nn.attention.flex_attention import flex_attention,create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass
def create_mac_block_mask(seq_len,window_size,persist_mem_len,sliding=False):
    def create_mac_mask(_,__,q_idx,kv_idx):
        is_persist_mem = kv_idx < persist_mem_len
        kv_without_mem = kv_idx - persist_mem_len
        casual_mask = q_idx >=  kv_without_mem
        if not sliding:
            block_diagonal = (q_idx//window_size) == (kv_without_mem//window_size)
            casual_mask = casual_mask & block_diagonal 
        else:
            sliding_mask = (q_idx - kv_without_mem) < window_size
            casual_mask = casual_mask & block_diagonal 
        return is_persist_mem | (~is_persist_mem & casual_mask)
    block_mask = create_block_mask(create_mac_mask,B=None,H=None,Q_LEN=seq_len,LV_LEN=seq_len+persist_mem_len,_compile=True)
    return block_mask 

from einops import repeat,rearrange,pack,unpack,einsum
from einops.layers.torch import Rearrange

from axial_positional_embedding import ContinuousAxialPositionalEmbedding
from rotary_embedding_torch import RotaryEmbedding

from x_transformers.attend import Attend
from hyper_connections import get_init_and_expand_reduce_stream_functions

from  titans_pytorch.neural_memory 
    
    
