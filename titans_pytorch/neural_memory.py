from __future__ import annotations
from typing import Callable 

import math 
from functools import partial 
from itertools import zip_longest
from collections import namedtuple 

import torch 
from torch import nn,stack,cat,is_tensor,tensor,Tensor
import torch.nn.functional as F 
from torch.nn import Linear,Module,Parameter,ParameterList,ParameterDict
from torch.func import functional_call,vmap,grad
from torch.utils._pytree import tree_map,tree_flatten,tree_unflatten

from tensordict import TensorDict

from titans_pytorch.associative_scan import AssoScan 
from titans_pytorch.memory_models import (
    MemoryMLP,
    ResidualNorm
)
LinearNoBias = partial(Linear,bias=False)
NeuralMemState = namedtuple('NeuralMemState',[
    'seq_index',
    'weight',
    'cache_storage_segment',
    'states',
    'updates',
])

def mem_state_detach(state,NeuralMemState):
    assert isinstance(state,NeuralMemState)
    state = tree_map(lambda t: t.detach() if is_tensor(t) else t,tuple(state))
    return NeuralMemState(*state)

def exists(x):
    return x is not None
def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None
def identity(x):
    return x

def xnor(x,y):
    return(x^y)

def divisible_by(x,y):
    return (x%y) == 0

def safe_cat(inputs,dim=-2):
    inputs = tuple(filter(exists,inputs))
    if len(inputs) == 0:
        return None 
    elif len(inputs) == 1:
        return inputs[0]
    return cat(inputs,dim=dim)

def is_empty_tensor(t):
    return t.numel() == 0
def dict_get_value_shapes(td):
    return [v.shape for k,v in td.items() ]

def rearrange_dict_values(td,pattern,**kwargs):
    return td.apply(lambda t : rearrange(t,pattern,**kwargs))


def repeat_dict_values(td,pattern,**kwargs):
    return td.apply(lambda t: repeat(t,pattern,**kwargs))
def pair(v):
    return (v,v) if not isinstance(v,tuple) else v

def round_down_multiple(seq,mult):
    return seq // mult * mult
def round_up_multiple(seq,mult):
    return math.ceil(seq / mult) * mult 
