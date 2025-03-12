from __future__ import annotations 
import math 
from typing import Callable 

import torch   
from torch import Tensor
from torch.nn import Module 
import torch.nn.functional as F 

from einops import rearrange,repeat,reduce,pack,unpack 

def exists(v):
    return v is not None 

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None 

def pad_at_dim(t,pad,dim=-1,value=0.):
    dims_from_right = -(-dim-1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0,0)*dims_from_right)
    return F.pad(t,(*zeros,*pad),value=value)

def pack_one_with_inverse(t,pattern):
    packed,packed_shape = pack([t],pattern)

    def inverse(out,inv_pattern):
        inv_pattern = default(inv_pattern,pattern)
        return unpack(out,packed_shape,inv_pattern)[0]
    return packed,inverse

@torch.jit.script
def binary_operator(
    a : tuple[Tensor,Tensor],
    b : tuple[Tensor,Tensor]
):
    a_i , kv_i = a 
    a_j , kv_j = b 
    return a_j * a_i ,  torch.addcmul(kv_j,a_j,kv_i)

def associative_scan(
        operator:Callable,
        elems : tuple[Tensor,Tensor]
):
    num_elems = int(elems[0].shape[1])
    if not all(int(elem.shape[1])==num_elems for elem in elems[1:]):
        raise ValueError('Array inputs to associative scan must have the same'
                         'first dimension. (saw:{})'
                         .format([elem.shape for elem in elems]))
    
    def _scan(elems):
        num_elems = elems[0].shape[1]

        if num_elems < 2 :
            return elems 
        
        reduced_elems = operator(
            [elem[:,:-1:2] for elem in elems],
            [elem[:,1:2] for elem in elems])
        
        odd_elems  = _scan(reduced_elems)

        if num_elems % 2 == 0 :
            even_elems  = operator(
                [e[:,:-1] for e in odd_elems],
                [e[:,2::2] for e in elems])
        else:
            even_elems = operator(
                odd_elems,
                [e[:,2::2] for e in elems])
        
        even_elems = [
            torch.cat([elem[:,:1],result],dim=1)
            for (elem,result) in zip(elems,even_elems)]
        return list(map(_interleave,even_elems,odd_elems))
    return _scan(elems)
def _interleave(a,b):
    a_axis_len , b_axis_len = a.shape[1],b.shape[1]
    output_axis_len = a_axis_len + b_axis_len

    if (a_axis_len == (b_axis_len + 1)):
        b = pad_at_dim(b,(0,1),dim=1)
    
    stacked = torch.stack([a,b],dim=2)
    interleaved = torch.flatten(stacked,start_dim=1,end_dim=2)
    return interleaved[:,:output_axis_len]

class AssoScan(Module):
    def __init__(
            self,
            use_accelerated = False
    ):
        super().__init__()
        self.use_accelerated = use_accelerated
    def forward(self,gates,inputs,prev=None,remove_prev=None):
        remove_prev = default(remove_prev,exists(prev))

        inputs,inverse_pack_weight_shape = pack_one_with_inverse(inputs,'b n * ') 
        gates , _ = pack_one_with_inverse(gates,'b n *')
        if exists(prev):
            prev , _  = pack_one_with_inverse(prev,'b *')
        if exists(prev):
            inputs,_ = pack([prev,inputs],'b*d')
            gates = pad_at_dim(gates,(1,0),value=1.,dim=-2)
        if not self.use_accelerated:
            _,out = associative_scan(binary_operator,(gates,inputs))
            if remove_prev:
                out = out[:,1:]
            return inverse_pack_weight_shape(out)
        from accelerated_scan.triton import scan as triton_scan
        from accelerated_scan.warp import scan as warp_scan
        scan = triton_scan if gates.is_cuda else warp_scan

        def accelerate_scan_fn(gates,inputs):
            gates = gates.expand_as(inputs)
            gates , inputs = tuple(rearrange(t,'b n d -> b d n') for t in (gates,inputs))
            seq_len = gates.shape[-1]
            next_power_two_seq_len = 2 ** max(5,int(math.ceil(math.log2(seq_len))))

            gates = F.pad(gates,(0,next_power_two_seq_len  - seq_len))
            inputs = F.pad(inputs,(0,next_power_two_seq_len - seq_len))

            outputs  = scan(gates.contiguous(),inputs.contiguous())
            outputs = outputs[...,:seq_len]
            outputs = rearrange(outputs,'b d n -> b n d')
            return outputs 
        out = accelerate_scan_fn(gates,inputs)
        if remove_prev:
            out = out[:,1:]
        return inverse_pack_weight_shape(out)
    