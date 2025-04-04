import torch 
from torch import nn , cat 
import torch.nn.functional as F 
from torch.nn import Module,ModuleList,Parameter,ParameterList

from einops import rearrange

def l2norm(t):
    return F.normalize(t,dim=-1)

class LayerNorm(Module):
    def __init__(self,dim):
        super().__init__()
        self.ln  = nn.LayerNorm(dim,elementwise_affine=False)
        self.gamma  = Parameter(torch.zeros(dim))
    def forward(self,x):
        gamma = self.gamma 
        if gamma.ndim == 2:
            gamma = rearrange(gamma,'b d -> b 1 d')
        return self.ln(x) * (gamma + 1.)

class ResidualNorm(Module):
    def __init__(self,dim,model:Module):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.model = model
    def forward(self,x):
        out = self.model(x)
        return self.norm(out) + x 

class MemoryMLP(Module):
    def __init__(self,dim,depth,expansion_factor=2):
        super().__init__()
        dim_hidden = int(dim*expansion_factor)
        dims = (dim,*((dim_hidden,)*(depth-1)),dim)
        self.weights = ParameterList([Parameter(torch.randn(dim_in,dim_out)) for dim_in,dim_out in zip(dims[:-1],dims[1:])])
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
    def forward(self,x):
        for ind,weight in enumerate(self.weights):
            is_first = ind == 0 
            if not is_first:
                x = F.gelu(x)
            x = x @ weight
        return x
class GatedResidualMemoryMLP(Module):
    def __init__(self,dim,depth,expansion_factor=4.):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)
        self.weights = ParameterList([
            Parameter([
                Parameter(torch.randn(dim,dim_hidden)),
                Parameter(torch.randn(dim_hidden,dim)),
                Parameter(torch.randn(dim*2,dim)),
            ]) for _ in range(depth)
        ])
        self.final_proj = Parameter(torch.randn(dim,dim))
        for param in self.parameters():
            nn.init.xavier_uniform(param)
    def forward(self,x):
        for weight1,weight2,to_gates in self.weights:
            res = x
            hidden = x@weight1
            hidden = F.gelu(hidden)
            branch_out = hidden@weight2
            gates = cat((branch_out,res),dim=-1)@ to_gates
            x = res.lerp(branch_out,gates.sigmoid())
            return x@self.final_proj
        
class FactorizedMemoryMLP(Module):
    def __init__(self,dim,depth,k=32):
        super().__init__()
        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim,k)),
                Parameter(torch.randn(k,dim)),
            ]) for _ in range(depth)
        ])
        for weight1,weight2 in self.weights:
            nn.init.xavier_uniform_(weight1)
            nn.init.xavier_uniform_(weight2)
    def forward(self,x):
        for ind,(weight1,weight2) in enumerate(self.weights):
            is_first = ind == 0 
            if not is_first:
                x = F.gelu(x)
            x = x@weight1@weight2
        return x

class MemorySwigGluMLP(Module):
    def __init__(self,dim,depth=1,expansion_factor=4.):
        super().__init__()
        dim_inner = int(dim*expansion_factor*2/3)
        weights = []
        for _ in range(depth):
            weights.append(ParameterList([
                Parameter(torch.randn(dim,dim_inner*2)),
                Parameter(torch.randn(dim_inner,dim)),
            ]))
        self.weights = ParameterList(weights)
        self.norm = LayerNorm(dim)
    def forward(self,x):
        for w1,w2 in self.weights:
            residual = x 
            x,gates = (x@w1).chunk(2,dim=-1)
            x = x*F.gelu(gates)
            x = x@w2
            x  = x + residual
        return self.norm(x)

class MemoryAttention(Module):
    def __init__(self,dim,scale=8.,expansion_factor=2.):
        super().__init__()
        self.scale= scale 
        dim_ff_hidden = int(dim*expansion_factor)
        self.weights = ParameterList([
            Parameter(torch.randn(dim,dim)),
            Parameter(torch.randn(dim,dim)),
            Parameter(torch.randn(dim,dim)),
            Parameter(torch.randn(dim,dim_ff_hidden)),
            Parameter(torch.randn(dim_ff_hidden,dim)),
        ])
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
    def forward(self,x):
        wq,wk,wv,ffw1,ffw2 = self.weights
        q = l2norm(x@wq)
        k = l2norm(x@wk)
        v = x@wv
        attn_out = F.scaled_dot_product_attention(
            q,k,v,
            scale = self.scale,
            is_casual = True
        )
        h = F.gelu(x@ffw1)
        ff_out = h@ffw2
        return attn_out + ff_out 