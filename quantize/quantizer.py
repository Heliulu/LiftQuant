import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np
import pdb
import math

CLIPMIN = 1e-5




def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x



class ActivationQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        quant_group=[1],
        lwc=False,
        lac=False,
        **kwargs
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        #print(shape,'3')
        super().__init__()
        
        self.symmetric = symmetric
        
        self.n_bits = n_bits
        
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        
        self.metric = metric
        
        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

       
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        
        
        self.lac = lac
        init_value = 2.           # inti value of learnable weight clipping
        if lac: 
            self.bound_factor = nn.Parameter(torch.ones(quant_group)*init_value)
        self.sigmoid = nn.Sigmoid()

        self.enable = True

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        
        

    def fake_quant(self, x, scale, round_zero_point):
        
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        
        return x_dequant
    

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)

        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError()   
        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        
        return x_dequant

    def per_token_dynamic_calibration(self, x):
        
        reduce_shape = [-1]
        
        if self.lac:
            if self.n_bits > 1:
                xmin = x.amin(reduce_shape, keepdim=True)
                xmax =  x.amax(reduce_shape, keepdim=True)
                xavg = (xmax+xmin)/2
                xmax = self.sigmoid(self.bound_factor)*(xmax-xavg)+xavg
                xmin = self.sigmoid(self.bound_factor)*(xmin-xavg)+xavg    
            else:
                xavg = x.mean(reduce_shape, keepdim=True)
                xstd = x.std(reduce_shape, keepdim=True)
                xmax = xavg + 0.8*xstd
                xmin = xavg -0.8* xstd
        else:
            xmin = x.amin(reduce_shape, keepdim=True)
            xmax =  x.amax(reduce_shape, keepdim=True)
            
        if self.symmetric:
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max / (2**(self.n_bits-1)-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            
            scale = range / (2**self.n_bits-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)
            
        self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4)
        
    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point


if __name__ == "__main__":
    weight_quant_params = {
        "n_bits": 8,
        "symmetric": False,
        "dynamic_method": "per_channel",
    }
    
    weight=torch.randn([4096,4096])
    quanter = ActivationQuantizer(**weight_quant_params,shape=weight.shape)
    quanter(weight)