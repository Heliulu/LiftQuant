import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import ActivationQuantizer

from scipy import linalg

class QuantMatMul(nn.Module):
    def __init__(
        self,
        x1_quant_params: dict = {},
        x2_quant_params: dict = {},
        disable_act_quant=False,
        matmul_func=torch.bmm,
    ):
        super().__init__()
        # de-activate the quantized forward default
        self.use_act_quant = False
        # initialize quantizer
        self.i_cluster_counts = None
        self.x1_quantizer = ActivationQuantizer(**x1_quant_params)
        self.x2_quantizer = ActivationQuantizer(**x2_quant_params)
        self.matmul_func = matmul_func

        self.disable_act_quant = disable_act_quant


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def quant_x1(self, x1):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
        return x1

    def quant_x2(self, x2):
        if self.use_act_quant:
            x2 = self.x2_quantizer(x2)
        return x2

    def forward(self, x1, x2):
        out = self.matmul_func(x1, x2)
        return out


class QuantPCAMatMul(nn.Module):
    def __init__(
        self,
        x1_quant_params: dict = {},
        x2_quant_params: dict = {},
        disable_act_quant=False,
        matmul_func=torch.bmm,
        out_quant=False,
        heads=32,
        head_dim=128,
        trans_dim=128
    ):
        super().__init__()
        # de-activate the quantized forward default
        self.use_act_quant = False
        # initialize quantizer
        self.i_cluster_counts = None
        self.x1_quantizer = ActivationQuantizer(**x1_quant_params,quant_group=[heads,1,1])
        self.x2_int4_quantizer =  ActivationQuantizer(**x2_quant_params,quant_group=[heads,1,1])
       

        self.matmul_func = matmul_func

        self.disable_act_quant = disable_act_quant

        if x2_quant_params['quant_bias']:
            self.PCA_means = nn.Parameter(torch.zeros((heads,head_dim)))
        #self.PCA_trans = None
        
        self.x1_quant_params = x1_quant_params
        self.x2_quant_params = x2_quant_params
        
        self.out_quant=out_quant
        
        self.heads=heads
        self.head_dim=head_dim
        self.trans_dim = trans_dim

    def set_quant_state(self, act_quant: bool = False):
        self.use_act_quant = act_quant
    
    def hadamardtrans(self,x):
        hadamard_trans = torch.tensor(linalg.hadamard(self.trans_dim))/torch.sqrt(torch.tensor(self.trans_dim))
        x = torch.einsum('bhti,ij->bhtj', x, hadamard_trans.to(x.device).to(x.dtype))
        return x
    
    def ihadamardtrans(self,x):
        hadamard_trans = torch.tensor(linalg.hadamard(self.trans_dim))/torch.sqrt(torch.tensor(self.trans_dim))
        x = torch.einsum('bhti,ji->bhtj', x, hadamard_trans.to(x.device).to(x.dtype))
        return x
    
    def quant_x1(self, x1):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
        return x1
    
    def quant_x2(self, x2):
        
        if self.use_act_quant:
            if self.x2_quant_params['quant_bias']:
                x2 = x2-self.PCA_means.view([1, self.heads,1, self.head_dim])
                
            if self.trans_dim<self.head_dim:
                x2 = x2.transpose(1,2)
                x2 = x2.reshape([x2.sahpe[0],x2.shape[1],-1,self.trans_dim])
                x2 = x2.transpose(1,2)
                
            if self.x2_quant_params['quant_method'] == "Hadamard":
                x2 = self.hadamardtrans(x2)
                
            x2 = self.x2_int4_quantizer(x2)
            
            if self.x2_quant_params['quant_method'] == "Hadamard":
                x2 = self.ihadamardtrans(x2)

            if self.trans_dim<self.head_dim:
                x2 = x2.transpose(1,2)
                x2 = x2.reshape([x2.sahpe[0],x2.shape[1],-1,self.head_dim])
                x2 = x2.transpose(1,2)
                
            if self.x2_quant_params['quant_bias']:
                x2 = x2+self.PCA_means.view([1,self.heads,1, self.head_dim])
                
        return x2

    def forward(self, x1, x2):
        out = self.matmul_func(x1, x2)
        return out