import torch
import torch.nn as nn
import torch.nn.functional as F


from scipy import linalg


class PCAQuant(nn.Module):
    def __init__(
        self,
        act_quant_params: dict = {},
        trans_dim = 128,
    ):
        super().__init__()
        self.act_quant_params = act_quant_params
        self.trans_dim = trans_dim
    
    def hadamardtrans(self,x):
        hadamard_trans = torch.tensor(linalg.hadamard(self.trans_dim))/torch.sqrt(torch.tensor(self.trans_dim))
        x = torch.einsum('bthi,ij->bthj', x, hadamard_trans.to(x.device).to(x.dtype))
        return x
    def ihadamardtrans(self,x):
        hadamard_trans = torch.tensor(linalg.hadamard(self.trans_dim))/torch.sqrt(torch.tensor(self.trans_dim))
        x = torch.einsum('bthi,ji->bthj', x, hadamard_trans.to(x.device).to(x.dtype))
        return x
    
    def forward(self, input: torch.Tensor):
        return input
    
    def Rotation(self, input:torch.Tensor):
        # 进行trans
        #print(self.act_quant_params['quant_method'] )
        input = input.reshape([input.shape[0],input.shape[1],-1,self.trans_dim]) 
        input = self.hadamardtrans(input)
        input = input.reshape([input.shape[0],input.shape[1],-1])
        return input 
    
    def IRotation(self, input:torch.Tensor):
    
        # 进行逆trans
        input = input.reshape([input.shape[0],input.shape[1],-1,self.trans_dim]) 
        
        input = self.ihadamardtrans(input)
        
        input = input.reshape([input.shape[0],input.shape[1],-1])
        return input 
    
    def set_quant_state(self, act_quant: bool = False):
        self.use_act_quant = act_quant


#from quant import *
class WQuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

        
    def __init__(
        self,
        org_module: nn.Linear,
        wbits = 4
    ):
        super().__init__()
        
        self.fwd_func = F.linear
        self.quantizer = Quantizer()
        self.quantizer.configure(
            wbits, perchannel=True, sym=False, mse=True
        )
        self.register_buffer('weight',org_module.weight.data)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        
    def find_params(self):   
        self.quantizer.find_params(self.weight, weight=True)
        
    def forward(self, input: torch.Tensor):
        
        bias = self.bias
        #print(weight)
        # No set quant hear
        out = self.fwd_func(input, quantize(self.weight, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq), bias)


        return out
    


class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

        
    def __init__(
        self,
        org_module: nn.Linear,
        qnums = 15,
        LoRa = False,
        LoRa_dim = 0,
    ):
        super().__init__()
       
        self.fwd_func = F.linear
        
        qweight, scale, zero = self.quantize_to_int4(org_module, qnums)
        
        qweight = self.pack_int4(qweight)
        #print(org_module.weight.max(dim=1))

        weight = ((self.unpack_int4(qweight)-zero)*scale).to(org_module.weight.data.dtype)
        #print(weight.max(dim=1))
        #print((org_module.weight-weight).abs().sum())
        del weight
        self.dtype = org_module.weight.data.dtype
        self.register_buffer('weight',qweight)
        self.register_buffer('scale',scale)
        self.register_buffer('zero',zero)
        
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        
        
    def forward(self, input: torch.Tensor):
        weight = ((self.unpack_int4(self.weight)-self.zero)*self.scale ).to(self.dtype)
        bias = self.bias
        out = self.fwd_func(input, weight, bias)
        

        return out
    
    def quantize_to_int4(self, model, qnums):
        ## per dim0 quant
        scale = model.scale
        zero = model.zero
        #qtensor = torch.round(( tensor - minw ) / scale).clamp(0, qnums).to(torch.int8)
        #print(tensor)
        #print(( tensor - minw ) / scale)
        tensor = model.weight.data
        qtensor = (torch.round( tensor.to(scale) /scale) + zero).clamp(0, qnums).to(torch.int8)
        return qtensor, scale, zero      

    def pack_int4(self, qtensor):
        qtensor = (qtensor[:,::2]<<4)|(qtensor[:, 1::2])
        return qtensor
        
    def unpack_int4(self, qtensor):
        #print(qtensor)
        low = qtensor & 0x0F
        high = (qtensor >> 4) & 0x0F
        OC = high.shape[0]
        
        tensor = torch.cat([high.unsqueeze(-1),low.unsqueeze(-1)], dim=-1).reshape(OC,-1)
        return tensor
        
class Quant_LoRA_Linear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

        
    def __init__(
        self,
        org_module: nn.Linear,
        input_indices
    ):
        super().__init__()
        OC,IC = org_module.weight.data.shape
        pIC  = input_indices.shape[0]
        bias = org_module.bias is not None
        self.Linear1 = torch.nn.Linear(IC+pIC,OC,bias = bias,device = org_module.weight.data.device, dtype=org_module.weight.data.dtype)
        #self.Linear2 = torch.nn.Linear(pIC,OC, bias = False)
        
        weight = org_module.weight.data
        weight1 = self.Linear1.weight.data
        weight1[:,pIC:] = weight.clone()
        #weight1[:,input_indices+pIC] = 0
        weight1[:,:pIC] = weight[:,input_indices]*0
        #print(org_module.weight.data)
        #print(weight1)
        #print(weight2)
        #weight1 = weight.clone()
        #weight2 = weight[:,input_indices]
        #weight1[:,input_indices] = 0
        self.Linear1.weight.data = weight1
        #self.Linear2.weight.data=weight2
        
        
        if org_module.bias is not None:
            self.Linear1.bias.data = org_module.bias
        
        select_matrix = torch.zeros(pIC,IC,device = org_module.weight.data.device, dtype=org_module.weight.data.dtype)
        rows = torch.arange(pIC)
        select_matrix[rows,input_indices] = 1
        self.input_indices = input_indices
        self.gptq_flag = False
        self.register_buffer('select_matrix', select_matrix)
        
    def Right_Hadamard(self, tensor, transdim=128):
        W = tensor.shape[-1]
        if W%transdim>0:
            print("error")
        tensor = tensor.reshape(-1,W//transdim,transdim)
        R = (torch.tensor(linalg.hadamard(transdim))/torch.sqrt(torch.tensor(transdim))).to(tensor.device).to(tensor.dtype)
        tensor =  tensor@R  
        tensor = tensor.reshape(-1,W)
        return tensor    
    
    def forward(self, input: torch.Tensor):
        input = torch.cat([input @ (self.select_matrix.T) , input],dim=-1)
        out = self.Linear1(input)
        #out +=  self.Linear2(input @ (self.select_matrix.T))

        return out
    
class Quant_LoRA_Linear1(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

        
    def __init__(
        self,
        org_module: nn.Linear,
        input_indices
    ):
        super().__init__()
        OC,IC = org_module.weight.data.shape
        pIC  = input_indices.shape[0]
        bias = org_module.bias is not None
        self.Linear1 = torch.nn.Linear(IC,OC,bias = bias,device = org_module.weight.data.device, dtype=org_module.weight.data.dtype)
        self.Linear2 = torch.nn.Linear(pIC,OC, bias = False)
        
        weight = org_module.weight.data
     
        #print(org_module.weight.data)
        #print(weight1)
        #print(weight2)
        weight1 = weight.clone()
        weight2 = weight[:,input_indices]
        weight1[:,input_indices] = 0
        self.Linear1.weight.data = weight1
        self.Linear2.weight.data=weight2
        
        
        if org_module.bias is not None:
            self.Linear1.bias.data = org_module.bias
        
        select_matrix = torch.zeros(pIC,IC,device = org_module.weight.data.device, dtype=org_module.weight.data.dtype)
        rows = torch.arange(pIC)
        select_matrix[rows,input_indices] = 1
        self.input_indices = input_indices
        self.gptq_flag = False
        self.register_buffer('select_matrix', select_matrix)
        
    def Right_Hadamard(self, tensor, transdim=128):
        W = tensor.shape[-1]
        if W%transdim>0:
            print("error")
        tensor = tensor.reshape(-1,W//transdim,transdim)
        R = (torch.tensor(linalg.hadamard(transdim))/torch.sqrt(torch.tensor(transdim))).to(tensor.device).to(tensor.dtype)
        tensor =  tensor@R  
        tensor = tensor.reshape(-1,W)
        return tensor    
    
    def forward(self, input: torch.Tensor):
        #input = torch.cat([input @ (self.select_matrix.T) , input],dim=-1)
        out = self.Linear1(input)
        out +=  self.Linear2(input @ (self.select_matrix.T))

        return out

    
if __name__ == "__main__":
    layer0 = nn.Linear(256,256,bias=False)
    _, input_indices = torch.topk(torch.randn(256),64)
    layer1 = Quant_LoRA_Linear1(layer0,input_indices)
    layer1.gptq_flag = True
    a = torch.randn(1,56,256)
    
    
    print((layer1(a)-layer0(a)).abs().sum())
    print(layer1(a)[0,0,:10])
    print(layer0(a)[0,0,:10])
    
    