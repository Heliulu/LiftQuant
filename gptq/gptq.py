import math
import time

import torch
import torch.nn as nn
import transformers

from .quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.float()
        variance = inp.to(torch.float32).pow(2).mean(-1, keepdim=True)
        inp = inp * torch.rsqrt(variance+0.00001)
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.02, groupsize=-1, actorder=False, static_groups=False, LoRa = False, LoRa_dim = 0, Use_RTN = False
    ):
       
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()
        
        #Use_RTN = True
        if not self.quantizer.ready():
            if getattr(self.layer, 'scaleflag', None):
                self.quantizer.scaleflag = self.layer.scaleflag
                self.quantizer.scale = self.layer.scale
            self.quantizer.find_params(W, weight=True)
        if Use_RTN:
            Q = quantize(
                        self.layer.weight.data, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    )
            self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
            print(123)
            return 0 
        
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)
                
      
        if LoRa:
            import copy
            quantizer1 = copy.deepcopy(self.quantizer)
            #quantizer1.maxq = torch.tensor(255)
            quantizer2 = copy.deepcopy(self.quantizer)
            quantizer1.find_params(W[:, :LoRa_dim], weight=True)
            quantizer2.find_params(W[:, LoRa_dim:], weight=True)
            #quantizer1.find_params(W, weight=True)
            #quantizer2.find_params(W, weight=True)
            groups= [quantizer1, quantizer2]
            #print(quantizer1.scale.mean())
            #print(quantizer2.scale)
        
        
     
        if actorder:
            if LoRa:
                tmp = torch.diag(H)
                #scale
                #tmp[:LoRa_dim] = tmp[:LoRa_dim] *0.999
                perm1 = torch.argsort(tmp[:LoRa_dim], descending=True)
                perm2 = torch.argsort(tmp[LoRa_dim:], descending=True)
                #tmp [:LoRa_dim] = tmp[:LoRa_dim]/quantizer1.scale.mean()
                #tmp [LoRa_dim:] = tmp[LoRa_dim:]/quantizer2.scale.mean()
                perm = torch.argsort(tmp, descending=True)
                #perm = torch.cat([perm1,perm2+LoRa_dim],dim=0)
                perm = torch.cat([perm2+LoRa_dim,perm1],dim=0)
                del tmp
            else:
                perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
        else:
            if LoRa:
                perm = torch.cat([torch.arange(LoRa_dim,self.columns,1), torch.arange(0,LoRa_dim,1)],dim=0)
                W = W[:, perm]
                H = H[perm][:, perm]
                invperm = torch.argsort(perm)

        #print(perm[0:4])
        
        
        def robust_cholesky(H,p):
            if not torch.allclose(H, H.T):
                raise ValueError("Matrix is not symmetric.")
            for percdamp in [0.01,0.02,0.05,0.1,0.2,0.5,1]:
                try:
                    damp = p * percdamp * torch.mean(torch.diag(H))
                    diag = torch.arange(self.columns, device=self.dev)
                    H[diag, diag] += damp
                    H = torch.linalg.cholesky(H)
                    return H, p*percdamp
                except torch._C._LinAlgError as e:
                    print("Continue add damp")
            
            raise RuntimeError("Can't do cholesky.")

        H_copy = H.clone()
        W_copy = W.clone()
        for p in [1,2,3,4,5]:
            Losses = torch.zeros_like(W)
            Q = torch.zeros_like(W)
            W = W_copy.clone()
            H, percdamp = robust_cholesky(H_copy, p)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

        
            for i1 in range(0, self.columns, blocksize):
                if LoRa and i1== self.columns-LoRa_dim:
                    groups[0].find_params(W[:, -LoRa_dim:], weight=True)
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()

                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]
                if LoRa:
                    if actorder:
                        perm1 = perm[i1:i2]
                for i in range(count):
                    w = W1[:, i]
                    
                    d = Hinv1[i, i]
                    if groupsize != -1:
                        if not static_groups:
                            if (i1 + i) % groupsize == 0:
                                self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                        else:
                            idx = i1 + i
                            if actorder:
                                idx = perm[idx]
                            self.quantizer = groups[idx // groupsize]
             
                    if LoRa:
                        if actorder:
                            if perm1[i]<LoRa_dim:
                                self.quantizer= groups[0]
                
                            else:
                                self.quantizer= groups[1]
                        else:
                            if i1+i<LoRa_dim:
                                self.quantizer= groups[0]
                          
                            else:
                                self.quantizer= groups[1]
                   
                    
                    q = quantize(
                        w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()
                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d
      
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1
                
                Q[:, i1:i2] = Q1
                Losses[:, i1:i2] = Losses1 / 2

                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

                if DEBUG:
                    self.layer.weight.data[:, :i2] = Q[:, :i2]
                    self.layer.weight.data[:, i2:] = W[:, i2:]
                    print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                    print(torch.sum(Losses))

            torch.cuda.synchronize()
      
            print('weight round error', torch.sum(Losses).item(),'percdamp', percdamp)
           
            if actorder:
                Q = Q[:, invperm]

            if isinstance(self.layer, transformers.Conv1D):
                Q = Q.t()
            
            if torch.any(torch.isnan(torch.sum(Losses))):
                print('!!! Nan Loss when use GPTQ at percdamp', percdamp)
    
            else:
                self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
                self.layer.scale = self.quantizer.scale.clone()
                self.layer.zero  = self.quantizer.zero.clone().to(self.layer.weight.data.dtype)

                if DEBUG:
                    print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                return 0 
            

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


