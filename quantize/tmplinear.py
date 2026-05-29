import torch
import torch.nn as nn
import torch.nn.functional as F
from trans_utils import Hadamard_trans, SVDDecomposeTransMatrix, HalfSVDDecomposeTransMatrix, GHalfSVDDecomposeTransMatrix
from quantize.int_linear import *
from gptq.quant import Quantizer

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import os 
from tqdm import tqdm
from quantize.E8Q import *
from trans_utils import *
from itertools import combinations
from torch.utils.checkpoint import checkpoint

from itertools import product
import math
def plot_box_plot(datalist, figname, titlelist, limit=0.5):

    # 创建并排的子图
    if len(datalist) ==1:
        fig, axes = plt.subplots(1, 1, figsize=(30, 10))
    elif len(datalist) ==2:
        fig, axes = plt.subplots(1, 2, figsize=(30, 10))
    elif len(datalist) ==3:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    elif len(datalist) ==4:
        fig, axes = plt.subplots(1, 4, figsize=(40, 10))
    # 绘制箱线图
    for i, data in enumerate(datalist):
        q1 = torch.quantile(data, 0.25, dim=0).to('cpu').numpy()
        q3 = torch.quantile(data, 0.75, dim=0).to('cpu').numpy()
        p1 = torch.quantile(data, 0.01, dim=0).to('cpu').numpy()
        p99 = torch.quantile(data, 0.99, dim=0).to('cpu').numpy()
        min = torch.min(data, dim=0)[0].to('cpu').numpy()
        max = torch.max(data, dim=0)[0].to('cpu').numpy()

        for j in range(data.shape[1]):
            # 第一个胡须（到最小值和最大值）
            axes[i].plot([j+1, j+1], [min[j], max[j]], color="brown", linewidth=1)
            
            # 第二个胡须（到99%分位数）
            axes[i].plot([j+1, j+1], [p1[j], p99[j]], color="orange", linewidth=1)
            
            # 第三个胡须（到25%分位数）
            axes[i].plot([j+1, j+1], [q1[j], q3[j]], color="gray", linewidth=1)

        axes[i].set_title(titlelist[i], fontsize=16)
        if i ==0:
            axes[i].set_ylabel('Activation Value', fontsize=14)
        axes[i].set_xlabel('Channel Index', fontsize=14)
        #axes[i].set_xticklabels([]) 
    
    

    # 添加图例
    
    legend_elements = [
        Line2D([0], [0], color='gray', lw=4, label='25%-75% Percentile'),
        Line2D([0], [0], color='orange', lw=4, label='1%-99% Percentile'),
        Line2D([0], [0], color='brown', lw=4, label='Min-Max')
    ]
    

    plt.tight_layout()
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    plt.savefig(figname)
    plt.close()

    return 0 

class FWTLinear(nn.Module):
    def __init__(
        self,
    ):
        super(FWTLinear, self).__init__()

    def convert_form_tmplinear(
        self,
        tmp_module,
        expc = 'n',
        training_trans = False,
        bits = 2,
        groupsize = -1,
        fast_nearest = True
    ):
        self.maxq = 2**bits-1
        #print(self.maxq)
        self.training_trans = training_trans
        self.groupsize = groupsize
        self.adaquant = True
        self.oc, self.ic =  tmp_module.oc, tmp_module.ic
        
        parts = expc.split('to')
        self.root = int(parts[1])
        self.root2 = int(parts[0])
        if self.root == 8:
            if self.ic >10000:
                self.transdim2 = 128
            else:
                self.transdim2 = 64
        #self.transdim2 = round(math.sqrt(self.ic)/self.root)*self.root
        self.transdim1 = math.ceil(self.ic/self.transdim2)
        self.expic = self.transdim1 * self.transdim2
        
        self.fast_nearest = fast_nearest
        self.expc = expc

        self.bias = tmp_module.bias
        with torch.no_grad():
            self.register_parameter('scale', nn.Parameter(tmp_module.quantizer.scale.data*2*F.sigmoid(tmp_module.quantizer.alpha.data)))
            self.register_parameter('a1', nn.Parameter(self.clamp_ste(tmp_module.a1.data, 0.02, 50)) )
            #self.register_parameter('a2', nn.Parameter( (tmp_module.a3.data.reshape(-1)/tmp_module.a2.data.reshape(-1))))
            self.register_parameter('a2', nn.Parameter( 1./tmp_module.a2.data.reshape(-1)))

            self.weight = tmp_module.orilinear.weight.reshape(self.oc, self.transdim1, -1)  * self.a1.data

            if self.training_trans:
                self.Trans = tmp_module.Trans
                self.weight = self.Trans(self.weight)
                self.Trans.to_buffer()
            else:
                self.weight = Hadamard_trans(self.weight, self.transdim1, self.transdim2)
            self.weight = self.weight * tmp_module.a2.data.repeat_interleave(self.root, dim=-1)
            self.weight = self.weight.reshape(self.oc, -1)
            weight = self.weight
            del self.weight
            self.register_parameter('weight', nn.Parameter(weight))
            self.a1.data = self.a1.data.reshape(-1)
        self.packed_flag = False

    def bit_channel_convert(self, fast=False):
        # convert 8x2bits to 16x1bits
        if self.weight.dtype == torch.float16:
            minmin = 1e-4
            maxmax = 1e4
        else:
            minmin = 1e-7
            maxmax = 1e7
        l2 = torch.clamp(self.weight.pow(2).mean(dim=-1,keepdim=True).pow(0.5), minmin, maxmax)
        norm_weight = self.weight/l2
        norm_weight = norm_weight.reshape(-1,self.root)
        device = self.weight.device
        
        M_path = './lattice/' + self.expc +'.pt'
        T = torch.load(M_path).to(device)

        if fast:   # use to generate null weight in e2e finetune
            alpha = self.root2/self.root
            qnorm_weight = torch.randn(self.weight.shape[0], int(self.weight.shape[1]*alpha)).to(device)
        else:
            if self.fast_nearest:
                qnorm_weight = self.find_nearest_fast(norm_weight, T, self.root2-self.root, batch_size=128, device=device)
            else:
                codes = torch.empty((2**self.root2, self.root2), dtype=torch.float32, device=device)
                for i in range(self.root2):
                    # 周期长度：2^(i+1)
                    repeat_len = 2 ** (i + 1)
                    block = torch.cat([torch.full((2**i,), -1.), torch.full((2**i,), 1.)]).to(self.weight)
                    reps = 2**self.root2 // repeat_len
                    codes[:, i] = block.repeat(reps)
                points = codes@T.t()
                indices = self.find_nearest_in_batches(norm_weight, points, batch_size=1024)    # (N,)
                qnorm_weight = codes[indices] 
            print('related std error in trans domain',(qnorm_weight@T.t() - norm_weight).std()/ norm_weight.std())
        qnorm_weight = qnorm_weight.reshape(l2.shape[0],-1)
    
        ow = self.get_oriweight()
        oqw = self.get_oldqweight()
        qw = ((qnorm_weight*l2).reshape(-1,self.root2)@T.t()).reshape(l2.shape[0],-1)
        if fast==False:
            print('std error in ori domain', (self.weight-qw).std(), self.weight.std())
        del self.weight
        self.register_parameter('weight', nn.Parameter(qnorm_weight * l2 ))
        
        self.maxq = 1
        self.scale.data = l2*2. 
        M = torch.zeros(self.transdim2//self.root * self.root2, self.transdim2).to(T)
        for i in range(self.transdim2//self.root):
            row_start = i * self.root2
            col_start = i * self.root
            M[row_start:row_start+self.root2, col_start:col_start+self.root] = T.t()

        a2 = self.a2.data.repeat_interleave(self.root2, dim=-1)       
        del self.a2
        self.register_parameter('a2', nn.Parameter(a2.to(self.weight)))
        self.root = 1
        mr = self.Trans.linear_right.data.to(device)
        del self.Trans.linear_right 

        self.Trans.register_parameter('linear_right', nn.Parameter(M@mr))
        if fast == False:
            print('related std error in ori domain', (self.get_weight()-ow).std()/ow.std())
            print('related std error in ori domain, Uniform Quantizater', (oqw-ow).std()/ow.std())

    def find_nearest_in_batches(self, A, points, batch_size=128):
        N = A.shape[0]
        M = points.shape[0]
        indices_list = []
        with torch.no_grad():
            x_sq = (points ** 2).sum(dim=1, keepdim=True).T  #[1, N]
            for i in tqdm(range(0, N, batch_size)):
                A_chunk = A[i:i+batch_size]                           # (bs, 8)
                # 距离计算: (bs, M)
                #dist2 = ((A_chunk[:, None, :] - points[None, :, :])**2).sum(dim=2)
                
                # 3) 计算距离矩阵
                z_sq = (A_chunk ** 2).sum(dim=1, keepdim=True)  # (bs, 1)
                dists = z_sq + x_sq - 2 * A_chunk @ points.T  # (bs, N)

                idx = dists.argmin(dim=1)                             # 最近格点索引
                indices_list.append(idx)
        return torch.cat(indices_list, dim=0)

    def find_nearest_fast(self, W, M, padding_length, batch_size=128, device='cuda'):
        M = M.to(W).to(torch.float32)
        # 预计算零空间基和逆矩阵（对于固定的M，这些是不变的）
        D_out, D_in = M.shape
        with torch.no_grad():
            try:
                _, _, Vh = torch.linalg.svd(M)
                N = Vh[D_out:]
                M_square = torch.cat([M, N], dim=0)
                M_square_inv = torch.linalg.inv(M_square)
            except torch.linalg.LinAlgError:
                print("Benchmark SVD/inv failed. M might be singular.")
                return float('inf')
                
            # 预计算 padding 向量
            num_candidates_exp = padding_length
            padding_vectors = torch.tensor(
                list(product([-1, 1], repeat=num_candidates_exp)), 
                dtype=torch.float32, device=device
            )
            num_candidates = padding_vectors.shape[0]
        encoded_vectors_list = []
        vectors_to_encode = W.view(-1, D_out)
        num_vectors = vectors_to_encode.shape[0]
        for start in tqdm(range(0, num_vectors, batch_size)):
            end = min(start + batch_size, num_vectors)
            
            with torch.no_grad():
                z_batch = vectors_to_encode[start:end]
                bs = z_batch.shape[0]
                # --- 生成候选集 (与训练时逻辑相同) ---
                z_expanded = z_batch.unsqueeze(1).expand(-1, num_candidates, -1)
                paddings_expanded = padding_vectors.unsqueeze(0).expand(bs, -1, -1)
                Y_subset = torch.cat([z_expanded, paddings_expanded], dim=2)
                Y_subset = Y_subset @ M_square_inv.T
                Y_subset = torch.sign(Y_subset)
                
                # --- 在候选集中寻找真正的最近邻 ---
                grid_points_subset = Y_subset @ M.T
                z_expanded_for_dist = z_batch.unsqueeze(1)
                dist_sq_matrix_subset = torch.sum((z_expanded_for_dist - grid_points_subset) ** 2, dim=2)
                
                # 找到每个 z 的最近邻
                _, nn_idx = torch.min(dist_sq_matrix_subset, dim=1)

                # ------ 从 Y_subset 中选出最终的码字 ------
                # nn_idx: (bs,) -> (bs, 1, 1) -> (bs, 1, D_in)
                # 使用 gather 从 Y_subset 中精确地挑选出每个向量对应的最佳码字
                best_y = torch.gather(Y_subset, 1, nn_idx.view(-1, 1, 1).expand(-1, 1, D_in)).squeeze(1)
                
                encoded_vectors_list.append(best_y)
            
        W_encoded_flat = torch.cat(encoded_vectors_list, dim=0)
        return W_encoded_flat

    def round_ste(self, x: torch.Tensor):
        """
        Implement Straight-Through Estimator for rounding operation.
        """
        return (x.round() - x).detach() + x

    def Hadamard_trans(self, data, dim1, dim2, inv):
        data_shape = data.shape 
        data=data.reshape(-1,dim1,dim2)
        H1 = self.H1
        H2 = self.H2
        if inv:
            H1 = H1.T
            H2 = H2.T
        H1 = H1.to(data)
        H2 = H2.to(data)
        data = H1@data@H2
        return data.reshape(data_shape)

    def clamp_ste(self, x: torch.Tensor, min, max):
        return (x.clamp(min,max) - x).detach() + x
    
    def get_oldqweight(self):
        if True:
            if self.groupsize == -1:
                scale = torch.clamp(self.scale, 1e-6, 1e6)
                weight = (torch.clamp(self.round_ste(self.weight/scale+self.maxq/2), 0, self.maxq) - self.maxq/2) * self.scale 
            else:
                shape = self.weight.shape
                weight = (torch.clamp(self.round_ste(self.weight.reshape([-1,160])/scale+self.maxq/2), 0, self.maxq) - self.maxq/2) * self.scale 
                weight = weight.reshape(shape)
            
            weight = weight * self.a2.repeat_interleave(self.root, dim=-1)  
            if self.training_trans:
                weight = self.Trans(weight, True)
            else:
                weight = Hadamard_trans(weight, dim1= self.transdim1, dim2=self.transdim2, inv = True)
            weight = weight.reshape(self.oc, self.transdim1, self.transdim2)[:,:,:self.expic//self.transdim1]
            weight = weight.reshape(self.oc, self.expic)/self.a1
        return weight

    def miniFunction1(self,x, scale):
        return  x * scale 
    def miniFunction2(self,x, a2):
        return  x * a2.repeat_interleave(self.root, dim=-1)  
    def miniFunction3(self,x, a1):
        return  x.reshape(self.oc, self.expic)/a1

    def get_weight(self):
        if True:
            if self.groupsize == -1:
                if self.packed_flag:
                    weight = (self.unpack_bits_uint8(self.packed_weight).reshape(self.oc, -1)- self.maxq/2)*self.scale
                else:
                    scale = torch.clamp(self.scale, 1e-7, 1e7)
                    weight = (torch.clamp(self.round_ste(self.weight+self.maxq/2), 0, self.maxq) - self.maxq/2) * self.scale 
                    #weight = checkpoint(self.miniFunction1, weight, self.scale)
            else:
                shape = self.weight.shape
                weight = (torch.clamp(self.round_ste(self.weight.reshape([-1,160])/scale+self.maxq/2), 0, self.maxq) - self.maxq/2) * self.scale 
                weight = weight.reshape(shape)
            
            weight = weight * self.a2.repeat_interleave(self.root, dim=-1)   #checkpoint(self.miniFunction2, weight, self.a2)
            if self.training_trans:
                weight = self.Trans(weight, True) #checkpoint(self.Trans, weight, True) # 
            else:
                weight = Hadamard_trans(weight, dim1= self.transdim1, dim2=self.transdim2, inv = True)
            weight = weight.reshape(self.oc, self.transdim1, self.transdim2)[:,:,:self.expic//self.transdim1]
            weight = weight.reshape(self.oc, self.expic)/self.a1#checkpoint(self.miniFunction3, weight, self.a1)
            
        return weight
    
    def get_oriweight(self):
        if True:
            weight = self.weight * self.a2.repeat_interleave(self.root, dim=-1)  
            if self.training_trans:
                weight = self.Trans(weight, True)
            else:
                weight = Hadamard_trans(weight, dim1= self.transdim1, dim2=self.transdim2, inv = True)
            weight = weight.reshape(self.oc, self.transdim1, self.transdim2)[:,:,:self.expic//self.transdim1]
            weight = weight.reshape(self.oc, self.expic)/self.a1
        return weight

    '''def unpack_bits_uint8(self, packed: torch.Tensor):
        bits = torch.stack([(packed >> i) & 1 for i in range(8)], dim=-1)
        return bits.view(packed.shape[0], -1)'''
    def unpack_bits_uint8(self, packed: torch.Tensor):
        # 1. 创建掩码: [1, 2, 4, 8, 16, 32, 64, 128]
        # 这一步可以作为类的常量预先定义好，避免每次调用都创建
        mask = 2 ** torch.arange(8, dtype=packed.dtype, device=packed.device)
        
        # 2. 利用广播机制解包
        # packed.unsqueeze(-1) 形状变为 [N, 1]
        # mask 形状为 [8]
        # 两者位与运算后形状变为 [N, 8]
        return ((packed.unsqueeze(-1) & mask) > 0).to(torch.uint8).view(packed.shape[0], -1)
    def pack_to_int8(self):
        if self.scale.dtype == torch.float16:
            minmin = 1e-4
            maxmax = 1e4
        else:
            minmin = 1e-6
            maxmax = 1e6
        scale = self.clamp_ste(self.scale, minmin, maxmax)
        weight = torch.clamp(self.round_ste(self.weight/scale+self.maxq/2), 0, self.maxq)
        del self.weight
        weight = weight.reshape(-1,  8).to(torch.uint8)
        shifts = (1 << torch.arange(8, dtype=torch.uint8, device=weight.device))
        packed_weight = (weight * shifts).sum(dim=-1).to(torch.uint8)
        self.register_buffer('packed_weight', packed_weight)
        self.packed_flag = True

    def forward(self, x):
        weight = checkpoint(self.get_weight, use_reentrant=False)
        return F.linear(x, weight[:, :self.ic].to(x), self.bias) 

class TmpLinear(nn.Module):
    def __init__(
        self,
        org_module: nn.Linear,
        w_bits,
        expc = '32to16',
        training_trans = False, 
        groupsize = -1, 
        fast_nearest = True
    ):
        super(TmpLinear, self).__init__()
        self.orilinear = org_module
        self.ic = self.orilinear.weight.data.shape[1]
        self.oc = self.orilinear.weight.data.shape[0]
        
        parts = expc.split('to')
        self.root = int(parts[1])
        if self.root == 8:
            if self.ic >10000:
                self.transdim2 = 128
            else:
                self.transdim2 = 64
        #self.transdim2 = round(math.sqrt(self.ic)/self.root)*self.root
        self.transdim1 = math.ceil(self.ic/self.transdim2)
        self.expic = self.transdim1 * self.transdim2
        
        self.orilinear.weight.data = F.pad(self.orilinear.weight.data, (0, self.expic - self.ic), mode="constant", value=0)
        self.a1 = nn.Parameter(torch.ones(self.expic).to(self.orilinear.weight))
        self.a2 = nn.Parameter(torch.ones(self.transdim1, self.transdim2//self.root).to(self.orilinear.weight))
        #self.a3 = nn.Parameter(torch.ones(self.transdim1, self.transdim2//self.root).to(self.orilinear.weight))
        self.fwd_func = F.linear
        self.quantizer = Quantizer()
        self.quantizer.configure(
            w_bits, perchannel=True, sym=True, mse=True
        )

        self.input_trans = False
        self.output_trans = False
       
        self.rotation = None
        if self.orilinear.bias is not None:
            self.bias = self.orilinear.bias
        else:
            self.bias = None
        self.showflag=False

        self.norm = 2
        self.GPTQ = False
        self.training_trans = training_trans
        if self.training_trans:
            if groupsize ==128 :
                #self.Trans = GHalfSVDDecomposeTransMatrix(self.transdim1//2 ,8,20)
                #self.Trans = HalfSVDDecomposeTransMatrix(self.transdim1, self.transdim2, diag_init = True)
                self.Trans = HalfSVDDecomposeTransMatrix(self.transdim1, self.transdim2)
            else:
                self.Trans = HalfSVDDecomposeTransMatrix(self.transdim1, self.transdim2)
        self.groupsize = groupsize

    def round_ste(self, x: torch.Tensor):
        """
        Implement Straight-Through Estimator for rounding operation.
        """
        return (x.round() - x).detach() + x
    def check_nan(self,x,name):
        non_finite_mask = ~torch.isfinite(x)
        indices = torch.nonzero(non_finite_mask)
        if indices.numel() > 0:
            print(f"find nan in {name}")
            print(x.shape)
            #for index in indices:
            #    print(f" {name}- 索引: {index.tolist()}, 值为: {x[tuple(index)]}")
        
    def quantize(self, x, scale, zero, maxq):
        
        if maxq < 0:
            return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
        scale = torch.clamp(scale, 1e-6, 1e6)
        q = torch.clamp(self.round_ste(x / scale + zero) , 0, maxq)
        q = scale * (q - zero)
        
        
        return q
    
    def clamp_ste(self, x: torch.Tensor, min, max):
        return (x.clamp(min,max) - x).detach() + x
    def find_params(self):
        self.a1.data = self.a1.data.reshape(self.transdim1, self.expic//self.transdim1)
        self.weight = self.orilinear.weight.reshape(self.oc, self.transdim1, -1) * self.clamp_ste(self.a1,0.02,50)
        self.weight = self.weight.detach()
        if self.input_trans:
            if self.training_trans:
                self.weight = self.Trans(self.weight)
            else:
                self.weight = Hadamard_trans(self.weight, self.transdim1, self.transdim2)
        
        self.weight = self.weight * self.a2.repeat_interleave(self.root, dim=-1)
        
        if self.groupsize == -1:
            self.quantizer.find_params(self.weight.reshape(self.oc, -1), weight=True)
        elif self.groupsize == 128:
            self.quantizer.find_params(self.weight.reshape(-1, 160), weight=True)
       
    def quant_tmpweight(self):
        #self.check_nan(self.orilinear.weight,'oriweight')
        #print(self.a1.max(),self.a1.min())
        self.weight = self.orilinear.weight.reshape(self.oc, self.transdim1, -1)  * self.clamp_ste(self.a1,0.02,50)
        #self.check_nan(self.weight,'transweight1')
        if self.input_trans:
            if self.training_trans:
                self.weight = self.Trans(self.weight)
            else:
                self.weight = Hadamard_trans(self.weight, self.transdim1, self.transdim2)
        #self.check_nan(self.weight,'transweight2')
        self.weight = self.weight * self.a2.repeat_interleave(self.root, dim=-1)
        #self.check_nan(self.weight,'transweight3')
        if self.showflag:
            print('tmp')
        
        if self.groupsize == -1:
            self.weight = self.weight.reshape(self.oc, -1)
        elif self.groupsize == 128:
            self.weight = self.weight.reshape(-1, 128)

        self.weight = self.quantize(self.weight, self.quantizer.scale*2*F.sigmoid(self.quantizer.alpha), self.quantizer.zero, self.quantizer.maxq)
            

        self.weight = self.weight.reshape(self.oc, self.transdim1, -1)
        
        #self.weight = self.weight * (self.a3 / self.a2).repeat_interleave(self.root, dim=-1)
        self.weight = self.weight / (self.a2).repeat_interleave(self.root, dim=-1)
        if self.showflag:
            print('scale, a2',self.weight.flatten()[:16])
        if self.input_trans:
            if self.training_trans:
                self.weight = self.Trans(self.weight, inv_t=True)
            else:
                self.weight = Hadamard_trans(self.weight, self.transdim1, self.transdim2, inv=True)
        if self.showflag:
            print('hadmard',self.weight.flatten()[:16])
        self.weight = self.weight[:, :, :self.expic//self.transdim1] / self.clamp_ste(self.a1,0.02,50)
        self.weight = self.weight.reshape(self.oc, self.expic)
        if self.showflag:
            print('clip a1',self.weight.flatten()[:16])

    def forward(self, input: torch.Tensor):
        #self.check_nan(input,'input')
        input = self.fwd_func(input, self.weight[:, :self.ic], self.bias) 
        #self.check_nan(input,'output')
        return input



def replace_TmpLinaer_with_FWTLinear(model, args, layers):
    for n,m in model.named_children():
        if isinstance(m, TmpLinear):
            for layer in layers:
                if layer in n:
                    print(n)
                    fwtl = FWTLinear()
                    fwtl.convert_form_tmplinear(m, bits=args.wbits, expc = args.expc, training_trans = args.training_trans, groupsize = args.groupsize, fast_nearest = args.fast_nearest)
                    fwtl.bit_channel_convert()
                    setattr(model, n, fwtl)
        else:
            replace_TmpLinaer_with_FWTLinear(m, args, layers)

def replace_TmpLinaer_with_FWTLinear_mix(model, args, layers, expc_list):
    for n,m in model.named_children():
        if isinstance(m, TmpLinear):
            for layer in layers:
                if layer in n:
                    print(n)
                    if 'q_proj' in n or'k_proj' in n or'v_proj' in n :  
                        expc = expc_list[0]
                    if 'o_proj' in n :  
                        expc = expc_list[1]
                    if 'up_proj' in n or 'gate_proj' in n :  
                        expc = expc_list[2]
                    if 'down_proj' in n :  
                        expc = expc_list[3]
                    fwtl = FWTLinear()
                    fwtl.convert_form_tmplinear(m, bits=args.wbits, expc = expc, training_trans = args.training_trans, groupsize = args.groupsize, fast_nearest = args.fast_nearest)
                    fwtl.bit_channel_convert()
                    setattr(model, n, fwtl)
        else:
            replace_TmpLinaer_with_FWTLinear_mix(m, args, layers, expc_list)


def replace_linear_with_TmpLinear(model, args):
    for n,m in model.named_children():
        if isinstance(m, nn.Linear):
            if 'q_proj' in n or'k_proj' in n or'v_proj' in n or'o_proj' in n or'in_proj_qkv' in n or 'in_proj_z' in n or'out_proj' in n or 'up_proj' in n or 'gate_proj' in n or 'down_proj' in n:
                if 'orilinear' not in n:
                    print(n)
                    setattr(model, n, TmpLinear(m, args.wbits, expc = args.expc, training_trans = args.training_trans, groupsize = args.groupsize, fast_nearest = args.fast_nearest))
        else:
            replace_linear_with_TmpLinear(m, args)


def replace_linear_with_TmpLinear_part(model, args, layers):
    for n,m in model.named_children():
        if isinstance(m, nn.Linear):
            for layer in layers:
                if layer in n:
                    if 'orilinear' not in n:
                        setattr(model, n, TmpLinear(m, 3, expc = args.expc, training_trans = args.training_trans, groupsize = args.groupsize, fast_nearest = args.fast_nearest))
        else:
            replace_linear_with_TmpLinear_part(m, args,layers)


def replace_linear_with_TmpLinear_mix(model, args, expc_list):
    for n,m in model.named_children():
        if isinstance(m, nn.Linear):
            if 'q_proj' in n or'k_proj' in n or'v_proj' in n or'o_proj' in n or 'up_proj' in n or 'gate_proj' in n or 'down_proj' in n:
                if 'orilinear' not in n:
                    if 'q_proj' in n or'k_proj' in n or'v_proj' in n :  
                        expc = expc_list[0]
                    if 'o_proj' in n :  
                        expc = expc_list[1]
                    if 'up_proj' in n or 'gate_proj' in n :  
                        expc = expc_list[2]
                    if 'down_proj' in n :  
                        expc = expc_list[3]
                    setattr(model, n, TmpLinear(m, args.wbits, expc = expc, training_trans = args.training_trans, groupsize = args.groupsize, fast_nearest = args.fast_nearest))
        else:
            replace_linear_with_TmpLinear_mix(m, args, expc_list)

def strtrans(inp):
    if inp =='n':
        return 2
    if inp =='m':
        return 2.3
    if inp =='h':
        return 2.7
    if inp =='p':
        return 3
    if inp =='nt':
        return 1.625
    if inp =='nm':
        return 2.071
    if inp =='nl':
        return 1.875
    if inp =='nh':
        return 2.78
    if inp =='np':
        return 2.25

def get_layer_parameters(model, bitslist):
    pnums = 0
    lbytes = 0
    tmpnums = 0
    for n,m in model.named_modules():
        if isinstance(m, nn.Linear):
            if 'q_proj' in n or'k_proj' in n or'v_proj' in n :
                tmpnums += m.weight.shape[0]*m.weight.shape[1]
    lbytes += tmpnums * strtrans(bitslist[0])
    pnums += tmpnums

    tmpnums = 0
    for n,m in model.named_modules():
        if isinstance(m, nn.Linear):
            if 'o_proj' in n :
                tmpnums += m.weight.shape[0]*m.weight.shape[1]
    lbytes += tmpnums * strtrans(bitslist[1])
    pnums += tmpnums

    tmpnums = 0
    for n,m in model.named_modules():
        if isinstance(m, nn.Linear):
            if 'up_proj' in n or'gate_proj' in n  :
                tmpnums += m.weight.shape[0]*m.weight.shape[1]
    lbytes += tmpnums * strtrans(bitslist[2])
    pnums += tmpnums

    tmpnums = 0
    for n,m in model.named_modules():
        if isinstance(m, nn.Linear):
            if 'down_proj' in n :
                tmpnums += m.weight.shape[0]*m.weight.shape[1]
    lbytes += tmpnums * strtrans(bitslist[3])
    pnums += tmpnums
    return pnums, lbytes