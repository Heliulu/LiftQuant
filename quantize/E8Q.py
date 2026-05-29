import torch.nn as nn
import torch
from trans_utils import Hadamard_trans
from quantize.int_linear import *
from gptq.quant import Quantizer

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import os 
import math
from torch.autograd import Function

def clamp_to_ball(x, R):
    # x: [...,8]
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # factor = min(1, R/||x||)
    factor = torch.minimum(torch.ones_like(norm), R / norm)
    return x * factor   


class E8FastSTE(Function):
    @staticmethod
    def forward(ctx, x):
        """
        x: [...,8]
        返回最近的 E8 点 v 或 0
        """
        # 绝对值和符号
        a = x.abs()                        # [...,8]
        sgn = x.sign()                     # [...,8]

        # ---- 类型1: 取前 two max abs dims ----
        # top2 及其索引
        top2_vals, top2_idx = a.topk(2, dim=-1)  # top2_vals: [...,2], top2_idx:[...,2]
        s1 = top2_vals.sum(dim=-1)               # [...], 候选得分

        # ---- 类型2: 全部符号 /2 的情况（偶数正号）或反转最小那维 ----
        total = a.sum(dim=-1)                    # [...], ∑|xₖ|
        # 先当作偶数正号
        s2_even = total * 0.5                    # [...], ∑|xₖ|/2
        # 统计正号个数的奇偶性
        pos_count = (sgn > 0).sum(dim=-1)         # [...]
        is_odd   = (pos_count & 1) == 1           # [...]，True 表示需要翻转
        # 翻转后得分 = ∑|x|/2 - min|xₖ|
        s2_odd  = s2_even - a.min(dim=-1).values # [...]
        # 最终类型2得分
        s2 = torch.where(is_odd, s2_odd, s2_even) # [...]

        # ---- 选最佳邻居得分 ----
        s12 = torch.max(s1, s2)                  # […] 得分

        # ---- 和原点比较: s12>1 则选邻居，否则选 0 ----
        use_nb = s12 > 1.0                       # bool mask [...]

        # ---- 构造输出 v ----
        # v0 全 0
        v0 = torch.zeros_like(x)

        # v1 类型1 的构造 v
        # 把 top2_idx 拆成 i,j
        i = top2_idx[...,0]
        j = top2_idx[...,1]
        # 构造全 0，再在 i,j 位置上 scatter sign
        v1 = torch.zeros_like(x)
        # 先放第一个维度
        idx_i = i.unsqueeze(-1)
        vi   = sgn.gather(-1, idx_i)
        v1.scatter_(-1, idx_i, vi)
        # 再放第二个维度
        idx_j = j.unsqueeze(-1)
        vj   = sgn.gather(-1, idx_j)
        v1.scatter_(-1, idx_j, vj)

        # v2 类型2 的构造
        # 全部都是 sign(x)/2
        v2 = sgn * 0.5
        if is_odd.any():
            m = a.argmin(dim=-1)  
            flip_mask = is_odd
            # 只在is_odd的batch里翻转
            flip_idx = m[flip_mask]
            v2[flip_mask, flip_idx] = -sgn[flip_mask, flip_idx] * 0.5

        # 依据 s1>=s2 先挑类型1/2，再和原点 pick
        pick12 = s1 >= s2                       # […]
        v12    = torch.where(
            pick12.unsqueeze(-1), v1, v2)       # [...,8]

        v = torch.where(use_nb.unsqueeze(-1), v12, v0)

        return v

    @staticmethod
    def backward(ctx, grad_v):
        # STE: 直接恒等回传
        return grad_v

def e8_quant_ste(x):
    # x: [...,8]
    return E8FastSTE.apply(x)


x = torch.randn(32,8)
x = clamp_to_ball(x, math.sqrt(2))
y = x + (e8_quant_ste(x) - x).detach() 