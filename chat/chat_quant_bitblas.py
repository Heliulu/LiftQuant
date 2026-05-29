#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BitBLAS 加速版本的量化模型
关键优化：
1. 使用 BitBLAS 进行 1-bit 矩阵乘法加速
2. M=1 时使用 BitBLAS，M>1 时使用 PyTorch
3. 权重打包为 BitBLAS int1 格式
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

import bitblas

bitblas.set_log_level("WARNING")


BITBLAS_KERNELS = []
KERNEL_CACHE = {}


def pack_weight_int1(weight):
    """
    将 int8 权重矩阵打包为 int1 格式 (并行版本)
    weight: [N, K] 的 int8 张量，值为 -1 或 +1，需要在 GPU 上
    返回: [N, K//8] 的 uint8 张量，在 GPU 上
    """
    N, K = weight.shape
    packed = torch.zeros(N, K // 8, dtype=torch.uint8, device=weight.device)
    
    bit_vals = (weight == 1).to(torch.uint8)
    
    for b in range(8):
        if b < 4:
            k_start = b * 2
        else:
            k_start = (b - 4) * 2 + 1
        
        k_indices = torch.arange(k_start, K, 8, device=weight.device)
        j_indices = k_indices // 8
        
        packed[:, j_indices] |= (bit_vals[:, k_indices] << b)
    
    return packed


def unpack_weight_to_int1(packed):
    """
    将原始打包权重解包为 int1 格式 (-1, +1)
    packed: [packed_shape] 的 uint8 张量
    返回: [N, K] 的 bfloat16 张量，值为 -1 或 +1
    """
    K = packed.shape[1] * 8
    unpacked = torch.bitwise_and(
        packed.unsqueeze(-1).bitwise_right_shift(torch.arange(8, device=packed.device)), 
        1
    ).view(packed.shape[0], K)
    return unpacked.to(torch.bfloat16) * 2 - 1


def get_or_create_kernel(out_features, in_features):
    """获取或创建 BitBLAS kernel"""
    key = (out_features, in_features)
    if key in KERNEL_CACHE:
        return KERNEL_CACHE[key]
    
    matmul_config = bitblas.MatmulConfig(
        M=1, N=out_features, K=in_features,
        A_dtype="float16", W_dtype="int1",
        accum_dtype="float32", out_dtype="bfloat16",
        layout="nt", with_bias=False,
        group_size=None, with_scaling=False,
        with_zeros=False, zeros_mode=None
    )
    # 开启自动调优 (由于 M=1 是静态尺寸，Roller 可以找到针对 decode 阶段最优的 gemv 模板，避免 dynamic 的 NoneType 崩溃)
    kernel = bitblas.Matmul(config=matmul_config, enable_tuning=True)
    kernel_id = register_bitblas_kernel(kernel)
    KERNEL_CACHE[key] = (kernel, kernel_id)
    return kernel, kernel_id


def register_bitblas_kernel(kernel):
    BITBLAS_KERNELS.append(kernel)
    return len(BITBLAS_KERNELS) - 1


@torch.library.custom_op("mylib::bitblas_dispatch", mutates_args=())
def bitblas_dispatch(x: torch.Tensor, w: torch.Tensor, kernel_id: int) -> torch.Tensor:
    return BITBLAS_KERNELS[kernel_id](x, w)


@bitblas_dispatch.register_fake
def _(x, w, kernel_id):
    kernel = BITBLAS_KERNELS[kernel_id]
    out_features = kernel.config.N
    out_shape = list(x.shape)
    out_shape[-1] = out_features
    return x.new_empty(out_shape, dtype=torch.bfloat16)


@bitblas_dispatch.register_kernel("cuda")
def _(x, w, kernel_id):
    return BITBLAS_KERNELS[kernel_id](x, w)


def get_named_linears(module, type):
    return {name: m for name, m in module.named_modules() if isinstance(m, type)}


def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


class FWTLinearBitBLAS(nn.Module):
    """
    使用 BitBLAS 加速的 FWTLinear
    M=1 时使用 BitBLAS，M>1 时使用 PyTorch
    """
    def __init__(self, in_features, out_features, wbits, expc):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.ic = in_features
        self.oc = out_features
        self.bits = wbits
        self.maxq = 1
        self.groupsize = -1
        self.training_trans = True
        self.packed_flag = True
        
        parts = expc.split('to')
        self.root = int(parts[1])
        self.root2 = int(parts[0])
        self.expc = expc
        
        if self.root == 8:
            if self.ic > 10000:
                self.transdim2 = 128
            else:
                self.transdim2 = 64
        else:
            self.transdim2 = 64
        self.transdim1 = math.ceil(self.ic / self.transdim2)
        self.expic = self.transdim1 * self.transdim2
        
        self.scale = nn.Parameter(torch.empty(out_features, 1, dtype=torch.bfloat16, device='meta'))
        self.a1 = nn.Parameter(torch.empty(self.ic, dtype=torch.bfloat16, device='meta'))
        self.a2 = nn.Parameter(torch.empty(self.transdim1 * self.transdim2//self.root * self.root2, dtype=torch.bfloat16, device='meta'))
        self.bias = None
        
        self.w1 = nn.Parameter(torch.empty(self.transdim1, self.transdim1, dtype=torch.bfloat16, device='meta'))
        self.w2 = nn.Parameter(torch.empty(self.transdim2, self.transdim2//self.root*self.root2, dtype=torch.bfloat16, device='meta'))
        
        packed_shape = out_features * (self.root2 * (in_features // self.root)) // 8
        self.register_buffer('packed_weight', torch.empty(packed_shape, dtype=torch.uint8, device='meta'))
        
        self.bitblas_weight = None
        self.pytorch_weight = None
        self.kernel_id = None
        self.transdim3 = self.transdim2 // self.root * self.root2
    
    def convert_weights_for_bitblas(self):
        """将原始权重转换为 BitBLAS 格式"""
        with torch.no_grad():
            packed = self.packed_weight.cuda()
            
            mask = 2 ** torch.arange(8, dtype=packed.dtype, device=packed.device)
            unpacked = ((packed.unsqueeze(-1) & mask) > 0).to(torch.int8)
            
            weight = unpacked.view(self.oc, -1)*2-1
            
            self.bitblas_weight = pack_weight_int1(weight)
            
            del weight, packed, unpacked
            self.packed_weight = self.packed_weight.cpu()
            torch.cuda.empty_cache()
            
            transformed_k = self.expic // self.root * self.root2
            _, self.kernel_id = get_or_create_kernel(self.oc, transformed_k)
            
            self.scale.data = self.scale.data.cuda()
    
    def unpack_bitblas_weight(self):
        """从 bitblas_weight 解包权重（用于 M>1 时的 PyTorch 运算）- 返回 0/1 格式"""
        # 如果 self.bitblas_weight 是 None，说明还没调用 convert_weights_for_bitblas，
        # 则直接使用 self.packed_weight 并解包。
        if getattr(self, "bitblas_weight", None) is None:
            # self.packed_weight 是我们在加载层权重时塞进去的 uint8 数据
            packed = self.packed_weight.view(self.oc, -1).cuda()
        else:
            packed = self.bitblas_weight
            
        N, packed_K = packed.shape
        K = packed_K * 8
        
        mask = 2 ** torch.arange(8, dtype=packed.dtype, device=packed.device)
        unpacked = ((packed.unsqueeze(-1) & mask) > 0).to(torch.int8)
        
        weight = torch.zeros(N, K, dtype=torch.bfloat16, device=packed.device)
        
        for b in range(8):
            if b < 4:
                k_offset = b * 2
            else:
                k_offset = (b - 4) * 2 + 1
            
            k_indices = torch.arange(0, K, 8, device=packed.device) + k_offset
            valid_mask = k_indices < K
            k_valid = k_indices[valid_mask]
            j_valid = k_valid // 8
            
            weight[:, k_valid] = unpacked[:, j_valid, b].to(torch.bfloat16)
        
        
        return weight
    
    def forward(self, x):
        input_dtype = x.dtype
        
        Bsz, lens, dim = x.shape
        
        x = x.view(-1, self.transdim1, self.transdim2)
        
        a1 = self.a1.view(self.transdim1, self.transdim2).to(input_dtype)
        a2 = self.a2.view(self.transdim1, self.transdim3).to(input_dtype)
        
        x_mapped = (self.w1.to(input_dtype) @ ((x * a1) @ self.w2.to(input_dtype))) * a2

        x_mapped = x_mapped.view(Bsz, lens, -1)
        assert torch.isfinite(x_mapped).all(), "X_mapped contains NaN or Inf!"
        if Bsz * lens == 1 and self.kernel_id is not None:  
            # 严格遵循用户最佳数值格式：截断并转 fp16，防止溢出
            x_mapped = torch.clamp(x_mapped, min=-65500.0, max=65500.0).to(torch.float16) 
            
            output = torch.ops.mylib.bitblas_dispatch(x_mapped, self.bitblas_weight, self.kernel_id)
            # output 已经是 bfloat16，但为了保险依然确保 dtype 正确
            output = output.to(input_dtype)
            #assert torch.isfinite(output).all(), "Kernel Output contains NaN or Inf!"
            if not torch.isfinite(output).all():
                has_nan = torch.isnan(output).any().item()
                has_pos_inf = (output == float('inf')).any().item()
                has_neg_inf = (output == float('-inf')).any().item()
                
                error_msg = f"Kernel Output contains invalid values! "
                issues = []
                if has_nan: issues.append("NaN")
                if has_pos_inf: issues.append("+Inf")
                if has_neg_inf: issues.append("-Inf")
                
                error_msg += "Found: " + ", ".join(issues)
                
                # 计算并追加 x_mapped 的统计信息
                x_mapped_abs = x_mapped.abs()
                max_abs = x_mapped_abs.max().item()
                mean_abs = x_mapped_abs.mean().item()
                error_msg += f"\nx_mapped stats -> Max Abs: {max_abs:.4f}, Mean Abs: {mean_abs:.4f}"
                
                # 检查 bitblas_weight 的统计信息
                weight = self.unpack_bitblas_weight().to(input_dtype)*2-1
                weight_abs = weight.abs()
                w_max_abs = weight_abs.max().item()
                w_mean_abs = weight_abs.mean().item()
                error_msg += f"\nbitblas_weight stats -> Max Abs: {w_max_abs:.4f}, Mean Abs: {w_mean_abs:.4f}"
                
                # 保存崩溃现场的输入张量和权重
                import uuid
                dump_id = uuid.uuid4().hex[:8]
                dump_path_x = f"crash_x_mapped_{dump_id}.pt"
                dump_path_w = f"crash_bitblas_weight_{dump_id}.pt"
                torch.save(x_mapped, dump_path_x)
                torch.save(self.bitblas_weight, dump_path_w)
                error_msg += f"\nDumped tensors for reproduction:\n - {dump_path_x}\n - {dump_path_w}"
                
                assert False, error_msg
        else:
            # 兼容 M > 1 的情况，或者 kernel 未初始化时，手动做 F.linear
            weight = self.unpack_bitblas_weight().to(input_dtype)*2-1
            output = F.linear(x_mapped, weight, self.bias)
            if not torch.isfinite(output).all():
                has_nan = torch.isnan(output).any().item()
                has_pos_inf = (output == float('inf')).any().item()
                has_neg_inf = (output == float('-inf')).any().item()
                
                error_msg = f"Pytorch Output contains invalid values! "
                issues = []
                if has_nan: issues.append("NaN")
                if has_pos_inf: issues.append("+Inf")
                if has_neg_inf: issues.append("-Inf")
                
                error_msg += "Found: " + ", ".join(issues)
                assert False, error_msg
        return output * self.scale.view(-1).to(input_dtype) / 2


def create_quantized_model_structure_bitblas(fp_model_path, wbits, expc):
    """
    使用 meta tensor 创建量化模型结构（BitBLAS 版本）
    """
    print(f"Creating quantized model structure (BitBLAS) from {fp_model_path}")
    
    config = AutoConfig.from_pretrained(fp_model_path)
    
    # Qwen3.5 (实际上是 Qwen2.5 架构) 可能在 config 里把关键属性放在 text_config 里
    if hasattr(config, "text_config"):
        for attr in ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "pad_token_id"]:
            if not hasattr(config, attr) and hasattr(config.text_config, attr):
                setattr(config, attr, getattr(config.text_config, attr))
        # fallback for pad_token_id if missing entirely
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = getattr(config.text_config, "eos_token_id", 151645)
            
        # Qwen3.5 模型实例化需要用到 layer_types 数组
        if not hasattr(config, "layer_types") and hasattr(config.text_config, "layer_types"):
            config.layer_types = getattr(config.text_config, "layer_types")
        
        # 还有一些其他的常见 Qwen3.5 属性
        for attr in ["intermediate_size", "max_position_embeddings", "rope_theta", "sliding_window",
                     "linear_num_value_heads", "linear_num_key_heads", "linear_value_head_dim", 
                     "linear_key_head_dim", "linear_conv_kernel_dim", "attn_output_gate", "hidden_act", 
                     "mtp_num_hidden_layers", "rms_norm_eps", "head_dim", "attention_dropout", 
                     "mtp_use_dedicated_embeddings", "attention_bias", "mlp_only_layers", "rope_parameters"]:
            if not hasattr(config, attr) and hasattr(config.text_config, attr):
                setattr(config, attr, getattr(config.text_config, attr))
        
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config=config, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            attn_implementation="eager"
        )
    
    layers = model.model.layers
    
    print("Converting layers to FWTLinearBitBLAS structure...")
    
    # 明确定义被量化的较大 linear 层的名称
    quantized_layer_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj', 'out_proj', 'in_proj_qkv', 'in_proj_z']
    
    for i in tqdm(range(len(layers)), desc="Creating structure"):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        
        for name, module in named_linears.items():
            # 只转换在列表中的层
            should_quantize = False
            for q_name in quantized_layer_names:
                if q_name in name:
                    should_quantize = True
                    break
                    
            if should_quantize:
                fwtlinear = FWTLinearBitBLAS(
                    module.in_features, 
                    module.out_features, 
                    wbits, 
                    expc
                )
                set_op_by_name(layer, name, fwtlinear)
    
    model.tie_weights()
    return model


def load_layer_weights_to_bitblas_model(model, quant_model_path):
    """
    加载分层权重到 BitBLAS 模型
    """
    print(f"Loading quantized weights from {quant_model_path}")
    
    num_layers = len(model.model.layers)
    
    non_layer_path = f'{quant_model_path}-non_layer.pth'
    if os.path.exists(non_layer_path):
        print("Loading non-layer weights...")
        non_layer_state_dict = torch.load(non_layer_path, map_location="cpu")
                
        result = model.load_state_dict(non_layer_state_dict, assign=True, strict=False)
        if result.missing_keys:
            # 精简输出，只提示有几个缺失的键
            print(f"[DEBUG] {len(result.missing_keys)} non-layer keys missing (expected for layer parameters).")
        
    print("Loading layer weights...")
    for i in tqdm(range(num_layers), desc="Loading layers"):
        layer_path = f'{quant_model_path}-layer{i}.pth'
        if not os.path.exists(layer_path):
            print(f"Warning: {layer_path} not found, skipping...")
            continue
        
        state_dict = torch.load(layer_path, map_location="cpu")
        
        new_state_dict = {}
        for key, value in state_dict.items():
           
            if key.endswith('.packed_weight'):
                new_state_dict[key] = value.flatten()
            elif '.a1' in key:
                new_state_dict[key] = 1.0 / value
            elif 'Trans.linear_left' in key:
                
                new_key = key.replace('Trans.linear_left', 'w1')
                new_state_dict[new_key] = value
            elif 'Trans.linear_right' in key:
                new_key = key.replace('Trans.linear_right', 'w2')
                new_state_dict[new_key] = value.T
            else:
                new_state_dict[key] = value

        # 开启 strict=True 确保当前层的每一项都被严格检查
        # 由于 new_state_dict 的 key 带有 'model.layers.i.' 前缀，直接对 model 调用 strict=False 会漏掉检查
        # 我们这里把 key 前缀去掉，直接针对 model.layers[i] 开启 strict=True
        clean_state_dict = {}
        prefix = f"model.layers.{i}."
        for k, v in new_state_dict.items():
            if k.startswith(prefix):
                clean_state_dict[k[len(prefix):]] = v
            else:
                clean_state_dict[k] = v

        try:
            model.model.layers[i].load_state_dict(clean_state_dict, assign=True, strict=True)
        except Exception as e:
            print(f"\n[ERROR] Layer {i} failed to load with strict=True!")
            print(e)
            raise e
    
    # 获取所有非层权重的 key，以便后续排查哪些没被覆盖
    non_layer_keys_expected = []
    for name, _ in model.named_parameters():
        if "model.layers" not in name:
            non_layer_keys_expected.append(name)
            
    print("\n[DEBUG] Expected non-layer parameters:")
    for name in non_layer_keys_expected:
        print(f"  - {name}")
    
    # [撤销暴力 to_empty] 不要掩盖问题，保留真正的 meta 状态让程序报错，
    # 或者由外部脚本去精确加载 non_layer 权重
    # model.to_empty(device="cpu") 
    
    # Debug: 检查是否还有任何 meta tensor 残留
    meta_tensors = []
    for name, param in model.named_parameters():
        if param.is_meta:
            meta_tensors.append(name)
    if meta_tensors:
        print("\n[DEBUG] 仍然处于 meta 状态的参数：")
        for name in meta_tensors:
            print(f"  - {name}")
    else:
        print("\n[DEBUG] 没有任何参数处于 meta 状态！")
        
    print("Loaded quantized weights successfully!")
    return model


def convert_model_to_bitblas(model):
    """
    将模型的所有 FWTLinearBitBLAS 层转换为 BitBLAS 格式
    """
    print("Converting weights to BitBLAS format...")
    
    # 这个函数现在接收的是一个已经在 GPU 上的 model
    # 不能把它切回 CPU，因为后续的推理和编译都需要在 GPU 上
    
    for name, module in model.named_modules():
        if isinstance(module, FWTLinearBitBLAS):
            # 将该层内部缓存的 CPU 数据推向 GPU，并调用底层转换
            module.convert_weights_for_bitblas()
            
    print("BitBLAS conversion completed!")
    return model


def save_bitblas_model(model, save_path):
    """
    保存 BitBLAS 转换后的模型
    """
    print(f"Saving BitBLAS model to {save_path}...")
    
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, FWTLinearBitBLAS):
            state_dict[f"{name}.scale"] = module.scale.data.cpu()
            state_dict[f"{name}.a1"] = module.a1.data.cpu()
            state_dict[f"{name}.a2"] = module.a2.data.cpu()
            state_dict[f"{name}.w1"] = module.w1.data.cpu()
            state_dict[f"{name}.w2"] = module.w2.data.cpu()
            state_dict[f"{name}.bitblas_weight"] = module.bitblas_weight.cpu()
            state_dict[f"{name}.kernel_id"] = torch.tensor([module.kernel_id if module.kernel_id is not None else -1])
            state_dict[f"{name}.in_features"] = torch.tensor([module.in_features])
            state_dict[f"{name}.out_features"] = torch.tensor([module.out_features])
    
    non_layer_state = {}
    for name, param in model.named_parameters():
        if not any(x in name for x in ['FWTLinearBitBLAS', '.scale', '.a1', '.a2', '.w1', '.w2']):
            non_layer_state[name] = param.data.cpu()
    for name, buf in model.named_buffers():
        if not any(x in name for x in ['FWTLinearBitBLAS', '.bitblas_weight', '.packed_weight']):
            non_layer_state[name] = buf.cpu()
    
    state_dict.update(non_layer_state)
    
    torch.save(state_dict, save_path)
    print(f"Model saved to {save_path}")
    return save_path


def load_bitblas_model(fp_model_path, bitblas_model_path, wbits, expc):
    """
    直接加载已转换的 BitBLAS 模型
    """
    print(f"Loading BitBLAS model from {bitblas_model_path}...")
    
    # 加载空模型 (bfloat16)
    model = create_quantized_model_structure_bitblas(fp_model_path, wbits, expc)
    
    print(f"Loading weights from {bitblas_model_path}")
    state_dict = torch.load(bitblas_model_path, map_location='cpu', weights_only=True)
    
    for name, module in model.named_modules():
        if isinstance(module, FWTLinearBitBLAS):
            if hasattr(module, 'packed_weight'):
                delattr(module, 'packed_weight')
    
    new_state_dict = {}
    kernel_ids = {}
    
    for key, value in state_dict.items():
        if key.endswith('.kernel_id'):
            layer_name = key.rsplit('.kernel_id', 1)[0]
            kernel_ids[layer_name] = value.item()
            continue
        if key.endswith('.in_features') or key.endswith('.out_features'):
            continue
        if key.endswith('.bitblas_weight'):
            continue
        new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict, assign=True, strict=False)
    
    model = model.cuda()
    
    for name, module in model.named_modules():
        if isinstance(module, FWTLinearBitBLAS):
            prefix = f"{name}."
            module.bitblas_weight = state_dict[f"{prefix}bitblas_weight"].cuda()
            
            kernel_id = kernel_ids.get(name, -1)
            if kernel_id >= 0:
                transformed_k = module.expic // module.root * module.root2
                _, module.kernel_id = get_or_create_kernel(module.oc, transformed_k)
            else:
                module.kernel_id = None
    
    model.eval()
    print("BitBLAS model loaded successfully!")
    return model


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='测试 BitBLAS 加速版本')
    parser.add_argument('--fp_model_path', type=str, 
                        default='/mnt/bn/adsinfra-gpu-dev-hl/heliulu/LLMs/Qwen3.5-27B')
    parser.add_argument('--quant_model_path', type=str,
                        default='/mnt/bn/adsinfra-gpu-dev-hl/heliulu/qmodels/LiftQuant/24to8f1/Qwen3.5-27B/Qwen3.5-27B+24to8')
    parser.add_argument('--wbits', type=int, default=3)
    parser.add_argument('--expc', type=str, default='24to8')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("测试 BitBLAS 加速版本")
    print("=" * 60)
    
    start = datetime.now()
    model = create_quantized_model_structure_bitblas(
        args.fp_model_path, 
        args.wbits, 
        args.expc
    )
    model = load_layer_weights_to_bitblas_model(model, args.quant_model_path)
    model = model.cuda()
    model = convert_model_to_bitblas(model)
    end = datetime.now()
    
    print(f"\n模型加载和转换耗时: {end - start}")