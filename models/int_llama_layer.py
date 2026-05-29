import torch
from torch import nn
from typing import Optional, Tuple, List
from quantize.int_linear import QuantLinear, PCAQuant
from quantize.int_matmul import QuantMatMul, QuantPCAMatMul
import torch.nn.functional as F

from collections import OrderedDict
import math
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding,apply_rotary_pos_emb,LlamaRMSNorm,repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.activations import ACT2FN
import pdb
import copy

from quantize.quantizer import ActivationQuantizer
from trans_utils import Hadamard_trans

class LlamaRMSNorm(nn.Module):
    def __init__(self, ori_norm, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.register_buffer('weight',ori_norm.weight)
        self.bias = None
        self.variance_epsilon = eps
        self.use_temporary_parameter = False


    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias if hasattr(self, 'bias') else None

        return (weight * hidden_states+bias).to(input_dtype) if bias is not None else (weight * hidden_states).to(input_dtype)


class detector(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
    

class QuantLlamaMLP(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        args=None,
    ):
        super().__init__()
        
        self.gate_proj = org_module.gate_proj
        self.up_proj = org_module.up_proj
        self.down_proj = org_module.down_proj
        
        self.act_fn = ACT2FN[hidden_act]
        n = self.down_proj.weight.shape[-1]
        transdim = n & -n
        self.Rotation4_enable = False
        #print('init Rotation4')
        self.detector6 = detector()
        self.detector7 = detector()
        self.detector8 = detector()
        
        L1 = self.gate_proj.weight.data.shape[-1]
        L2 = self.down_proj.weight.data.shape[-1]
        device = 'cuda'
        self.args=args
        dtype = self.args.dtype
        self.qbias3 = nn.Parameter(torch.zeros(L1,device=device,dtype=dtype))
        self.qbias4 = nn.Parameter(torch.zeros(L2,device=device,dtype=dtype))
        self.down_scale = nn.Parameter(torch.ones(L2,device=device,dtype=dtype)*1.)
        self.register_buffer('asyscale4', torch.zeros(L2,device=device,dtype=dtype))
        self.quantizer3 = ActivationQuantizer(**args.act_quant_params)
        self.quantizer4 = ActivationQuantizer(**args.act_quant_params)
        self.asyscale4_en = False
        self.down_scale_en = False
        self.use_act_quant = False
        self.down_quant_enable = False
        self.quant_mask = args.quant_mask
    def forward(self, x):

        ##########act quantize3
        x = x - self.qbias3
        if self.use_act_quant:
            if self.quant_mask[2]=='1':
                x = self.quantizer3(x)
        ##########
         
        x = x + self.qbias3
        x = self.act_fn(self.gate_proj(x)) * (self.up_proj(x))
        
        if self.down_scale_en:
            x = x * self.down_scale
        if self.Rotation4_enable:
            x = Hadamard_trans(x)

        ##########act quantize4
        if self.asyscale4_en:
            x = x * self.asyscale4
        x = x - self.qbias4
        if self.use_act_quant or self.down_quant_enable:
            if self.quant_mask[3]=='1':
                x = self.quantizer4(x)
        
        ##########
        x = x + self.qbias4
        x = self.down_proj(x) 
       
        return x


class QuantLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, 
                 org_module: nn.Module,
                 config: LlamaConfig,
                 args=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.layer_idx = org_module.layer_idx
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.rotary_emb = copy.deepcopy(org_module.rotary_emb)
        
        self.k_proj=org_module.k_proj
        
        self.v_proj=org_module.v_proj

        self.args=args
        self.q_proj=org_module.q_proj
        self.o_proj=org_module.o_proj
        

        
        self.qkt_matmul = QuantPCAMatMul(
            args.q_quant_params, args.R2_quant_params, matmul_func=torch.matmul, heads = args.heads, head_dim= args.head_dim, trans_dim=args.head_dim
        )
        #print('init qkt_matmul')
        self.pv_matmul = QuantPCAMatMul(
            args.p_quant_params, copy.deepcopy(args.R3_quant_params), matmul_func=torch.matmul, out_quant=True, heads= args.heads, head_dim= args.head_dim, trans_dim=args.head_dim
        )
        #print('init pv_matmul')
        self.use_weight_quant = False
        self.use_act_quant = False
        
        self.detector1 = detector()
        

        L1 = self.v_proj.weight.data.shape[-1]
        L2 = self.o_proj.weight.data.shape[-1]
        device = 'cuda'
        dtype = self.args.dtype
        self.qbias1 = nn.Parameter(torch.zeros(L1, device=device, dtype=dtype))
        self.qbias2 = nn.Parameter(torch.zeros(L2, device=device, dtype=dtype))
        self.register_buffer('asyscale3', torch.zeros(L2,device=device,dtype=dtype))
        self.asyscale3_en = False
        self.quantizer1 = ActivationQuantizer(**args.act_quant_params)
        self.quantizer2 = ActivationQuantizer(**args.act_quant_params)
        self.quant_mask = args.quant_mask
        self.heads = args.heads

        self.register_buffer('o_scale',torch.ones(L2, device=device, dtype=dtype)*1.)
        self.o_scale2 = nn.Parameter(torch.ones(L2, device=device, dtype=dtype)*1.)
       

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        '''
        if past_key_value is not None:    
            
            hidden_states = hidden_states[:,-1,:].unsqueeze(1)
            print('KV cache is',past_key_value[0].shape,hidden_states.shape)
        else:
            print('None KV cache')'''
        
        bsz, q_len, _ = hidden_states.size()
        ##########act quantize1
        hidden_states = hidden_states - self.qbias1
        if self.use_act_quant:
            if self.quant_mask[0]=='1':
                hidden_states = self.quantizer1(hidden_states)
        
        ##########
        hidden_states = hidden_states + self.qbias1
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states) 
        #

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        if position_embeddings is None:
            
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if True:
            key_states = self.qkt_matmul.quant_x2(key_states)
            attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
            
            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            value_states = self.pv_matmul.quant_x2(value_states)
            attn_output = self.pv_matmul(attn_weights, value_states)

        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        ##########act quantize2
        
        if self.o_scale is not None:
            attn_output = attn_output*self.o_scale
        attn_output = attn_output*self.o_scale2
        


        if self.asyscale3_en:
            attn_output = attn_output*self.asyscale3

        attn_output = attn_output - self.qbias2
        if self.use_act_quant:
            if self.quant_mask[1]=='1':
                attn_output = self.quantizer2(attn_output)
        
        ##########
        
        attn_output = attn_output + self.qbias2
        attn_output = self.o_proj(attn_output) 
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):

        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul, QuantPCAMatMul)):
                m.set_quant_state(weight_quant, act_quant)
                


class QuantLlamaDecoderLayer(nn.Module):
    def __init__(self, 
                 config: LlamaConfig,
                 ori_layer,
                 args):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QuantLlamaAttention(
            org_module=ori_layer.self_attn,
            config=config,
            args=args
            )
        self.mlp = QuantLlamaMLP(
            org_module=ori_layer.mlp,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            args=args,
        )
        self.input_layernorm = LlamaRMSNorm(ori_layer.input_layernorm,eps=ori_layer.input_layernorm.variance_epsilon)
        self.post_attention_layernorm = LlamaRMSNorm(ori_layer.post_attention_layernorm,eps=ori_layer.post_attention_layernorm.variance_epsilon)
      

        self.pre_Rotation1_enable = False
        self.post_Rotation1_enable = False
        self.pre_PCA_enable = False
        self.post_PCA_enable = False
        self.detector0 = detector()
        self.detector9 = detector()
        self.args = args
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
       
        
        if self.pre_PCA_enable:
            hidden_states = (hidden_states.double() @ self.P.double()).to(self.args.dtype) 
      
        if self.pre_Rotation1_enable:
            hidden_states = Hadamard_trans(hidden_states)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        self.detector0(hidden_states)
        

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        
        hidden_states = self.mlp(hidden_states)
        hidden_states = (residual) + hidden_states
        
        if self.post_Rotation1_enable:
            hidden_states = Hadamard_trans(hidden_states,inv=True)
        if self.post_PCA_enable:
            hidden_states = (hidden_states.double()@self.P.T.double()).to(self.args.dtype)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
            
        return outputs        

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        #print('hello')
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul, QuantPCAMatMul)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant)

    def clear_temp_variable(self):
       for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias
