from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map
import torch
from tqdm import tqdm
import gc  
from quantize.tmplinear import TmpLinear, FWTLinear

def get_named_linears(module, type):
    # return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}
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

def check_meta_tensors(model, context: str = "当前状态"):
    """
    遍历一个 PyTorch 模型的所有参数 (parameters) 和缓冲区 (buffers)，
    检查是否有任何张量位于 'meta' 设备上，并打印详细信息。

    参数:
    - model (nn.Module):需要检查的 PyTorch 模型。
    - context (str): 一个描述性字符串，用于说明当前是在哪个代码阶段进行检查。
    """
    print(f"\n--- 检查模型中的 Meta 张量 ({context}) ---")
    
    found_meta_tensor = False

    # 1. 检查模型参数 (Parameters)
    for name, param in model.named_parameters():
        print(name, param.shape, param.dtype)
        if param.device.type == 'meta':
            print(f"[参数 - Meta] 位于 'meta' 设备: {name}")
            found_meta_tensor = True

    # 2. 检查模型缓冲区 (Buffers)
    # 缓冲区通常用于存储非训练参数，比如 BatchNorm 的 running_mean
    for name, buf in model.named_buffers():
        print(name, param.shape, param.dtype)
        if buf.device.type == 'meta':
            print(f"[缓冲区 - Meta] 位于 'meta' 设备: {name}")
            found_meta_tensor = True

    if not found_meta_tensor:
        print(">>> 结论: 所有参数和缓冲区都在具体的物理设备上 (非 'meta' 设备)。模型已正确具象化。")
    else:
        print(">>> 结论: 发现 'meta' 张量！模型尚未完全加载权重或未正确移动到设备上。")
        
    print(f"--- 检查完成 ({context}) ---\n")

    return found_meta_tensor

def load_quantized_model(fp_model_path, quant_model_path, wbits, expc, w_ternary, load_per_layer, auto_mix_precision = False):
    print(f"Loading quantized model from {fp_model_path}")

    # import pdb;pdb.set_trace()
    tokenizer = AutoTokenizer.from_pretrained(fp_model_path, use_fast=False)
    config = AutoConfig.from_pretrained(fp_model_path)
    with init_empty_weights(): # 生成空的占位模型
        model = AutoModelForCausalLM.from_config(config=config,torch_dtype=torch.float16, trust_remote_code=True)
    #if load_per_layer:
    #    # 加载模型的非layer权重
    #    non_layer_state_dict = torch.load(quant_model_path+'-non_layer.pth', map_location="cpu")
    #    model.load_state_dict(non_layer_state_dict, strict=False)
    #    print(non_layer_state_dict)
    #    print(model.model.norm.weight)
    #    model.model.norm = model.model.norm.to('cpu')
    #    print(model.model.norm.weight)
    layers = model.model.layers
    expc_choice = None
    if auto_mix_precision == True:
        if 'llama-3' in fp_model_path.lower():
            expc_choice = [2, 2, 1, 2, 3, 3, 1, 3, 3, 2, 1, 2, 3, 3, 1, 2, 3, 3, 1, 2, 3, 3,
                                    1, 2, 2, 3, 1, 2, 2, 3, 1, 1, 1, 3, 1, 1, 2, 3, 1, 1, 1, 2, 1, 1,
                                    1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2,
                                    1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                                    1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2,
                                    1, 2, 2, 2, 1, 2, 3, 2, 1, 2, 3, 3, 2, 3, 3, 3, 3, 3]
        else:
            expc_choice = [0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2,
       0, 2, 0, 2, 0, 2, 0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 0, 1, 1, 2,
       0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 0, 1,
       1, 2, 1, 2, 2, 2, 0, 2, 2, 2, 0, 1, 2, 2, 1, 1, 2, 2]
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():

            with torch.no_grad():
                if expc_choice!= None:
                    if "q_proj" in name or "q_proj" in name  or "q_proj" in name:
                        expc = expc_choice[i*4]
                    if "o_proj" in name:
                        expc = expc_choice[i*4+1]
                    if "gate_proj" in name or "up_proj" in name :
                        expc = expc_choice[i*4+2]
                    if "down_proj" in name:
                        expc = expc_choice[i*4+3]
                    if expc ==0:
                        expc = 'nl'
                    if expc ==1:
                        expc = 'nm'
                    if expc ==2:
                        expc = 'np'
                    if expc ==3:
                        expc = 'nh'
                fake_linear = torch.nn.Linear(module.in_features,module.out_features,not module.bias is None, device = 'cuda', dtype = torch.float16) 
                #convert to tmplinear
                tmplinear = TmpLinear(fake_linear, wbits, expc = expc, training_trans = True)
                tmplinear.find_params()
                tmplinear.quantizer.alpha = torch.nn.Parameter(0.*torch.ones(tmplinear.quantizer.scale.shape, device = tmplinear.orilinear.weight.device , dtype = tmplinear.orilinear.weight.dtype ))
                
                # conert to adalinear
                '''adalinear= AdaLinear()
                adalinear.convert_form_tmplinear(tmplinear, maxq=2, expc=expc, training_trans =True)
                del tmplinear, fake_linear
                # convert to fuse buffer
                adalinear.remove_adaquant()
                adalinear = adalinear.to('cpu')
                set_op_by_name(layer, name, adalinear)'''

                # convert to FWTLinear
                fwtlinear =  FWTLinear()
                #fwtlinear.convert_form_tmplinear(tmplinear, expc = expc, training_trans = True, groupsize=-1 )
                fwtlinear.convert_form_tmplinear(tmplinear, bits=wbits, expc=expc, training_trans = True, groupsize = -1)
                fwtlinear.bit_channel_convert(True)
                fwtlinear.pack_to_int8()
                fwtlinear = fwtlinear.to('cpu')
                set_op_by_name(layer, name, fwtlinear)
                #if load_per_layer:
                #    #model.model.layers[i]
                #    layer.load_state_dict(torch.load(quant_model_path+'-layer'+str(i)+'.pth', map_location="cpu"), strict=False)

        #print(model.model.layers[0].mlp.up_proj.Trans.linear_u_left.dtype)
    #print(model.model.layers[0].mlp.up_proj.Trans.linear_u_left.dtype)
    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()
    
    device_map = infer_auto_device_map(model)
    #print(model.model.layers[0].mlp.up_proj.Trans.linear_u_left.dtype)
    print("Loading pre-computed quantized weights...")
    
    if load_per_layer:
        #1+1
        # 加载模型的非layer权重
        non_layer_state_dict = torch.load(quant_model_path+'-non_layer.pth', map_location="cpu")
        model.load_state_dict(non_layer_state_dict, assign=True, strict=False)
        for i in range(len(model.model.layers)):
            layer_path = f'{quant_model_path}-layer{i}.pth'
            state_dict = torch.load(layer_path, map_location="cpu")
            for key in list(state_dict.keys()):
                if key.endswith('.packed_weight'):
                    state_dict[key] = state_dict[key].flatten()

            model.model.layers[i].load_state_dict(state_dict, assign=True, strict=False)
    else: #分支2
        state_dict = torch.load(quant_model_path, map_location='cpu')
        '''for param_name, tensor in state_dict.items():
            print(f"参数名称 (Key): {param_name}")
            print(f"  - 形状 (Shape): {tensor.shape}")
            print(f"  - 数据类型 (Dtype): {tensor.dtype}")
            print(f"  - 所在设备 (Device): {tensor.device}")
            print("-" * 30)'''
        for key in list(state_dict.keys()):
            if key.endswith('.packed_weight'):
                state_dict[key] = state_dict[key].flatten()
        model.load_state_dict(state_dict, assign=True, strict=False)

    #check_meta_tensors(model)


    model = dispatch_model(model, device_map=device_map)
    print("Loaded quantized weights successfully.")

    model = model.to(torch.float)
    #load_checkpoint_in_model(model,checkpoint=model_path,device_map=device_map,offload_state_dict=True)
    #print("Loading pre-computed quantized weights Successfully")
    #print(model.model.layers[0].mlp.up_proj.Trans.linear_u_left.dtype)
    return model, tokenizer