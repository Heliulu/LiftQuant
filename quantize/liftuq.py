import torch
import torch.nn as nn
import torch.nn.functional as F
from models.int_llama_layer import QuantLlamaDecoderLayer
from quantize.tmplinear import *
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from quantize.utils import  get_parameters, get_act_means


from tqdm import tqdm

import functools
from scipy import linalg

from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
import os


from scipy import linalg

# GPTQ
from gptq.gptq import *
from gptq.modelutils import *
from gptq.quant import *

from trans_utils import Hadamard_trans, ORTransMatrix, pca_cov, PCA_rotation
    
 
import torch.nn.functional as F

def get_n_set_parameters_byname(model, required_names):
    params = []
    for r_name in required_names:
        for name, param in model.named_parameters():
            if name.find(r_name) > -1:
                params.append(param)
    for param in params:
        param.requires_grad = True
    return params

def get_n_set_parameters_byname_FWT(model, required_names):
    params = []
    for r_name in required_names:
        for n,m in model.named_modules():
            if isinstance(m, FWTLinear):
                for name, param in m.named_parameters():
                    if name.find(r_name) > -1:
                        params.append(param)
    for param in params:
        param.requires_grad = True
    return params

def print_trainable_parameters(model):      
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    print('trainable module')
    print('*'*80)
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            print(name, "is trainable")
            trainable_params += param.numel()
    print('*'*80)
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def liftuq(
    lm,
    args,
    dataloader,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if args.info:
        print(args)
        print(model)
        print(type(model))

    if "llama" in args.net.lower() or "qwen" in args.net.lower(): 
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for qwen2.5, llama-2, Llama-3/3.1/3.2 now")
    
    
    
    if args.save_dir and args.save_per_layer :
        # 如果目录不存在则创建
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir,args.net, args.net+'+'+args.expc+'-non_layer.pth')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        non_layer_state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith("model.layers.")}
        torch.save(non_layer_state_dict, save_path)
        print('save non-layer-statedict')
    
    args.quant_end = min(args.quant_end, len(layers))
    for i in range(len(layers)):
        if i >= args.quant_end:
            layers[i] = None 
        gc.collect()
    layers[0] = layers[0].to(dev)
    print(layers[0])
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
  
    if args.deactive_amp and args.epochs1>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = args.dtype
        traincast = torch.amp.autocast
    
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device='cpu'
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.to('cpu')
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_embeddings"] = kwargs["position_embeddings"]
            
            raise ValueError

    
    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    # move embedding layer and first layer to cpu
    # print(cache["position_embeddings"] )
    layers[0] = layers[0].module 
    layers[0] = layers[0].cpu() 
    
    

    if "llama" in args.net.lower() or "qwen" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    else:
        raise ValueError("Only support for qwen2.5, llama-2, Llama-3/3.1/3.2 now")
    torch.cuda.empty_cache()
    
    # same input of first layer for fp model and quant model
    
    inps = inps[:args.nsamples].to('cpu')
    quant_inps = inps
    # take output of fp model as input
    fp_outs = copy.deepcopy(inps)
    
        
    
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    
    position_embeddings = cache["position_embeddings"]
    
    


        
        
        
    #### Fuse parameters of RMSNorm and Rotation, abtain new model arch 
    if 'qwen3.' not in args.net.lower():
        logger.info(f"=== Start fuse nrom layers ===")
        for i in tqdm(range(args.quant_end)):
            layer = layers[i].to(dev)
            
            for n,m in layer.named_modules():
                if 'input_layernorm' in n:
                    for name, module in  layer.named_modules():
                        if (isinstance(module, nn.Linear)) and( ('q_proj' in name) or ('k_proj' in name) or ('v_proj' in name)or ('in_proj_qkv' in name)or ('in_proj_z' in name) or ('in_proj_a' in name) or ('in_proj_b' in name)):
                            module.weight.data = module.weight.data * m.weight
                            
                    m.weight.data=torch.ones(m.weight.shape).to(dev).to(dtype)
                    
                if 'post_attention_layernorm' in n:
                    for name, module in  layer.named_modules():
                        if (isinstance(module, nn.Linear)) and( ('up_proj' in name) or ('gate_proj' in name)):
                            module.weight.data = module.weight.data * m.weight
                            
                    m.weight.data=torch.ones(m.weight.shape).to(dev).to(dtype)
            #if "llama" in args.net.lower() or "qwen" in args.net.lower():  
            #    qlayer = DecoderLayer(lm.model.config, layer, args) 
            #qlayer = qlayer.to(dev).to(dtype)
            qlayer = layer.to(dev).to(dtype)

            layers[i] = qlayer.to("cpu")
            del qlayer
            del layer
    

    
    fp_outs = fp_outs.to('cpu')
    
    quant_inps = quant_inps.to('cpu')

    if args.auto_mix_precision:
        modelpnums = 0
        modelbytes = 0
        '''
        fp_inps = fp_outs.to('cpu')[:256].clone()
        layer_alpha = torch.ones(len(layers))
        # each layer output give a 0.01 energy noisy, 0.01 std.
        
        test_inpts = fp_outs.to('cpu')[:16].clone()
        test_inpts = test_inpts.repeat((len(layers)+1,1,1))
        for i in tqdm(range(len(layers))):
            layer = layers[i].to(dev)
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    for j in range(test_inpts.shape[0]//args.batch_size):
                        index = j * args.batch_size
                        test_inpts[index:index+args.batch_size,] = layer(test_inpts[index:index+args.batch_size,].to(dev), attention_mask=attention_mask,position_ids=position_ids)[0].to('cpu')
                test_inpts[(i+1)*16:(i+2)*16] += 0.01*torch.randn(test_inpts[(i+1)*16:(i+2)*16].shape, dtype = test_inpts.dtype)
                #print(test_inpts[0])
                #print(test_inpts[16])
                #print(test_inpts[(i+1)*16])
        energy = test_inpts[:16].pow(2).mean()
        print(energy)
        for i in range(len(layers)):
            layer_alpha[i] = (test_inpts[(i+1)*16:(i+2)*16] - test_inpts[:16]).pow(2).mean()/0.0001
        print(layer_alpha)'''
    layer_alpha = [491.8, 241.0, 114.8, 67.0, 42.2, 31.9, 20.6, 12.7, 10.0,
                   8.5, 7.8, 7.0,    6.5,   6.2,  6.0,  5.8,  5.4, 5.24, 
                   4.86, 4.47, 4.08, 3.72, 3.34, 3.03, 2.74, 2.47, 2.21,
                   1.97, 1.75, 1.53, 1.34, 1.]
    final_loss_list = []
    ########### 
   
    for i in range(args.quant_end):
        #i=4
        logger.info(f"=== Start quantize layer {i} ===")
        qlayer = layers[i].to(dev)
        #if i==27:
        #    qlayer.to(float)
        if 'moe' in args.net.lower():
            act_disturb = get_act_means(qlayer, fp_outs, 32, 4,['q_proj', 'o_proj', 'experts.0.up_proj', 'experts.1.up_proj'],attention_mask=attention_mask,position_embeddings=position_embeddings)
        else:
            if any(name.endswith('q_proj') for name, _ in qlayer.named_modules()):
                act_disturb = get_act_means(qlayer, fp_outs, 8, 4,['q_proj', 'o_proj', 'up_proj', 'down_proj'],attention_mask=attention_mask,position_embeddings=position_embeddings)
            else:
                act_disturb = get_act_means(qlayer, fp_outs, 8, 4,['in_proj_qkv', 'out_proj', 'up_proj', 'down_proj'],attention_mask=attention_mask,position_embeddings=position_embeddings)
            
        if args.auto_mix_precision:
            fp_inps = fp_outs.to('cpu')[:256].clone()
        if args.epochs1 > 0:
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=args.dtype):
                    batch_size = args.batch_size * 2
                    #args.batch_size = 2
                    for j in tqdm(range(args.nsamples//batch_size)):
                        index = j * batch_size
                        fp_outs[index:index+batch_size,] = qlayer(fp_outs[index:index+batch_size,].to(dev), attention_mask=attention_mask,position_embeddings=position_embeddings).to('cpu').to(dtype)
                        
        logger.info(f"=== Prepared quantize layer {i} ===")
        for m in qlayer.modules():
            if type(m) == nn.Linear:
                m.weight.requires_grad_(False)

        if True:
            
            if args.auto_mix_precision:
                #final_loss_list = [0.0001, 9.5841e-05, 0.0003, 0.0007, 0.1009, 0.0092, 0.0181, 0.0072, 0.0041, 0.0010, 0.0025, 0.0023, 0.0052, 0.0009, 0.0032, 0.0026, 0.0038, 0.0011, 0.0035, 0.0030, 0.0032, 0.0014, 0.0041, 0.0034, 0.0031, 0.0013, 0.0039, 0.0033, 0.0021, 0.0010, 0.0033, 0.0028, 0.0021, 0.0013, 0.0032, 0.0027, 0.0022, 0.0015, 0.0032, 0.0028, 0.0025, 0.0019, 0.0033, 0.0029, 0.0027, 0.0018, 0.0034, 0.0029, 0.0027, 0.0019, 0.0037, 0.0031, 0.0027, 0.0020, 0.0041, 0.0036, 0.0031, 0.0023, 0.0046, 0.0041, 0.0031, 0.0024, 0.0054, 0.0049, 0.0039, 0.0030, 0.0066, 0.0062, 0.0035, 0.0023, 0.0073, 0.0068, 0.0035, 0.0021, 0.0081, 0.0072, 0.0033, 0.0021, 0.0086, 0.0072, 0.0036, 0.0022, 0.0094, 0.0083, 0.0032, 0.0017, 0.0096, 0.0080, 0.0039, 0.0023, 0.0100, 0.0081, 0.0038, 0.0020, 0.0102, 0.0077, 0.0037, 0.0021, 0.0101, 0.0073, 0.0040, 0.0020, 0.0102, 0.0070, 0.0043, 0.0020, 0.0102, 0.0069, 0.0036, 0.0018, 0.0104, 0.0070, 0.0047, 0.0022, 0.0107, 0.0072, 0.0043, 0.0021, 0.0111, 0.0076, 0.0038, 0.0018, 0.0144, 0.0114, 0.0053, 0.0017, 0.0215, 0.0143]
                if 'llama-3' in args.net.lower():
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
                
                expc_list = []
                
                '''
                for layersgroup in [['q_proj','k_proj','v_proj'],
                                    ['o_proj'],
                                    ['gate_proj','up_proj'],
                                    ['down_proj']]:
               
                    
                    tmp_qlayer = copy.deepcopy(qlayer)
                    replace_linear_with_TmpLinear_part(tmp_qlayer, args, layersgroup)
                  
                    tmp_qlayer.float() 
                    tmp_qlayer = tmp_qlayer.to(dev)

                    if layersgroup[0] == 'q_proj':
                        tmp = ((act_disturb['q_proj'].std(dim=0)/ act_disturb['q_proj'].std()).to(tmp_qlayer.self_attn.q_proj.a1.data))
                        tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                        tmp = torch.min(tmp, torch.tensor(32.).to(tmp))
                        expic = tmp_qlayer.self_attn.q_proj.expic
                        tmp = F.pad(tmp, (0, expic - tmp.shape[-1]), mode="constant", value=1.)
                        tmp_qlayer.self_attn.q_proj.a1.data = tmp
                        tmp_qlayer.self_attn.k_proj.a1.data = tmp
                        tmp_qlayer.self_attn.v_proj.a1.data = tmp
                    if layersgroup[0] == 'o_proj':
                        tmp = ((act_disturb['o_proj'].std(dim=0)/ act_disturb['o_proj'].std()).to(tmp_qlayer.self_attn.o_proj.a1.data))
                        tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                        tmp = torch.min(tmp, torch.tensor(32.).to(tmp))
                        expic = tmp_qlayer.self_attn.o_proj.expic
                        tmp = F.pad(tmp, (0, expic - tmp.shape[-1]), mode="constant", value=1.)
                        tmp_qlayer.self_attn.o_proj.a1.data = tmp
                    if layersgroup[0] == 'up_proj':
                        tmp = ((act_disturb['up_proj'].std(dim=0)/ act_disturb['up_proj'].std()).to(tmp_qlayer.mlp.up_proj.a1.data))
                        tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                        tmp = torch.min(tmp, torch.tensor(32.).to(tmp))
                        expic = tmp_qlayer.mlp.up_proj.expic
                        tmp = F.pad(tmp, (0, expic - tmp.shape[-1]), mode="constant", value=1.)
                        tmp_qlayer.mlp.up_proj.a1.data  = tmp
                        tmp_qlayer.mlp.gate_proj.a1.data = tmp
                    if layersgroup[0] == 'down_proj':
                        tmp = ((act_disturb['down_proj'].std(dim=0)/ act_disturb['down_proj'].std()).to(tmp_qlayer.mlp.down_proj.a1.data))
                        tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                        tmp = torch.min(tmp, torch.tensor(32.).to(tmp))
                        expic = tmp_qlayer.mlp.down_proj.expic
                        tmp = F.pad(tmp, (0, expic - tmp.shape[-1]), mode="constant", value=1.)
                        tmp_qlayer.mlp.down_proj.a1.data = tmp

                    wq_alpha = []
                    scale_list1 = []
                    scale_list2 = []
                    w_list = []
                    
                    for n,m in tmp_qlayer.named_modules():
                        if isinstance(m, TmpLinear):
                            m.input_trans = True
                            m.find_params()
                            m.quantizer.register_parameter('alpha', nn.Parameter(0.*torch.ones(m.quantizer.scale.shape, device = m.orilinear.weight.device , dtype = m.orilinear.weight.dtype )))
                            wq_alpha += [m.quantizer.alpha]
                            scale_list1 += [ m.a3, m.a2]
                            scale_list2 +=  [m.a1]
                            w_list += [m.orilinear.weight]
                            
                    scale_list1 += get_n_set_parameters_byname(tmp_qlayer, ["Trans.linear", ])
                    optimizer = torch.optim.AdamW(
                            [{"params":wq_alpha,"lr":args.lwc_lr},  {"params":scale_list1,"lr":args.lscale_lr}, {"params":scale_list2,"lr":2*args.lscale_lr}],  weight_decay=args.wd)
                        

                    epochs = args.epochs
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs * (256// args.batch_size), eta_min=args.lscale_lr * 1e-2)
                    loss_scaler = utils.NativeScalerWithGradNormCount() 
                    with torch.no_grad():  
                        for n,m in tmp_qlayer.named_modules():
                            if isinstance(m, TmpLinear):
                                m.quant_tmpweight()

                    for epoch in range(1):
                        loss_list = []
                        norm_list = []
                        for j in range(256//args.batch_size): 
                            index = j * args.batch_size 
                            with traincast(device_type='cuda',dtype=args.dtype):
                                for n,m in tmp_qlayer.named_modules():
                                    if isinstance(m, TmpLinear):
                                        m.quant_tmpweight()
                                        m.showflag = False
                                quant_out = tmp_qlayer(fp_inps[index:index+args.batch_size,].to(dev), attention_mask=attention_mask_batch,position_ids=position_ids)
                                loss = loss_func(fp_outs[index:index+args.batch_size,].to(dev), quant_out)
                            if not math.isfinite(loss.item()):
                                logger.info("Loss is NAN, stopping training")
                            else:  
                                optimizer.zero_grad()  
                                loss_list.append(loss.detach().cpu())
                                norm = loss_scaler(loss, optimizer,parameters= get_parameters(tmp_qlayer)).cpu()
                                scheduler.step()
                                norm_list.append(norm.data)
                            loss_mean = torch.stack(loss_list).mean()
                            norm_mean = torch.stack(norm_list).mean()
                            if j%32 == 31:
                                logger.info(f"layer {i} batchs {j} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
                                loss_list = []
                                norm_list = []
                    optimizer.zero_grad()
                    del wq_alpha, w_list, scale_list1, scale_list2, optimizer
                   
                    newloss = loss_mean
                    #energy = (fp_outs[:16]-fp_inps[:16]).to(dev).to(torch.float32).pow(2).mean()
                    ratio = newloss*layer_alpha[i]
                    if ratio<args.mpp :
                        expc_list += ['nl']
                    elif ratio<args.mpp*1.5:
                        expc_list += ['nh']
                    else:
                        expc_list += ['np']
                    print('Setting ', layersgroup, 'as', expc_list[-1], ',loss:',newloss,'final_loss:',ratio)
                    final_loss_list+=[ratio]
                del tmp_qlayer'''
                
                for j in range(4):
                    expc = expc_choice[i*4+j]
                    if expc == 0:
                        expc_list += ['nl']
                    if expc == 1:
                        expc_list += ['nm']
                    if expc == 2:
                        expc_list += ['np']
                    if expc == 3:
                        expc_list += ['nh']
                layerpnums,layerbytes = get_layer_parameters(qlayer, expc_list)
                print('layeravgbits:', layerbytes/layerpnums)
                modelpnums += layerpnums
                modelbytes += layerbytes
                print('modelavgbits:', modelbytes/modelpnums)
                
                torch.cuda.empty_cache()
        
        if i >= args.quant_start:
            ################################
            #Stage0: prepare scale
            print("Doing scale Init...")
            with torch.no_grad():
                
                #qlayer.float() 
                print("Replacing")
                if args.auto_mix_precision:
                    replace_linear_with_TmpLinear_mix(qlayer, args, expc_list)
                else:
                    replace_linear_with_TmpLinear(qlayer, args)
                qlayer.float() 
                qlayer = qlayer.to(dev)
                if args.a1init:
                    print("Add scaling")
                    if any(name.endswith('q_proj') for name, _ in qlayer.named_modules()):
                        tmp = ((act_disturb['q_proj'].std(dim=0)/ act_disturb['q_proj'].std()).to(qlayer.self_attn.q_proj.a1.data))
                        tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                        tmp = torch.min(tmp, torch.tensor(16.).to(tmp))
                        expic = qlayer.self_attn.q_proj.expic
                        tmp = F.pad(tmp, (0, expic - tmp.shape[-1]), mode="constant", value=1.)
                        qlayer.self_attn.q_proj.a1.data = tmp
                        qlayer.self_attn.k_proj.a1.data = tmp
                        qlayer.self_attn.v_proj.a1.data = tmp

                        tmp = ((act_disturb['o_proj'].std(dim=0)/ act_disturb['o_proj'].std()).to(qlayer.self_attn.q_proj.a1.data))
                        tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                        tmp = torch.min(tmp, torch.tensor(16.).to(tmp))
                        expic = qlayer.self_attn.o_proj.expic
                        tmp = F.pad(tmp, (0, expic - tmp.shape[-1]), mode="constant", value=1.)
                        qlayer.self_attn.o_proj.a1.data = tmp
                    else:
                        tmp = ((act_disturb['in_proj_qkv'].std(dim=0)/ act_disturb['in_proj_qkv'].std()).to(qlayer.linear_attn.in_proj_qkv.a1.data))
                        tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                        tmp = torch.min(tmp, torch.tensor(16.).to(tmp))
                        expic = qlayer.linear_attn.in_proj_qkv.expic
                        tmp = F.pad(tmp, (0, expic - tmp.shape[-1]), mode="constant", value=1.)
                        qlayer.linear_attn.in_proj_qkv.a1.data = tmp
                        qlayer.linear_attn.in_proj_z.a1.data = tmp

                        tmp = ((act_disturb['out_proj'].std(dim=0)/ act_disturb['out_proj'].std()).to(qlayer.linear_attn.in_proj_qkv.a1.data))
                        tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                        tmp = torch.min(tmp, torch.tensor(16.).to(tmp))
                        expic = qlayer.linear_attn.out_proj.expic
                        tmp = F.pad(tmp, (0, expic - tmp.shape[-1]), mode="constant", value=1.)
                        qlayer.linear_attn.out_proj.a1.data = tmp

                    
                    
                    if 'moe' in args.net.lower():
                        
                        tmp = ((act_disturb['experts.0.up_proj'].std(dim=0)/ act_disturb['experts.0.up_proj'].std()).to(qlayer.self_attn.q_proj.a1.data))
                        tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                        tmp = torch.min(tmp, torch.tensor(16.).to(tmp))
                        non_finite_mask = ~torch.isfinite(tmp)
                        indices = torch.nonzero(non_finite_mask)
                        if indices.numel() > 0:
                            tmp = ((act_disturb['experts.1.up_proj'].std(dim=0)/ act_disturb['experts.1.up_proj'].std()).to(qlayer.self_attn.q_proj.a1.data))
                            tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                            tmp = torch.min(tmp, torch.tensor(16.).to(tmp))
                        non_finite_mask = ~torch.isfinite(tmp)
                        indices = torch.nonzero(non_finite_mask)
                        if indices.numel() > 0:
                            print("setting 1")
                            tmp.fill_(1.)
                        expic = qlayer.mlp.experts[0].up_proj.expic
                        tmp = F.pad(tmp, (0, expic - tmp.shape[-1]), mode="constant", value=1.)
                        for name, module in  qlayer.named_modules():
                            if (isinstance(module, TmpLinear)) and( ('up_proj' in name) or ('gate_proj' in name)):
                                module.a1.data = tmp
                        
                        expic = qlayer.mlp.experts[0].down_proj.expic
                        tmp = torch.ones(expic).to(qlayer.self_attn.q_proj.a1.data)*1.
                        for name, module in  qlayer.named_modules():
                            if (isinstance(module, TmpLinear)) and( ('down_proj' in name)):
                                module.a1.data = tmp
                            
                    else:
                        tmp = ((act_disturb['up_proj'].std(dim=0)/ act_disturb['up_proj'].std()).to(qlayer.mlp.up_proj.a1.data))
                        tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                        tmp = torch.min(tmp, torch.tensor(16.).to(tmp))
                        expic = qlayer.mlp.up_proj.expic
                        tmp = F.pad(tmp, (0, expic - tmp.shape[-1]), mode="constant", value=1.)
                        qlayer.mlp.up_proj.a1.data  = tmp
                        qlayer.mlp.gate_proj.a1.data = tmp
                    
                        tmp = ((act_disturb['down_proj'].std(dim=0)/ act_disturb['down_proj'].std()).to(qlayer.mlp.up_proj.a1.data))
                        tmp = torch.max(tmp, torch.tensor(1.).to(tmp))
                        tmp = torch.min(tmp, torch.tensor(16.).to(tmp))
                        expic = qlayer.mlp.down_proj.expic
                        tmp = F.pad(tmp, (0, expic - tmp.shape[-1]), mode="constant", value=1.)
                        qlayer.mlp.down_proj.a1.data = tmp
                
                del act_disturb
                print("Done scale Init...")
            
            ###############################################################  
            #Stage1: training transformation
            wq_alpha = []
            scale_list0 = []
            scale_list1 = []
            scale_list2 = []
            w_list = []
            
            for n,m in qlayer.named_modules():
                if isinstance(m, TmpLinear):
                    m.input_trans = True
                    #m.output_trans = True
                    #if 'o_proj' in n or 'down_proj':
                    #    m.output_trans = True
                    
                    m.find_params()
                    m.quantizer.register_parameter('alpha', nn.Parameter(0.*torch.ones(m.quantizer.scale.shape, device = m.orilinear.weight.device , dtype = m.orilinear.weight.dtype )))
                    #print(m.quantizer.alpha.device)
                    wq_alpha += [m.quantizer.alpha]
                    #scale_list1 += [ m.a3, m.a2]
                    scale_list1 += [ m.a2]
                    scale_list2 +=  [m.a1]
                    w_list += [m.orilinear.weight]
                
            scale_list0 += get_n_set_parameters_byname(qlayer, ["Trans.linear", ])
            if args.transmask[0] == '1':
                lrscale0 = args.lscale_lr
            else:
                lrscale0 = 0.
            if args.transmask[1] == '1':
                lrscale1 = args.lscale_lr
            else:
                lrscale1 = 0.
            if args.transmask[2] == '1':
                lrscale2 = 2*args.lscale_lr
            else:
                lrscale2 = 0.
            if args.transmask[3] == '1':
                optimizer = torch.optim.AdamW(
                    [{"params":wq_alpha,"lr":args.lwc_lr}, {"params":w_list,"lr":args.lw_lr},  {"params":scale_list0,"lr":lrscale0}, {"params":scale_list1,"lr":lrscale1}, {"params":scale_list2,"lr":lrscale2}],  weight_decay=args.wd)
                
            else:
                optimizer = torch.optim.AdamW(
                    [{"params":wq_alpha,"lr":args.lwc_lr},  {"params":scale_list0,"lr":lrscale0}, {"params":scale_list1,"lr":lrscale1}, {"params":scale_list2,"lr":lrscale2}],  weight_decay=args.wd)
                print(args.lwc_lr,lrscale0,lrscale1,lrscale2)

            epochs = args.epochs1
            if args.nsamples1 == args.nsamples:
                args.nsamples1 = args.nsamples1 - args.nsamples//32
            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs * (args.nsamples1// args.batch_size), eta_min=args.lscale_lr * 1e-2)
            empty_optimizer_list = [torch.optim.AdamW([torch.tensor(0)], lr=optimizer.param_groups[k]['lr']) for k in range(len(optimizer.param_groups))]
            scheduler_list = [torch.optim.lr_scheduler.CosineAnnealingLR(empty_optimizer_list[k], T_max=epochs * (args.nsamples1// args.batch_size), eta_min = optimizer.param_groups[k]['lr']/20) for k in range(len(optimizer.param_groups))]
            loss_scaler = utils.NativeScalerWithGradNormCount() 
            with torch.no_grad():  
                for n,m in qlayer.named_modules():
                    if isinstance(m, TmpLinear):
                        m.quant_tmpweight()
            
            for epoch in range(epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples1//args.batch_size): 
                    index = j * args.batch_size 
                    with traincast(device_type='cuda',dtype=args.dtype):
                        for n,m in qlayer.named_modules():
                            if isinstance(m, TmpLinear):
                                m.quant_tmpweight()
                                m.showflag = False
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,].to(dev), attention_mask=attention_mask_batch,position_embeddings=position_embeddings)
                        loss = loss_func(fp_outs[index:index+args.batch_size,].to(dev), quant_out)
                        
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        non_finite_mask_inp = ~torch.isfinite(quant_inps[index:index+args.batch_size,])
                        non_finite_mask_qout = ~torch.isfinite(quant_out)
                        non_finite_mask_fpout = ~torch.isfinite(fp_outs[index:index+args.batch_size,])
                        indices_inp = torch.nonzero(non_finite_mask_inp)
                        indices_qout = torch.nonzero(non_finite_mask_qout)
                        indices_fpout = torch.nonzero(non_finite_mask_fpout)
        
                        if indices_inp.numel() > 0:
                            for idex in indices_inp:
                                 print(f" input- 索引: {idex.tolist()}, 值为: {quant_inps[index:index+args.batch_size,][tuple(idex)]}")
                        if indices_qout.numel() > 0:
                            for idex in indices_qout:
                                 print(f" qoutput- 索引: {idex.tolist()}, 值为: {quant_out[tuple(idex)]}")
                        if indices_fpout.numel() > 0:
                            for idex in indices_fpout:
                                 print(f" fpout- 索引: {idex.tolist()}, 值为: {fp_outs[index:index+args.batch_size,][tuple(idex)]}")
                    else:  
                        optimizer.zero_grad()  
                        loss_list.append(loss.detach().cpu())
                        norm = loss_scaler(loss, optimizer,parameters= get_parameters(qlayer)).cpu()
                        #scheduler.step()
                        for k in range(len(optimizer.param_groups)):
                            scheduler_list[k].step()
                            optimizer.param_groups[k]['lr'] = scheduler_list[k].get_lr()[0]
                        norm_list.append(norm.data)
                    
                    if j%128 == 127:
                        loss_mean = torch.stack(loss_list).mean()
                        norm_mean = torch.stack(norm_list).mean()
                        logger.info(f"layer {i} batchs {j} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
                        loss_list = []
                        norm_list = []
                        
                        #print((qlayer.mlp.up_proj.a2))
                        #print((qlayer.mlp.up_proj.Trans.linear_diag_left))
            optimizer.zero_grad()
            del wq_alpha, w_list, scale_list1, scale_list2, optimizer


        torch.cuda.empty_cache()
        ####
        ###############################################################                  
        # Stage2: finetuning all weights
        if args.finetuning_weights and i>=args.quant_start:
            layerlist = [   ['q_proj',  'k_proj', 'v_proj'],
                            ['o_proj'],
                            ['in_proj_qkv', 'in_proj_z', 'out_proj'],
                            ['gate_proj', 'up_proj'],
                            ['down_proj'], ['ALL'],
                            ]
            for layer_group in layerlist:
                print("start finetuning all weights")
                if args.auto_mix_precision:
                    replace_TmpLinaer_with_FWTLinear_mix(qlayer, args, layer_group, expc_list)
                else:
                    replace_TmpLinaer_with_FWTLinear(qlayer, args, layer_group)

                qlayer = qlayer.to('cuda')
                for name, param in model.named_parameters():
                    param.requires_grad = False
                for n,m in qlayer.named_modules():
                    if isinstance(m, TmpLinear):
                        m.weight = m.weight.detach()
                if 'moe' in args.net.lower():
                    weight_params = [{"params":get_n_set_parameters_byname_FWT(qlayer, ["weight", ]),"lr": args.lw_lr}]
                else:
                    if any(name.endswith('q_proj') for name, _ in qlayer.named_modules()):
                        weight_params = [{"params":get_n_set_parameters_byname_FWT(l, ["weight", ]),"lr": min(args.lw_lr, l.weight.std().item()/50)} for l in [qlayer.self_attn.k_proj, qlayer.self_attn.v_proj, qlayer.self_attn.q_proj, qlayer.self_attn.o_proj, qlayer.mlp.up_proj, qlayer.mlp.gate_proj, qlayer.mlp.down_proj]]
                    else:
                        weight_params = [{"params":get_n_set_parameters_byname_FWT(l, ["weight", ]),"lr": min(args.lw_lr, l.weight.std().item()/50)} for l in [qlayer.linear_attn.in_proj_qkv, qlayer.linear_attn.in_proj_z, qlayer.linear_attn.out_proj, qlayer.mlp.up_proj, qlayer.mlp.gate_proj, qlayer.mlp.down_proj]]
                
                optimizer = torch.optim.AdamW(
                        weight_params + [ {"params":get_n_set_parameters_byname_FWT(qlayer, ["scale" ]),"lr": args.lw_lr/5}, {"params":get_n_set_parameters_byname_FWT(qlayer, ["linear_" ]),"lr": args.lt_lr}, {"params":get_n_set_parameters_byname_FWT(qlayer, ["a1","a2" ]),"lr": args.la_lr}] ,weight_decay=args.wd)

                #print_trainable_parameters(qlayer)
                if layer_group == ['ALL']:
                    epochs = args.epochs2
                    if args.nsamples2 == args.nsamples:
                        args.nsamples2 = args.nsamples2 - args.nsamples2//32
                    samplenums= args.nsamples2
                else:
                    epochs = 0
                    samplenums= 256

                T = epochs * ( samplenums // args.batch_size)
                #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs * ( samplenums // args.batch_size), eta_min= 5e-7)
                
                empty_optimizer_list = [torch.optim.AdamW([torch.tensor(0)], lr=optimizer.param_groups[k]['lr']) for k in range(len(optimizer.param_groups))]
                scheduler_list = [torch.optim.lr_scheduler.CosineAnnealingLR(empty_optimizer_list[k], T_max=T, eta_min = optimizer.param_groups[k]['lr']/20) for k in range(len(optimizer.param_groups))]
                #weight_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(empty_optimizer_1, T_max=T, eta_min=args.lw_lr/20)
                #empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.lt_lr)
                #trans_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(empty_optimizer_2, T_max=T, eta_min=args.lt_lr/20)
                #empty_optimizer_3 = torch.optim.AdamW([torch.tensor(0)], lr=args.la_lr)
                #ascale_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(empty_optimizer_3, T_max=T, eta_min=args.la_lr/20)
             

                loss_scaler = utils.NativeScalerWithGradNormCount() 

                for epoch in range(epochs):
                    loss_list = []
                    norm_list = []
                    for j in range(samplenums //args.batch_size): 
                        index = j * args.batch_size 
                        with traincast(device_type='cuda',dtype=args.dtype):
                            
                            quant_out = qlayer(quant_inps[index:index+args.batch_size,].to(dev), attention_mask=attention_mask_batch,position_embeddings=position_embeddings)
                            loss = loss_func(fp_outs[index:index+args.batch_size,].to(dev), quant_out)
                        if not math.isfinite(loss.item()):
                            logger.info("Loss is NAN, stopping training")
                            
                        else:  
                            optimizer.zero_grad()  
                            loss_list.append(loss.detach().cpu())
                            norm = loss_scaler(loss, optimizer,parameters= get_parameters(qlayer)).cpu()
                            #scheduler.step()
                            # adjust lr
                            
                            for k in range(len(optimizer.param_groups)):
                                scheduler_list[k].step()
                                optimizer.param_groups[k]['lr'] = scheduler_list[k].get_lr()[0]
                                if args.pvtuning:
                                    if (epoch+j//8)%2 == 0:
                                        if k>0:
                                            optimizer.param_groups[k]['lr'] = 0.
                                        else:
                                            optimizer.param_groups[k]['lr'] =  scheduler_list[k].get_lr()[0]*10
                                    else:
                                        if k==0:
                                            optimizer.param_groups[k]['lr'] = 0.
                                
                                    
                
                            norm_list.append(norm.data)
                        loss_mean = torch.stack(loss_list).mean()
                        norm_mean = torch.stack(norm_list).mean()
                        if j%128 == 127:
                            logger.info(f"layer {i} batchs {j} loss:{loss_mean} lr:{optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
                            loss_list = []
                            norm_list = []

        qlayer.to(dtype)
        with torch.no_grad():
            for n,m in qlayer.named_modules():
                if isinstance(m, FWTLinear):
                    m.pack_to_int8()

        if args.epochs1>0: 
            
            with torch.no_grad():
                #with torch.cuda.amp.autocast():
                with traincast(device_type='cuda',dtype=args.dtype):
                    batch_size = args.batch_size * 2
                    for j in tqdm(range(args.nsamples//batch_size)): 
                        index = j*batch_size
                        if i < args.quant_start or args.align <= 1:
                            quant_inps[index:index+batch_size,] = fp_outs[index:index+batch_size,]*1.
                        else:
                            quant_inps[index:index+batch_size,] = qlayer(quant_inps[index:index+batch_size,].to(dtype).to(dev), attention_mask=attention_mask,position_embeddings=position_embeddings).to('cpu')
                    
                # pack weight to int8
                
                layers[i] = qlayer.to("cpu")
            #if i==2:
            #    torch.save(fp_outs,"./layer2outputs.pth")
            print(fp_outs.flatten()[0:16])
            print(quant_inps.flatten()[0:16])
            logger.info(f"MSE: {(fp_outs[-args.nsamples//32:]- quant_inps[-args.nsamples//32:]).to(dev).to(torch.float32).pow(2).mean()}, Energy: {((fp_outs[:16]).to(dev).to(torch.float32).pow(2).mean())}")
            if args.align > 1:
                if i%args.align == args.align-1:
                    print('aligning')
                    for k in range(args.nsamples-args.nsamples//32):
                        quant_inps[k] = fp_outs[k]*1.
            
            del qlayer
        else:
            layers[i] = qlayer.to("cpu")
        torch.cuda.empty_cache()
        if args.save_dir and args.save_per_layer and i>=args.quant_start:
            
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir,args.net, args.net+'+'+args.expc+'-layer'+str(i)+'.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(layers[i].state_dict(), save_path)
            print(f"Quantized model has been saved to：{save_path}")
        if args.save_per_layer:
            layers[i] = None 
            gc.collect()
        '''
        if i == args.quant_end-1:
            print(final_loss_list)
            for r in range(100):
                modelpnums = 0
                modelbytes = 0
                mmp = 0.0002 * r
                for n in range(args.quant_end):
                    expc_list = []
                    for j in range(4):
                        ratio = final_loss_list[n*4+j]
                        if ratio<mmp :
                            expc_list += ['nl']
                        elif ratio<mmp*1.5:
                            expc_list += ['nh']
                        else:
                            expc_list += ['np']

                    layerpnums,layerbytes = get_layer_parameters(layers[i], expc_list)
                    modelpnums += layerpnums
                    modelbytes += layerbytes
                print('modelavgbits:', modelbytes/modelpnums,mmp)'''
    
    torch.cuda.empty_cache()
        
    del inps
    del quant_inps
    del fp_outs
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

