import os
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders, get_redpajama
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from pprint import pprint

import torch.nn as nn
from quantize.liftuq import liftuq
from tqdm import tqdm
import utils
from pathlib import Path



import pdb

from models.int_llama_layer import QuantLlamaDecoderLayer
from quantize.tmplinear import *
from datasets import Dataset

torch.backends.cudnn.benchmark = True

net_choices = [
    "Qwen2.5-3B",
    "Qwen2.5-14B",
    "Qwen2.5-32B",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-3-8B",
    "Llama-3-70B",
    "Llama-3.1-8B",
    "Llama-3.1-70B",
    "Llama-3.2-1B",
    "Llama-3.2-3B"
]


@torch.no_grad()
def evaluate(lm, args, logger):
    results = {}
    
    if "llama" in args.net.lower() or "qwen" in args.net.lower():
        lm.model = lm.model.to(lm.device)
   
    if args.eval_ppl:
        for dataset in ["wikitext2","c4" ]:
            cache_testloader = f'{args.cache_dir}/testloader_{args.cache_name}_{dataset}_all_{2048}.cache'
            if os.path.exists(cache_testloader) :
                testloader = torch.load(cache_testloader, weights_only=False)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=2048,
                )
                torch.save(testloader, cache_testloader)

            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // 2048
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []
            ########## the ppl computing process:
            ########## 
            if args.limit>0 :
                nsamples = args.limit
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * 2048) : ((i + 1) * 2048)].to(lm.device)
                with torch.amp.autocast(device_type='cuda', dtype=args.dtype):
                    if  "llama" in args.net.lower() or "qwen" in args.net.lower():
                        outputs = lm.model.model(batch)
                    #print(batch)
                    hidden_states = outputs[0]
                    logits = lm.model.lm_head(hidden_states)
                    shift_logits = logits[:, :-1, :]
                    shift_labels = testenc[:, (i * 2048) : ((i + 1) * 2048)][
                        :, 1:
                    ].to(lm.model.lm_head.weight.device)
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)).float(),
                        shift_labels.view(-1),
                    )
                    neg_log_likelihood = loss.float() * 2048
                    nlls.append(neg_log_likelihood)
                    
            ############ 
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))
            logger.info(f'{dataset} : {ppl.item()}')
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()

    if args.tasks != "":
        
        with torch.amp.autocast(device_type='cuda', dtype=args.dtype):
            hflm = HFLM(pretrained=lm.model, tokenizer=lm.tokenizer,batch_size=8)
            results = evaluator.simple_evaluate(
                hflm,
                tasks=args.tasks.split(','),
                num_fewshot=args.num_fewshot,
                limit=None if args.limit == -1 else args.limit,
            )

        
        logger.info(make_table(results))

        metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results['results'].items()}
        mean_acc_val = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
        std_vals = {task: round(result.get('acc_norm_stderr,none', result['acc_stderr,none']), 4) for task, result in results['results'].items()}
        mean_std_val =round(sum(std_vals.values()) / len(std_vals.values()), 4) 
        metric_vals['acc_avg'] = mean_acc_val
        results['results']['AVERAGE'] = {
            "acc,none":mean_acc_val,
            "acc_stderr,none":mean_std_val
        }
        logger.info(make_table(results))
            
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--net", type=str, default=None)
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="../log/", type=str, help="direction of logging file")
    parser.add_argument("--scale_load_dir", default=None, type=str, help="direction for loading clipping paras")
    parser.add_argument('--save_dir', type=str, default='', help='Directory to save the model weights (.pth)')
    parser.add_argument('--load_dir', type=str, default='', help='Directory to save the model weights (.pth)')

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--real_quant", default=False, action="store_true", help="real quantization, which can see memory reduce. Note that due to the limitations of AutoGPTQ kernels, the real quantization of weight-only quantization can only lead memory reduction, but with slower inference speed.")
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "redpajama"],
        help="Where to extract calibration data from.",
    )

    parser.add_argument("--epochs1", type=int, default=10, help="Training epochs for bias and asymmetric scaling")
    parser.add_argument("--epochs2", type=int, default=10, help="Training epochs for bias and asymmetric scaling")
    parser.add_argument("--pre_epochs", type=int, default=0, help="Pre Training epochs for scaling and rotation_ov")
    # rename
    parser.add_argument("--nsamples1", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--nsamples2", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--align", type=int, default=-1)
    parser.add_argument("--quant_start", type=int, default=0)
    parser.add_argument("--quant_end", type=int, default=999)
    parser.add_argument("--fast_nearest", default=False, action="store_true")
    parser.add_argument("--pvtuning", default=False, action="store_true")
    parser.add_argument("--save_per_layer", default=False, action="store_true")
    parser.add_argument("--transmask", type=str, default='1110')
    parser.add_argument("--a1init", default=True, action="store_false")
    parser.add_argument("--seqlen", type=int, default=2048)

    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--w_ternary", type=str,default=None)
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--kbits", type=int, default=4)
    parser.add_argument("--vbits", type=int, default=4)
    parser.add_argument("--auto_mix_precision", default=False, action="store_true")
    parser.add_argument("--mpp", type=float, default=0.001)

    parser.add_argument("--training_trans", default=False, action="store_true", help="Use SVD training matrix")
    parser.add_argument("--expc", type=str, default="n")
    parser.add_argument("--hex", default=False, action="store_true")
    parser.add_argument("--finetuning_weights", default=False, action="store_true")
    parser.add_argument("--usefullfp", default=False, action="store_true")
    #parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lab_lr", type=float, default=2e-3)
    parser.add_argument("--lscale_lr", type=float, default=5e-3)
    parser.add_argument("--lac_lr", type=float, default=1e-2)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--lexw_lr", type=float, default=5e-3)
    parser.add_argument("--lvr_lr", type=float, default=5e-3)
    parser.add_argument("--lw_lr", type=float, default=2e-5)
    parser.add_argument("--la_lr", type=float, default=1e-3)
    parser.add_argument("--lt_lr", type=float, default=1e-3)
    parser.add_argument("--lw2_lr", type=float, default=2e-5)
    parser.add_argument("--la2_lr", type=float, default=1e-3)
    parser.add_argument("--lt2_lr", type=float, default=1e-3)
    parser.add_argument("--ls_lr", type=float, default=0e-5)

    

    parser.add_argument("--wd", type=float, default=0)

    parser.add_argument("--aug_loss", default=False, action="store_true", help="calculate additional loss with same input")
    parser.add_argument("--lac",default=False, action="store_true",help="activate learnable activation clipping")

    parser.add_argument("--k_quant_method", type=str, default="Hadamard", choices=["None","FFT","Hadamard","PCA"])
    parser.add_argument("--v_quant_method", type=str, default="Hadamard", choices=["None","FFT","Hadamard","PCA"])
    parser.add_argument("--a_quant_method", type=str, default="Hadamard", choices=["None","FFT","Hadamard","PCA"])
    parser.add_argument("--r4_quant_method", type=str, default="Hadamard", choices=["None","FFT","Hadamard","PCA"])
    #

  
    parser.add_argument("--upscale_mask", type=str, default="1111")
    parser.add_argument("--quant_mask", type=str, default="1111")
    parser.add_argument("--scale_tmask", type=str, default="11")
    parser.add_argument("--scale_imask", type=str, default="11")
    

    parser.add_argument("--k_bias", default=False, action="store_true")
    parser.add_argument("--v_bias", default=False, action="store_true")
    parser.add_argument("--gptq", default=True, action='store_false')
    parser.add_argument("--scale_keep", default=False, action="store_true")
    parser.add_argument("--qw_ips", default=False, action="store_true")
    parser.add_argument("--eval_KVnoq", default=False, action="store_true")
    parser.add_argument("--pre_gptq", default=False, action="store_true")
    parser.add_argument("--use_fpinps", default=False, action="store_true")
    parser.add_argument("--use_fpinps_gptq", default=False, action="store_true")
    parser.add_argument("--warmup", default=False, action="store_true")
    parser.add_argument("--train_o_scale", default=False, action="store_true")
    parser.add_argument("--train_vo_rotation", default=False, action="store_true")
    parser.add_argument("--old_scale", default=False, action="store_true")
    parser.add_argument("--vo_PCA_init", default=False, action="store_true")
    parser.add_argument("--Rres_init", default='Hadamard', type=str, choices=["Hadamard","PCA"])
    parser.add_argument("--bias_init", default='zeros', type=str, choices=["zeros","means"])
    parser.add_argument("--norm_instruction", type=str, default="ori",choices=["ori","orib","hnorm", "norm"])
    parser.add_argument("--dtype", type=str, default="float16",choices=["float16","bfloat16"])
    parser.add_argument("--disable_realq_replace", default=False, action="store_true")
    parser.add_argument("--disable_rotation", default=False, action="store_true")
    parser.add_argument("--only_eval", default=False, action="store_true")
    parser.add_argument("--pre_eval", default=False, action="store_true")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)

    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager"],
        help="attention implementation that the model works with",
    )
    


    parser.add_argument('--mse', default=True,  action='store_false', help='Whether to apply the activation order GPTQ heuristic')

    parser.add_argument( '--Use_RTN', default=False, action='store_true', help='Whether to apply the activation order GPTQ heuristic')

    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')

    parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
    parser.add_argument('--w_sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.' )
    parser.add_argument( '--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument(  '--static-groups', action='store_true', help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.')
    
    parser.add_argument( '--info', action='store_true', help='Show info')
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    
    #if (args.wbits<16 and args.wbits>8) or (args.abits<16 and args.abits>8):
    #    args.deactive_amp = True

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]
    if args.dtype =="float16":
        args.dtype = torch.float16
    if args.dtype =="bfloat16":
        args.dtype = torch.bfloat16
    lm = LMClass(args)
    pprint(lm.model)
    lm.seqlen = args.seqlen
    lm.model.eval()
    args.hidden_dim = lm.model.config.hidden_size
    args.kv_group = lm.model.config.num_attention_heads // lm.model.config.num_key_value_heads
    args.ffn_dim = lm.model.config.intermediate_size
    args.heads = lm.model.config.num_attention_heads
    args.head_dim = args.hidden_dim//args.heads
    
    
    for param in lm.model.parameters():
        param.requires_grad = False
    
    args.q_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }
    
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }
    
    args.act_quant_params = {
        "n_bits": args.abits,
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "lac":args.lac,
    }
    
    args.R2_quant_params = {
        "n_bits": args.kbits,
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "quant_method":args.k_quant_method,
        "quant_bias": args.k_bias,
        "lac":args.lac,
    }
    args.R3_quant_params = {
        "n_bits": args.vbits,
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "quant_method":args.v_quant_method,
        "quant_bias": args.v_bias,
        "lac":args.lac,
    }



    # quantization
    if 'llama-2' in args.net.lower():
        args.cache_name = 'llama-2'
    elif 'llama-3' in args.net.lower():
        args.cache_name = 'llama-3'
    elif 'qwen2.5' in args.net.lower():
        args.cache_name = 'qwen2.5'
    elif 'qwen2' in args.net.lower():
        args.cache_name = 'qwen2'
    elif 'qwen3-moe' in args.net.lower():
        args.cache_name = 'qwen3-moe'
    elif 'qwen3.5' in args.net.lower():
        args.cache_name = 'qwen3.5'
    elif 'qwen3.6' in args.net.lower():
        args.cache_name = 'qwen3.6'
    elif 'qwen3' in args.net.lower():
        args.cache_name = 'qwen3'

    if args.pre_eval:
        evaluate(lm, args,logger)
    
    if args.load_dir !='':
        def get_named_linears(module, type):
            # return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}
            return {name: m for name, m in module.named_modules() if isinstance(m, type)}
        # change block
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
        for i in tqdm(range(len(lm.model.model.layers))):
            #lm.model.model.layers[i] = QuantLlamaDecoderLayer(lm.model.config, lm.model.model.layers[i], args).to('cpu') 
            # change layer
            layer = lm.model.model.layers[i] 
            named_linears = get_named_linears(layer, torch.nn.Linear)
            for name, module in named_linears.items():
                with torch.no_grad():
                    fake_linear = torch.nn.Linear(module.in_features,module.out_features,not module.bias is None, device = 'cuda', dtype = torch.float16) 
                    #convert to tmplinear
                    tmplinear = TmpLinear(fake_linear, args.wbits, expc = args.expc, training_trans = True)
                    tmplinear.find_params()
                    tmplinear.quantizer.alpha = torch.nn.Parameter(0.*torch.ones(tmplinear.quantizer.scale.shape, device = tmplinear.orilinear.weight.device , dtype = tmplinear.orilinear.weight.dtype ))
                    # convert to FWTLinear
                    fwtlinear =  FWTLinear()
                    #fwtlinear.convert_form_tmplinear(tmplinear, expc = expc, training_trans = True, groupsize=-1 )
                    fwtlinear.convert_form_tmplinear(tmplinear, bits=args.wbits, expc=args.expc, training_trans = True, groupsize = -1)
                    fwtlinear.bit_channel_convert(True)
                    fwtlinear.pack_to_int8()
                    fwtlinear = fwtlinear.to('cpu')
                    
                    set_op_by_name(layer, name, fwtlinear)
            #buffer_names = {n for n, _ in layer.named_buffers()}
            #print(buffer_names)
            #buffer_names = {n for n, _ in layer.named_parameters()}
            #print(buffer_names)
        # load pth
        state_dict = torch.load(args.load_dir, map_location='cpu')
        lm.model.load_state_dict(state_dict, assign=True, strict=False)

    if not args.only_eval:
        logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        args.nsamples = max(args.nsamples1, args.nsamples2)
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.cache_name}_{args.calib_dataset}_{args.nsamples}_{lm.seqlen}.cache'
        pprint(cache_dataloader)
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader, weights_only=False)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            if args.calib_dataset == 'redpajama':
                dataloader = get_redpajama(
                    nsamples=args.nsamples,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                )
            else:   
                dataloader, _ = get_loaders(
                    args.calib_dataset,
                    nsamples=args.nsamples,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                )
            torch.save(dataloader, cache_dataloader)    
  
        
        liftuq(
            lm,
            args,
            dataloader,
            logger,
        )

        logger.info(time.time() - tick)

        if args.save_dir and args.save_per_layer == False:
      
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir,args.net, args.net+'+'+args.expc+'.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(lm.model.state_dict(), save_path)
            print(f"Quantized model has been saved to：{save_path}")
        else:
            print("Haven't set save_dir")

    evaluate(lm, args,logger)

if __name__ == "__main__":
    print(sys.argv)
    main()
