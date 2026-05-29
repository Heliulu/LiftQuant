# This file is modified from https://github.com/artidoro/qlora/blob/main/qlora.py 
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np


import torch
import transformers
import argparse
from transformers import (
    set_seed,
    Seq2SeqTrainer,
    LlamaTokenizer
)
##############################

from datautils_block import test_ppl
from datautils_e2e import make_data_module # 修改3 改成目前的本地cache目录
#from bitsandbytes.optim import AdamW
from torch.optim import AdamW
import os
import utils
#from quantize.int_linear_real import load_quantized_model,QuantLinear # 修改1 加载我自己的量化模型以及替代量化层
from e2e_utils import load_quantized_model
from quantize.tmplinear import  FWTLinear
from pathlib import Path

###################################
    

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

@dataclass
class ModelArguments:
    fp_model_path: Optional[str] = field(
        default="",
        metadata={"help": "path of the fp model."}
    )
    quant_model_path: Optional[str] = field(
        default="",
        metadata={"help": "path of the quantization model."}
    )
    model_family: Optional[str] = field(
        default="llama-2",
        metadata={"help": "for the saving of dataset cache for faster experiments"}
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )
    auto_mix_precision: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    eval_tasks: str = field(
        default='',
        metadata={"help": "evaluation tasks for lm eval, example:piqa,arc_easy,arc_challenge,hellaswag,winogrande"}
    )
    mmlu: bool = field(
        default = False,
        metadata={"help": "evaluation tasks for lm eval, example:piqa,arc_easy,arc_challenge,hellaswag,winogrande"}
    )
    conv_temp: str = field(
        default='llama-2',
        metadata={"help": "Conversation template, only useful with deita datasets"}
    )
    mask_use: bool = field(
        default=True, metadata={"help": "mask the loss to role in dialogue datas"}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|redpajama]"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=32,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    do_expand_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the gsm8k evaluation."}
    )
    do_ppl_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the PPL evaluation."}
    )
    do_not_pre_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the PPL evaluation."}
    )
    pt_context_len: int = field(
        default=1024,
        metadata={"help": "language modeling length."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    wbits: int = field(
        default=2,
        metadata={"help": "How many bits to use."}
    )
    expc: str = field(
        default='none',
        metadata={"help": "How many bits to use."}
    )
    load_per_layer: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the PPL evaluation."}
    )
    early_exit: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the PPL evaluation."}
    )
    w_ternary: str = field(
        default=None,
        metadata={"help": "How many bits to use."}
    )

    group_size: int = field(
        default=64,
        metadata={"help": "How many group size to use."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    resume_from_checkpoint: str = field(default=None, metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=-1, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=2e-5, metadata={"help": 'The learnign rate'})
    factor_a: float = field(default=100, metadata={"help": 'The learnign rate factor for a1 a2'})
    factor_t: float = field(default=10, metadata={"help": 'The learnign rate factor for transformation'})
    factor_h: float = field(default=1, metadata={"help": 'The learnign rate factor for transformation'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='cosine', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=1, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=False, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='epoch', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=5, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)



def get_accelerate_model(args, checkpoint_dir):      ###### 2 这里需要修改 变成自己的模型

    if torch.cuda.is_available():                    # 获取多种硬件的信息，提高兼容性
        n_gpus = torch.cuda.device_count()
  
        
    max_memory = f'{args.max_memory_MB}MB'          # 获取最大显存信息
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device。 分布式训练DDP的设置
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}


    
    model, tokenizer = load_quantized_model(args.fp_model_path, args.quant_model_path, args.wbits, args.expc, args.w_ternary, args.load_per_layer, args.auto_mix_precision) #获得量化模型和分词器，修改
    tokenizer.model_max_length = args.pt_context_len
    
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))   # 设置计算精度     
    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        print('Intel XPU does not support float16 yet, so switching to bfloat16')

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)) # 这里的正则项是不是错了。原文里好像提到了可训练参数用fp32来跑
    # from peft import prepare_model_for_kbit_training
    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    model.cuda()
    model.train()
        
    if tokenizer.pad_token is None:    # Tokenizer缺pad token时补齐 提高兼容性
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )    

    # TODO
    # if 'llama1' in args.model_name_or_path or 'llama2' in args.model_name_or_path or 'llama-1' in args.model_name_or_path or 'llama-2' in args.model_name_or_path:
    if isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        '''
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                ),
        })'''
        # --- 修改开始：安全地获取 pad_token_id ---
        # 1. 优先尝试从 config 获取
        pad_id = model.config.pad_token_id
        
        # 2. 如果 config 里没有或者为 -1，尝试从 tokenizer 获取
        if pad_id is None or pad_id == -1:
            pad_id = tokenizer.pad_token_id
            
        # 3. 如果还是 None，鉴于代码注释提到 "pad_token_id is 0 which corresponds to <unk>"
        #    我们强制将其设为 0，防止报错
        if pad_id is None:
            pad_id = 0 
        # --- 修改结束 ---

        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id), 
                "unk_token": tokenizer.convert_ids_to_tokens(pad_id),
        })


    for name, param in model.named_parameters(): # 冻结所有可训练参数
        # freeze base model's layers
        param.requires_grad = False
        

    # cast all non INT8 parameters to fp32。        # 文中提到了，全部使用fp32进行训练
    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.data = param.data.to(torch.bfloat16)
    print(args.bf16)    
    if args.gradient_checkpointing:                 # 使用梯度检查点节约显存
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)    
            
        model.gradient_checkpointing_enable()

    for name, module in model.named_modules():   # 激活等模块还是使用bf16
        # if isinstance(module, QuantLinear):
        #     # transfer trainable step size into float32
        #     module.scales.data = module.scales.data.to(torch.float32)
        if 'norm' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
                    # module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model, tokenizer

def print_trainable_parameters(args, model):      
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    print('trainable module')
    print('*'*80)
    for name, param in model.named_parameters():
        #print(name,  param.numel())
        all_param += param.numel()
        if param.requires_grad:
            print(name, 'Trainable',  param.numel())
            trainable_params += param.numel()
    print('*'*80)
    if args.wbits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg






def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

def train():
    hfparser = transformers.HfArgumentParser((                      #  参数解析，并用args统一封装
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)       #    训练和log输出目录
    logger = utils.create_logger(args.output_dir)
    logger.info(args)
    
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir) # 断点检查 调用get_last_checkpoint函数，从输出目录检查训练是否已完成、或者有无历史断点可恢复。若已完成，则可以跳过训练或给用户提醒
    if completed_training:
        print('Detected that training was already completed!')

    model, tokenizer = get_accelerate_model(args, checkpoint_dir) # 加载模型和分词器
    
    model.config.use_cache = False
    print('loaded model')
    


     # ================================================================
    # =========== 在这里插入快速测试代码 ============
    # ================================================================
    print("\n" + "="*50)
    print("🚀 RUNNING A QUICK GENERATION TEST...")
    print("="*50)

    # 确保模型处于评估模式，这会关闭 dropout 等
    model.eval() 
    
    # 检查 pad_token，对于生成很重要
    if tokenizer.pad_token_id is None:
        print("Warning: pad_token_id is not set. Setting it to eos_token_id for generation.")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 准备输入
    prompt_text = "I am a PHD student"
    # 对于 Base 模型，我们不需要任何特殊模板
    print(f"Input prompt: '{prompt_text}'")

    # 使用分词器编码。`return_tensors='pt'` 会返回 PyTorch 张量
    # .to(model.device) 确保输入张量和模型在同一个设备上 (比如 GPU)
    inputs = tokenizer(prompt_text, return_tensors='pt').to(model.device)
    
    # 使用 no_grad 来禁用梯度计算，可以节省显存并加速
    with torch.no_grad():
        # 调用 model.generate() 来生成文本
        # max_new_tokens: 控制模型最多生成多少个新词
        # do_sample=True: 使用采样策略，让输出更多样性。如果想看最可能的输出，可以设为 False
        # top_k, top_p, temperature: 控制采样策略的参数
        outputs = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id  # 告诉模型什么时候该停下
        )

    # 将生成的 token IDs 解码成字符串
    # outputs[0] 是生成的完整序列 (包含输入)
    # skip_special_tokens=True 会移除像 <|endoftext|> 这样的特殊标记
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n--- Model Output ---")
    print(generated_text)
    print("--------------------")
    
    # 从完整输出中只提取新生成的部分
    newly_generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("\n--- Newly Generated Part ---")
    print(newly_generated_text)
    print("----------------------------\n")

    print("✅ QUICK GENERATION TEST FINISHED.")
    print("="*50 + "\n")

    # (可选) 将模型切换回训练模式，准备接下来的训练
    model.train() 



    set_seed(args.seed)

    data_module = make_data_module(tokenizer=tokenizer, args=args)  # 准备数据模块。这儿应该要修改 

    # 在 make_data_module 之后，trainer 定义之前：
    if 'eval_dataset' not in data_module:
        print("⚠️ 警告：没有发现 eval_dataset，正在从训练集切分以强制开启评估！")
        # 强行切分一个小的验证集，确保 Trainer 能跑起来
        split_data = data_module['train_dataset'].train_test_split(test_size=0.01)
        data_module['train_dataset'] = split_data['train']
        data_module['eval_dataset'] = split_data['test']
    optimizer_grouped_parameters = [] # 优化器设置，这儿用的AdamW是否需要换一下？
    for name, module in model.named_modules():
        # if isinstance(module, LoraLayer):
        #if 'head' in name:
        #    module.weight.requires_grad = True  
            #print(module.weight.std())
        if isinstance(module, FWTLinear) and not 'head' in name:   
            module.scale.requires_grad = True                          # 这儿需要添加参数
            #print(module.scale.std())
            module.a1.requires_grad = True
            module.a2.requires_grad = True
            module.Trans.linear_left.requires_grad = True
            module.Trans.linear_right.requires_grad = True
            #module.Trans.linear_diag_left.requires_grad = True
            #module.Trans.linear_diag_right.requires_grad = True
            
    optimizer_grouped_parameters.append({'params': [p for n, p in model.named_parameters() if ('scale' in n) ], 'weight_decay': 0.0, 'lr': args.learning_rate})
    optimizer_grouped_parameters.append({'params': [p for n, p in model.named_parameters() if ('a1' in n or 'a2' in n )], 'weight_decay': 0.0, 'lr': args.factor_a*args.learning_rate})
    optimizer_grouped_parameters.append({'params': [p for n, p in model.named_parameters() if ('linear_' in n ) ], 'weight_decay': 0.0, 'lr': args.factor_t*args.learning_rate})
    #optimizer_grouped_parameters.append({'params': [p for n, p in model.named_parameters() if ('head' in n)], 'weight_decay': 0.0, 'lr': args.factor_h * args.learning_rate})
    
    optimizer = AdamW(optimizer_grouped_parameters)   # 修改
 
    trainer = Seq2SeqTrainer(        #配置HuggingFace的Seq2SeqTrainer，实现训练/评估/预测统一流程
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizers=(optimizer, None),
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )
    
    if args.do_not_pre_eval==False:
        with torch.no_grad():
            
            results = test_ppl(trainer.model, trainer.tokenizer, datasets=['c4','wikitext2'],ppl_seqlen=2048)
            logger.info(results)
            trainer.log(results)
            
            results = test_ppl(trainer.model, trainer.tokenizer, datasets=['c4','wikitext2'],ppl_seqlen=4096)
            logger.info(results)
            trainer.log(results)
            
            
            results = test_ppl(trainer.model, trainer.tokenizer, datasets=['c4','wikitext2'],ppl_seqlen=8192)
            logger.info(results)
            trainer.log(results)
        
        if args.eval_tasks != "" or args.do_mmlu_eval or args.do_expand_eval:
            import lm_eval
            from lm_eval.models.huggingface import HFLM
            from lm_eval.utils import make_table
        if args.do_expand_eval:
            lm_eval_model = HFLM(pretrained=model, batch_size=16)
            results = lm_eval.simple_evaluate( # call simple_evaluate
            model=lm_eval_model,
            tasks=['ceval-valid'],
            cache_requests=True,)
            logger.info(make_table(results)) 
            results = lm_eval.simple_evaluate( # call simple_evaluate
            model=lm_eval_model,
            tasks=['leaderboard_gpqa_diamond'],
            cache_requests=True,)
            logger.info(make_table(results))
            del lm_eval_model
        if args.do_mmlu_eval:
            lm_eval_model = HFLM(pretrained=model, batch_size=8)
            task_manager = lm_eval.tasks.TaskManager()
            results = lm_eval.simple_evaluate( # call simple_evaluate
            model=lm_eval_model,
            tasks=['mmlu'],
            num_fewshot=5,
            cache_requests=True,
            )
            logger.info(make_table(results))
            total_acc = 0
            for task in results['results']:
                total_acc += results['results'][task]['acc,none']
            logger.info(f"Average MMLU Acc: {total_acc/len(results['results'])*100:.2f}%")
            del lm_eval_model
        
            
        if args.eval_tasks != "":
            task_list = args.eval_tasks.split(',')
            lm_eval_model = HFLM(pretrained=model, batch_size=16)
            task_manager = lm_eval.tasks.TaskManager()
            results = lm_eval.simple_evaluate( # call simple_evaluate
            model=lm_eval_model,
            tasks=task_list,
            num_fewshot=0,
            task_manager=task_manager,
            )
            logger.info(make_table(results))
            total_acc = 0
            for task in task_list:
                total_acc += results['results'][task]['acc,none']
            logger.info(f'Average Acc: {total_acc/len(task_list)*100:.2f}%')
            del lm_eval_model
    
    
    if args.early_exit:
        return
    
    if args.do_ppl_eval:                              # 若配置了do_ppl_eval，给Trainer挂一个钩子，训练过程中会在eval时对其它数据集（如wikitext2/c4）进行ppl困惑度测试，日志输出
        class PPLvalCallback(transformers.TrainerCallback):
            @torch.no_grad()
            def on_evaluate(self, args=None, state=None, control=None, model=None, **kwargs):
                results = test_ppl(trainer.model, trainer.tokenizer, datasets=['c4','wikitext2'],ppl_seqlen=2048)
                logger.info(results)
                trainer.log(results)
                results = test_ppl(trainer.model, trainer.tokenizer, datasets=['c4','wikitext2'],ppl_seqlen=8192)
                logger.info(results)
                trainer.log(results)

        trainer.add_callback(PPLvalCallback)
    
    # Verifying the datatypes and parameter counts before training. 打印可训练参数和当前dtype分布，方便检查量化/冻结/混合精度等是否符合预期
    # print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    all_metrics = {"run_name": args.run_name}



    print(args.output_dir)    # 训练 验证 预测
    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train(args.resume_from_checkpoint)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if (args.do_train or args.do_eval or args.do_predict):      # 汇总所有run到的metrics，保证各项结果统一输出， 保存 summary 指标
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))
    
    results = test_ppl(trainer.model, trainer.tokenizer, datasets=['c4','wikitext2'],ppl_seqlen=2048)
    logger.info(results)
    trainer.log(results)
            
    results = test_ppl(trainer.model, trainer.tokenizer, datasets=['c4','wikitext2'],ppl_seqlen=4096)
    logger.info(results)
    trainer.log(results)
    
    results = test_ppl(trainer.model, trainer.tokenizer, datasets=['c4','wikitext2'],ppl_seqlen=8192)
    logger.info(results)
    trainer.log(results)
    
    if args.eval_tasks != "" or args.do_mmlu_eval or args.do_expand_eval:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table

    if args.eval_tasks != "":
        task_list = args.eval_tasks.split(',')
        lm_eval_model = HFLM(pretrained=model, batch_size=16)
        task_manager = lm_eval.tasks.TaskManager()
        results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_eval_model,
        tasks=task_list,
        num_fewshot=0,
        task_manager=task_manager,
        )
        logger.info(make_table(results))
        total_acc = 0
        for task in task_list:
            total_acc += results['results'][task]['acc,none']
        logger.info(f'Average Acc: {total_acc/len(task_list)*100:.2f}%')
        del lm_eval_model

    if args.do_expand_eval:
        lm_eval_model = HFLM(pretrained=model, batch_size=16)
        
        results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_eval_model,
        tasks=['ceval-valid'],
        cache_requests=True,)
        logger.info(make_table(results)) 
        
        results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_eval_model,
        tasks=['leaderboard_gpqa_diamond'],
        cache_requests=True,)
        logger.info(make_table(results))
        del lm_eval_model
        
    if args.do_mmlu_eval:
        lm_eval_model = HFLM(pretrained=model, batch_size=8)
        task_manager = lm_eval.tasks.TaskManager()
        results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_eval_model,
        tasks=['mmlu'],
        num_fewshot=5,
        cache_requests=True,
        )
        logger.info(make_table(results))
        total_acc = 0
        for task in results['results']:
            total_acc += results['results'][task]['acc,none']
        logger.info(f"Average MMLU Acc: {total_acc/len(results['results'])*100:.2f}%")
        del lm_eval_model
if __name__ == "__main__":
    train()
