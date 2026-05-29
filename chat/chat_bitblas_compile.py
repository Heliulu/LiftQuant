import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
import argparse
import time
import torch
from transformers import AutoTokenizer
from transformers.cache_utils import StaticCache
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat.chat_quant_bitblas import (
    create_quantized_model_structure_bitblas,
    load_layer_weights_to_bitblas_model,
    convert_model_to_bitblas,
    save_bitblas_model,
    load_bitblas_model
)

# 开启 TensorFloat32 加速
torch.set_float32_matmul_precision('high')

# ==========================================
# 1. 编译部分：只负责输出 Logits (形状绝对固定)
# ==========================================
@torch.compile(mode="max-autotune", fullgraph=True)
def compiled_forward_step(model, input_ids, position_ids, past_key_values):
    outputs = model(
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    return outputs.logits[:, -1, :]

# ==========================================
# 2. 流式生成生成器 (Generator)
# ==========================================
def stream_generate(
    model, 
    tokenizer, 
    prompt, 
    past_key_values, 
    max_new_tokens=512, 
    max_seq_length=2048,
    temperature=0.7,
    top_k=20,
    top_p=0.9,
    repetition_penalty=1.1,
):
    device = model.device
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    prompt_length = input_ids.shape[1]
    
    processors = LogitsProcessorList()
    if repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if temperature > 0:
        if temperature != 1.0:
            processors.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            processors.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            processors.append(TopPLogitsWarper(top_p))

    # ------------------------------------------
    # 阶段 A: Prefill
    # ------------------------------------------
    position_ids = torch.arange(0, prompt_length, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token_logits = outputs.logits[:, -1, :]
        
        next_token_logits = processors(input_ids, next_token_logits)
        if temperature > 0:
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    
    generated_tokens = [input_ids, next_token]
    current_length = prompt_length

    generated_ids = [next_token.item()]
    current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    printed_len = len(current_text)
    
    yield current_text 

    # ------------------------------------------
    # 阶段 B: Decode (逐字生成)
    # ------------------------------------------
    current_position_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # --- 测速状态初始化 ---
    is_first_decode_step = True
    compilation_time = 0.0
    pure_decode_start_time = 0.0
    decode_tokens = 0
    
    step_start_time = time.time()
    
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            torch.compiler.cudagraph_mark_step_begin()
            
            current_position_ids.fill_(current_length)
            
            logits = compiled_forward_step(
                model=model,
                input_ids=next_token,
                position_ids=current_position_ids,
                past_key_values=past_key_values
            )
            
            logits = logits.clone()
            current_input_ids = torch.cat(generated_tokens, dim=-1)
            logits = processors(current_input_ids, logits)
            
            if temperature > 0:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_tokens.append(next_token)
            current_length += 1
            
            # next_token.item() 会隐式触发 CPU-GPU 同步，确保计时准确
            generated_ids.append(next_token.item())
            current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            chunk = current_text[printed_len:]
            printed_len = len(current_text)
            
            if chunk:
                yield chunk
                
            # --- 核心测速逻辑分离 ---
            if is_first_decode_step:
                # 记录第一步（包含编译）的耗时
                compilation_time = time.time() - step_start_time
                # 重置计时器，开始纯净测速
                pure_decode_start_time = time.time()
                is_first_decode_step = False
            else:
                # 只统计第二步及以后的 token 数量
                decode_tokens += 1
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    # 计算纯净的 Decode 速度
    pure_decode_time = time.time() - pure_decode_start_time
    speed = decode_tokens / pure_decode_time if pure_decode_time > 0 else 0
    
    # 友好的输出提示：如果第一步耗时超过 1.5 秒，说明触发了编译
    if compilation_time > 1.5:
        yield f"\n\n[🛠️ First Step (Compile): {compilation_time:.2f}s | ⚡ Pure Decode Speed: {speed:.2f} tokens/sec | Tokens: {decode_tokens}]"
    else:
        yield f"\n\n[⚡ Decode Speed: {speed:.2f} tokens/sec | Tokens: {decode_tokens}]"


# ==========================================
# 3. 交互式聊天主循环
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='BitBLAS Compile 加速版本 - 多轮对话')
    parser.add_argument('--fp_model_path', type=str, 
                        default='/mnt/bn/adsinfra-gpu-dev-hl/heliulu/LLMs/Qwen3.5-27B')
    parser.add_argument('--quant_model_path', type=str,
                        default='/mnt/bn/adsinfra-gpu-dev-hl/heliulu/qmodels/LiftQuant/24to8f1/Qwen3.5-27B/Qwen3.5-27B+24to8')
    parser.add_argument('--wbits', type=int, default=3)
    parser.add_argument('--expc', type=str, default='24to8')
    parser.add_argument('--bitblas_model_path', type=str, default=None,
                        help='直接加载已转换的BitBLAS模型路径（跳过转换步骤）')
    parser.add_argument('--save_bitblas_model', type=str, default=None,
                        help='转换完成后保存BitBLAS模型的路径')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BitBLAS Compile 加速版本 - 多轮对话")
    print("=" * 60)
    
    from datetime import datetime
    
    if args.bitblas_model_path and os.path.exists(args.bitblas_model_path):
        print(f"\n模式: 直接加载已转换的BitBLAS模型")
        print(f"\n步骤1: 加载BitBLAS模型")
        start = datetime.now()
        model = load_bitblas_model(
            args.fp_model_path,
            args.bitblas_model_path,
            args.wbits,
            args.expc
        )
        end = datetime.now()
        print(f"加载模型耗时: {end - start}")
    else:
        print(f"\n模式: 从原始权重加载并转换")
        print(f"\n步骤1: 创建量化模型结构")
        start = datetime.now()
        model = create_quantized_model_structure_bitblas(
            args.fp_model_path, 
            args.wbits, 
            args.expc
        )
        end = datetime.now()
        print(f"创建结构耗时: {end - start}")
        
        print(f"\n步骤2: 加载量化权重和浮点残余权重")
        start = datetime.now()
        model = load_layer_weights_to_bitblas_model(model, args.quant_model_path)
        
        end = datetime.now()
        print(f"加载权重耗时: {end - start}")
        
        print(f"\n步骤3: 移动模型到GPU并转换为BitBLAS格式")
        start = datetime.now()
        # 先把整个模型推向 GPU
        model = model.cuda()
        # 执行量化层的格式转换和打包（这部分在 C++ / GPU 上完成）
        model = convert_model_to_bitblas(model)
        model.eval()
        end = datetime.now()
        print(f"转换耗时: {end - start}")
        
        if args.save_bitblas_model:
            print(f"\n步骤4: 保存BitBLAS模型")
            start = datetime.now()
            save_bitblas_model(model, args.save_bitblas_model)
            end = datetime.now()
            print(f"保存模型耗时: {end - start}")
            
    print(f"\n加载tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.fp_model_path, use_fast=False)

    max_seq_length = 1024
    print("Initializing Static KV Cache...")
    global_past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_seq_length,
        device=model.device,
        dtype=model.dtype
    )

    messages = [
        {"role": "system", "content": "You are a helpful, smart, and concise assistant."}
    ]
    
    print("\n" + "="*50)
    print("🚀 Chat Demo Started! (Type 'quit' to exit)")
    print("💡 Note: The first turn will take a few seconds to compile.")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\033[92mUser:\033[0m ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue
                
            messages.append({"role": "user", "content": user_input})
            
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            prompt_length = len(tokenizer(prompt).input_ids)
            if prompt_length + 512 > max_seq_length:
                print("\n[Warning: Context length exceeded. Clearing history...]")
                messages = [messages[0], {"role": "user", "content": user_input}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            print("\033[96mAssistant:\033[0m ", end="", flush=True)
            
            global_past_key_values.reset()
            
            full_response = ""
            
            for chunk in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                past_key_values=global_past_key_values,
                max_new_tokens=max_seq_length - 512,
                max_seq_length=max_seq_length,
                temperature=1.0,
                top_p=0.95
            ):
                if chunk.startswith("\n\n[⚡") or chunk.startswith("\n\n[🛠️"):
                    print(f"\033[90m{chunk}\033[0m")
                else:
                    print(chunk, end="", flush=True)
                    full_response += chunk
            
            print() 
            
            messages.append({"role": "assistant", "content": full_response})
            
        except KeyboardInterrupt:
            print("\n[Chat interrupted by user]")
            break

if __name__ == "__main__":
    main()
