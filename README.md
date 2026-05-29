# LiftQuant [ICML 2026 Spotlight]

Official implementation of **LiftQuant: Arbitrary Bit-Width LLM via Dimensional Lifting and Projection**.

LiftQuant is a post-training quantization framework for large language models. It enables arbitrary bit-width weight quantization through dimensional lifting and projection, together with intra-block correction for improved accuracy.

> **Status:** This repository is under active development. More checkpoints, scripts, and documentation will be released soon.

---

## News

- **[2026.05.01]** LiftQuant was accepted as an **ICML 2026 Spotlight** paper.
- **[2026.05.28]** Initial release of code and example scripts. Released several pre-quantized checkpoints for quick evaluation.

---

## Quick Start

We provide several pre-quantized checkpoints for quick testing:

- Qwen3.5-27B, 3-bit
- Qwen3.5-27B, 2.5-bit
- Qwen3.6-27B, 3-bit
- Qwen3.6-27B, 2.5-bit
- Qwen3.5-9B, 3-bit

The 3-bit models use a `24-to-8` dimensional reduction mapping, while the 2.5-bit models use a `20-to-8` dimensional reduction mapping.

Checkpoints can be downloaded from:

```text
https://box.nju.edu.cn/d/2f63461b007b4a9c84af/
```

We also provide a chat script accelerated by `bitblas` and `torch.compile` for decoding.

```bash
python chat/chat_bitblas_compile.py \
    --fp_model_path /path/to/your/fp_hf_model \
    --quant_model_path /path/to/your/quant_checkpoint
```

Please make sure the environment is properly installed before running the chat script.

```shell
conda create -n liftquant_env python=3.12 -y
conda activate liftquant_env

pip install -r requirements.txt
```

If you plan to use the accelerated chat script, please also make sure that `bitblas` and the required CUDA/PyTorch versions are correctly installed.

## Step 1: Train the Projection Matrix `M` Optional

LiftQuant relies on projection matrices `M` for dimensional lifting and projection.

```shell
python lattice_generator2.py
```

However, we have already included pre-trained `M` matrices in the `./lattice` directory. Therefore, this step can be skipped when reproducing the main results.

## Step 2: Quantize a Model with Intra-Block Correction

```
python main.py \
    --model /path/to/your/Llama-2-7B \
    --save_dir ./qmodels \
    --eval_ppl \
    --wbits 2 \
    --expc 20to8 \
    --w_sym \
    --abits 16 \
    --kbits 16 \
    --vbits 16 \
    --true-sequential \
    --act-order \
    --use_fpinps \
    --Rres_init Hadamard \
    --nsamples 4096 \
    --epochs 2 \
    --batch_size 2 \
    --calib_dataset redpajama \
    --usefullfp \
    --training_trans \
    --finetuning_weights \
    --align 1 \
    --lscale_lr 5e-3 \
    --lexw_lr 2e-2 \
    --lw_lr 2e-5 \
    --la_lr 2e-3 \
    --lt_lr 2e-4
```

## Step3: End-to-End Fine-tuning of Quantization Parameters Optional

We provide an **optional** end-to-end fine-tuning script for quantization parameters.

However, we observe that although end-to-end fine-tuning can improve  perplexity and some zero-shot benchmark results, it may degrade  performance on more complex tasks. This paradigm is commonly adopted by  many 2-bit quantization methods to achieve stronger paper-level  benchmark numbers, but we do **not** recommend it if your goal is to deploy a practical chat model.

For deployment-oriented use cases, we recommend:

- For around 30B models: use **2.5-bit quantization** with only block correction.
- For 7B/14B models: use **3-bit quantization** with only block correction.

These settings usually provide better generalization ability for chat and reasoning tasks.

### Example Command

```
CUDA_VISIBLE_DEVICES=0 python e2efinetune.py \
    --fp_model_path /path/to/your/Llama-2-7B \
    --quant_model_path ./Llama-2-7B+n.pth \
    --model_family Llama-2 \
    --wbits 2 \
    --expc n \
    --factor_a 100 \
    --factor_t 10 \
    --learning_rate 2e-5 \
    --dataset redpajama \
    --dataset_format pt \
    --output_dir ./output/e2e-qp-output/Llama-2-7B-redpajama-4096 \
    --do_train True \
    --pt_context_len 4096 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --save_strategy epoch \
    --training_strategy epochs \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --max_train_samples 4096 \
    --num_train_epochs 1 \
    --eval_dataset_size 64 \
    --bf16 \
    --data_seed 42 \
    --max_grad_norm 0.3 \
    --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
    --preprocessing_num_workers 32 \
    --do_ppl_eval
```
