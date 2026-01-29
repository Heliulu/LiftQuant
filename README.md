# LiftQuant




# LiftQuant

This repository contains the official implementation for the paper "LiftQuant: Arbitrary Bit-Width LLM via Dimensional Lifting and Projection".

## Step 1. Installation

```
# Create and activate a conda environment
conda create -n liftuq_env python=3.12 -y
conda activate liftuq_env

# Install dependencies
pip install -r requirements.txt
```

## Step 2. Training the `M` Matrix (Optional)

You can train the projection matrix `M` using the provided scripts:

- `python lattice_generator.py`: Trains `M` using the exact exhaustive search method.
- `python lattice_generator2.py`: Trains larger `M` matrices using our heuristic search method.

However, we have already included pre-trained `M` matrices in the `./lattice` directory, so you can **skip this step** for reproducing our main results.

## Step 3. Quantize the Model with Intra-Block Correction

This step performs the main LiftUQ quantization process on a model.

#### Key Bit-Width Configurations:

- `--expc n`: Use 2-bit quantization (with a `20/10` `M` matrix).
- `--expc p`: Use 3-bit quantization (with an `18/6` `M` matrix).

#### Example Command:

```
python main.py \
    --model /path/to/your/Llama-2-7B \
    --save_dir ./qmodels \
    --eval_ppl \
    --wbits 2 \
    --expc n \
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

## Step 4. End-to-End Fine-tuning of Quantization Parameters (Optional)

To save significant time, you can skip Step 1/2/3 and use our pre-computed model checkpoints from the intra-block correction phase. These are also available at this MEGA link.

Download the checkpoint and run the E2E fine-tuning script.

#### Example Command:

```
CUDA_VISIBLE_DEVICES=1 python e2efinetune.py \
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

## Benchmarking Throughput

We provide a script to benchmark the end-to-end throughput for the Llama-2-70B model. This script requires the **BitBLAS** library to be installed.

```
python throughout_test_Llama2_70B.py
```
