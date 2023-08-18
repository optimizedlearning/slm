#!/bin/bash -l

# Request 8 CPUs
#$ -pe omp 8

# Request 1 GPUs
#$ -l gpus=2
#$ -l gpu_c=7.0
#$ -l gpu_memory=15G

#merge the error and output
#$ -j y

#send email at the end
#$ -m e


source scc_setup.sh


# make sure we get rid of any lingering shared memory
python clean_shared_memory.py
# TOKENIZERS_PARALLELISM=false disables some warnings from the tokenizers library:
# otherwise it detects multiprocessing and disables parallelism on its own with a warning.
TOKENIZERS_PARALLELISM=false composer -n 2 gpt_pile_composer.py \
train.max_steps=225000 \
train.lr_warmup=10000 \
train.lr=0.00005 \
train.val_check_interval=25000 \
train.val_batches=1000 \
train.per_device_batch_size=10 \
train.dataloader_workers=2 \
train.gradient_clip_val=10.0 \
train.compile=True \
train.precision='amp_fp16' \
model.context_length=1024 \
wandb.project=slm_gpt2
