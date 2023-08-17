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


cd /projectnb/aclab/cutkosky/slm
source scc_setup.sh


# TOKENIZERS_PARALLELISM=false merely desables some warnings from the tokenizers library
# when it detects multiprocessing and disables parallelism on its own.
# TORCH_DIST_INIT_BARRIER=0 is much more sketchy. According to https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py#L3956
# it seems that there is a particular barrier call that possibly unnecessary in the initialization of torch.distributed.
# In my tests, it seemed that a nontrivial fraction of the time, I would get an error in that barrier,
# but after disabling the barrier with TORCH_DIST_INIT_BARRIER=0 everything seemed to work.
# I would not be surprised if this actually created a race condition somewhere, but
# so far it seems to be the most reliable solution.
TORCH_DIST_INIT_BARRIER=0 TOKENIZERS_PARALLELISM=false composer -n 2 gpt_pile_mosaic.py \
train.max_steps=225000 \
train.lr_warmup=10000 \
train.lr=0.00005 \
train.val_check_interval=25000 \
train.val_batches=1000 \
train.per_device_batch_size=10 \
model.context_length=1024 \
train.compile=True \
train.precision='amp_fp16' \
wandb.project=slm_gpt2
