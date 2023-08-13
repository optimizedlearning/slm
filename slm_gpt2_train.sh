
#!/bin/bash -l

# Request 8 CPUs
#$ -pe omp 8

# Request 2 GPUs
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=15G

#merge the error and output
#$ -j y

#send email at the end
#$ -m e



source scc_setup.sh

# 225000 is roughly how many iterations we can get through in 12 hours with V100 GPUs
# for some reason we OOM if we do batch size 8 on 2 gpus, but not on 1 gpu.
# I think it's a torch.compile related thing...
python gpt_pile.py \
train.max_steps=225000 \
train.lr_warmup=10000 \
train.lr=0.0001 \
train.val_check_interval=25000 \
train.val_batches=1000 \
train.per_device_batch_size=8 \
model.context_length=1024 \
train.compile=True \
wandb.project=slm_gpt2
