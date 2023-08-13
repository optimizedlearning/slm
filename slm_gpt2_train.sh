
#!/bin/bash -l

# Request 8 CPUs
#$ -pe omp 8

# Request 2 GPUs
#$ -l gpus=2
#$ -l gpu_c=7.0
#$ -l gpu_memory=14G

#merge the error and output
#$ -j y

#send email at the end
#$ -m e



source scc_setup.sh
python gpt_pile.py \
train.max_steps=225000 \ # this is roughly how many examples we can get through in 12 hours with V100 GPUs
train.lr_warmup=10000 \
train.lr=0.00005 \
train.val_check_interval=25000 \
train.val_batches=1000 \
train.per_device_batch_size=8 \
model.context_length=1024 \
wandb.project=slm_gpt2
