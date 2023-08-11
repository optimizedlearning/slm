
#!/bin/bash -l

# Request 4 CPUs
#$ -pe omp 8

# Request 1 GPU
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=14G

#specify a project (probably not necessary, so currently off)
##     $ -P aclab

#merge the error and output
#$ -j y

#send email at the end
#$ -m e



source scc_setup.sh
python gpt_pile.py \
train.max_batches_per_epoch=10000000 \
train.lr_warmup=10000 \
train.lr=0.00005 \
train.batch_size=8 \
model.context_length=1024 \
wandb_project=slm_gpt2
