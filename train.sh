#!/bin/bash
#SBATCH --job-name=CNTQG
#SBATCH --output=results/GECO2_%j.txt
#SBATCH --error=results/GECO2_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=4-00:00:00
#SBATCH --exclude=gwn[01-10]

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=50197
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1s

module load Anaconda3
module load  CUDA/12.3.0
source activate cnt2
conda activate base
conda activate cnt2

srun --unbuffered python train.py \
--training \
--model_name GECO2_FSCD \
--model_path /d/hpc/projects/FRI/pelhanj/CNT_SAM2/models/ \
--data_path /d/hpc/projects/FRI/pelhanj/fsc147 \
--backbone resnet50 \
--reduction 16 \
--image_size 1024 \
--emb_dim 256 \
--num_heads 8 \
--kernel_dim 1 \
--num_objects 3 \
--epochs 200 \
--lr 1e-4 \
--backbone_lr 0 \
--lr_drop 50 \
--weight_decay 1e-5 \
--batch_size 4 \
--dropout 0.1 \
--num_workers 8 \
--max_grad_norm 0.1 \
--aux_weight 0.3 \
--tiling_p 0.5 \
--pre_norm