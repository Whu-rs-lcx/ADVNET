#!/bin/bash

#SBATCH --partition=a100x4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=60:05:00
#SBATCH --output=001a100.out
#SBATCH --error=001a100.error
#SBATCH --account=mengxiaoliang
#SBATCH --gres=gpu:1
#SBATCH --job-name=001a100


module load nvidia/cuda/12.2

cd $SLURM_SUBMIT_DIR

srun python main.py --model_name SEED_SwinT --pred_idx 1 --loss_weights 1.0 1.0 --batch_size 8 --exp_name SEED_SwinT_SYSU --dataset SYSU-CD --src_size 256 --lr 0.001 --max_epochs 80 --check_val_every_n_epoch 1 --early_stop 20 --warmup 3000 --no-comet