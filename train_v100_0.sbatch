#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=2
#SBATCH --time=60:05:00
#SBATCH --output=000.out
#SBATCH --error=000.error
#SBATCH --account=mengxiaoliang
#SBATCH --gres=gpu:2
#SBATCH --job-name=lenet_LEVIR


module load nvidia/cuda/12.2

cd $SLURM_SUBMIT_DIR

# srun python main.py --model_name SEED_SwinT --pred_idx 1 --loss_weights 1.0 1.0 --batch_size 16 --exp_name SEED_SwinT_CLCD --dataset CLCD --src_size 256 --lr 0.003 --max_epochs 200 --check_val_every_n_epoch 10 --early_stop 80 --warmup 3000 --no-comet --devices 1 --strategy auto --accelerator gpu

# srun python main.py --model_name SEED_SwinT --pred_idx 1 --loss_weights 1.0 1.0 --batch_size 12 --exp_name SEED_VIRCD0605 --dataset /home/dongsijun/scratch/VirCDData/0604 --src_size 512 --lr 0.0003 --min_lr 0.00003 --max_steps 120000 --val_check_interval 3000 --early_stop 80 --warmup 3000 --no-comet --devices 4 --strategy ddp --accelerator gpu

# srun python main.py --model_name Depth2CD --pred_idx 1 --loss_weights 1.0 1.0 --batch_size 8 --exp_name Depth2CD_LEVIR --dataset LEVIR-CD --src_size 1024 --crop_size 518 --lr 0.0001 --max_steps 40000 --val_check_interval 1000 --early_stop 80 --warmup 3000 --no-comet --devices 2 --strategy ddp --accelerator gpu

# srun python main.py --dataset SYSU-CD --model_type cd --model_arch SEED --model_name SNRNet --exp_name SNRNet_SYSU --max_steps 30000 --batch_size 12 --devices 2 --strategy ddp --accelerator gpu --src_size 256 --lr 0.0003 --no-comet

# srun python main.py --dataset WHUCD --model_type cd --model_arch SEED --model_name SNRNet --exp_name SNRNet_WHUCD --max_steps 30000 --batch_size 12 --devices 2 --strategy ddp --accelerator gpu --src_size 256 --lr 0.0003 --no-comet

srun python main.py --dataset PX-CLCD --model_type cd --model_arch SEED --model_name SNRNet --exp_name SNRNet_PXCLCD --max_steps 40000 --batch_size 12 --devices 2 --strategy ddp --accelerator gpu --src_size 256 --lr 0.0003 --no-comet

srun python main.py --dataset WaterCDPNG --model_type cd --model_arch SEED --model_name SNRNet --exp_name SNRNet_WaterCDPNG --max_steps 40000 --batch_size 12 --devices 2 --strategy ddp --accelerator gpu --src_size 512 --crop_size 256 --lr 0.0003 --no-comet