#!/usr/bin/env bash

# python main.py --model_name EfficientCD --pred_idx 5 --loss_weights 0.2 0.2 0.2 0.2 0.2 1.0 --batch_size 16 --exp_name EfficientCD_WHUCD --dataset WHUCD --src_size 256 --lr 0.0003 --max_epochs 160 --check_val_every_n_epoch 2 --early_stop 200 --warmup 3000

# python main_step.py --model_name SEED_SwinT --pred_idx 1 --loss_weights 1.0 1.0 --batch_size 20 --exp_name SEED_SwinT_LEVIR --dataset LEVIR-CD --src_size 1024 --lr 0.0003 --max_steps 30000 --val_check_interval 1000 --early_stop 60 --warmup 3000

python main.py --dataset WHUCD --model_type cd --model_arch SEEDAdv --strategy ddp_find_unused_parameters_true --model_name ADVNet --exp_name ADVNet_WHUCD --max_steps 30000 --batch_size 8 --devices 2 --accelerator gpu --src_size 256 --lr 0.0003