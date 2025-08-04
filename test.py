import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
from ModelFactory import BaseCD, SEED
from utils.dataset import CDTXTDataset
from utils.utils import get_best_model_checkpoint, define_transforms
import argparse
from torch.utils.data import DataLoader
from utils.metric import CM2Metric
from torchmetrics import ConfusionMatrix
import yaml


def update_args_from_yaml(args: argparse.Namespace, yaml_path: str) -> argparse.Namespace:
    """
    用 yaml 文件里的配置更新已有的 args Namespace。
    如果 yaml 中有 args 没有的字段，则会动态添加；如果有的字段则覆盖原有值。
    """
    # 1. 读取 yaml 配置
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 2. 用 cfg 中的键值更新 args
    for key, val in cfg.items():
        setattr(args, key, val)

    return args


def main():
    parser = argparse.ArgumentParser(description='Train a model for change detection')

    # ===== 1. 基本与运行模式 =====
    parser.add_argument('--dataset',        type=str, default='LEVIR-CD', help='Path to dataset root')
    parser.add_argument('--model_name',     type=str, default='EfficientCD', help='Model name')
    parser.add_argument('--mode',           type=str, default='train', help='Mode of the program (train/test)')
    parser.add_argument('--resume_path',    type=str, default=None,  help='Path to resume from checkpoint')
    parser.add_argument('--exp_name',       type=str, default='LEVIR', help='Experiment name')

    # ===== 2. 数据加载与预处理 =====
    parser.add_argument('--batch_size',     type=int, default=16,   help='Batch size for training')
    parser.add_argument('--num_workers',    type=int, default=8,    help='Number of workers for data loading')
    parser.add_argument('--src_size',       type=int, default=256,  help='Source size for input images')
    parser.add_argument('--crop_size',      type=int, default=256,  help='Crop size for input images')
    parser.add_argument('--overlap',        type=int, default=128,  help='Overlap size for sliding window')

    # ===== 3. 模型输出与类别 =====
    parser.add_argument('--pred_idx',       type=int, default=5,    help='GPU ID to use / index of output branch')
    parser.add_argument('--num_classes',    type=int, default=2,    help='Number of classes for model output')

    # ===== 4. 损失与优化超参 =====
    parser.add_argument('--loss_type',      type=str,   default='ce',    help='Loss type (bce/ce/focal)')
    parser.add_argument('--loss_weights',   type=float, nargs='+', default=[0.2, 0.2, 0.2, 0.2, 0.2, 1.0], help='各个 loss 的权重，例: --loss_weights 0.5 1.0 2.0')
    parser.add_argument('--lr',             type=float, default=3e-4,  help='Learning rate')
    parser.add_argument('--min_lr',         type=float, default=1e-5,  help='Minimum learning rate')
    parser.add_argument('--warmup',         type=int,   default=3000,  help='Number of steps for warmup')

    # ===== 5. 训练进度控制 =====
    parser.add_argument('--max_epochs',             type=int, default=1200,    help='Number of epochs to train (-1 for unlimited)')
    parser.add_argument('--max_steps',              type=int, default=-1, help='Number of steps to train')
    parser.add_argument('--early_stop',             type=int, default=10,    help='Patience for early stopping')

    # ===== 6. 验证与可视化 =====
    parser.add_argument('--check_val_every_n_epoch', type=int, default=20,  help='Check validation every n epochs')
    parser.add_argument('--val_check_interval',      type=int, default=None,   help='Check validation every n iterations')
    parser.add_argument('--val_vis_num',             type=int, default=0,     help='Number of validation images to visualize')

    # ===== 7. 日志与结果保存 =====
    parser.add_argument('--comet',            action=argparse.BooleanOptionalAction, default=True,  help='Use Comet logger')
    parser.add_argument('--save_test_results', type=str, default='test_results', help='Path to save test results')

    # ===== 8. 分布式与硬件配置 =====
    parser.add_argument('--accelerator',  type=str, default='gpu',   help='Accelerator for training')
    parser.add_argument('--devices',      type=str, default=1,     help='Number of devices for training')
    parser.add_argument('--strategy',     type=str, default='auto',  help='Strategy for distributed training')
    parser.add_argument('--precision',    type=int, default=16,      help='Precision for training (16 or 32)')

    # ===== 9. 测试阶段 =====
    parser.add_argument('--hparams_file',  type=str, default=None,   help='Path to hparams file for testing')
    parser.add_argument('--test_mode',          type=str, default='test_loader', help='Mode of the program (train_loader/val_loader/test_loader)')

    args = parser.parse_args()

    if os.path.exists(args.dataset) is False:
        args.dataset = os.path.join(os.environ.get("CDPATH"), args.dataset)

    args = update_args_from_yaml(args, args.hparams_file)

    # 打印一下看更新结果
    print("最终训练配置：")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # 加载transform（建议用和训练/验证一致的数据增强方式）
    _, test_transform = define_transforms(if_crop=args.src_size>args.crop_size, crop_size=args.crop_size)

    print('Test Data Augmentation Information:')
    print(test_transform)

    # 载入数据
    train_dataset = CDTXTDataset(os.path.join(args.dataset, 'train.txt'), transform=test_transform)
    val_dataset = CDTXTDataset(os.path.join(args.dataset, 'val.txt'), transform=test_transform)
    test_dataset = CDTXTDataset(os.path.join(args.dataset, 'test.txt'), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    loader_dict = {
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "test_loader":  test_loader,
    }

    if args.resume_path is not None:
        best_checkpoint_path = args.resume_path
    else:
        best_checkpoint_path = get_best_model_checkpoint(args.exp_name+'_TrainingFiles')

    if 'SEED' in args.model_name:
        model = SEED.load_from_checkpoint(best_checkpoint_path, args=args, strict=True)
    else:
        model = BaseCD.load_from_checkpoint(best_checkpoint_path, args=args, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    save_dir = os.path.join(args.exp_name+'_TrainingFiles', args.save_test_results)
    os.makedirs(save_dir, exist_ok=True)
    cm_func = ConfusionMatrix(task='multiclass', num_classes=args.num_classes).to(device)
    with torch.no_grad():
        for batch in tqdm(loader_dict[args.test_mode], desc='Inference'):
            imgs = batch['imgAB'].to(device)  # shape: (B, C, H, W)
            paths = batch['pathA']  # 用于命名保存
            labels = batch['lab'].to(device)  # shape: (B, H, W)

            # 判断是否需要滑窗推理
            if model.hyparams.src_size > model.hyparams.crop_size:
                # 这里不需要label，所以可以随便传一个全0的
                pred_logits, _ = model._slide_inference(imgs, labels)
            else:
                outs = model(imgs)
                pred_logits = outs[model.hyparams.pred_idx]

            pred_mask = pred_logits.argmax(dim=1)
            cm_func.update(pred_mask, labels)

            pred_mask_np = pred_mask.cpu().numpy().astype('uint8')
            for path, mask in zip(paths, pred_mask_np):
                name = os.path.splitext(os.path.basename(path))[0] + '.png'
                save_path = os.path.join(save_dir, name)
                cv2.imwrite(save_path, mask * 255)

    # 推理结束后，计算总混淆矩阵
    conf_mat = cm_func.compute()  # shape: (num_classes, num_classes)
    print('总混淆矩阵:\n', conf_mat.cpu().numpy())

    # 计算各项指标
    metrics = CM2Metric(conf_mat.cpu().numpy())
    names = 'OverallAccuracy, MeanF1, MeanIoU, Kappa, ClassIoU, ClassF1, ClassRecall, ClassPrecision'.split(', ')
    for i in range(len(names)):
        print(names[i]+': '+str(metrics[i]))


if __name__ == '__main__':
    main()
