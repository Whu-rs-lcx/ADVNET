import torch
import os
import cv2
import numpy as np
from torchmetrics.classification import Accuracy, ConfusionMatrix
import lightning as L
import torch.nn as nn
from utils.metric import CM2Metric
import torch.optim as optim
from change_detection import build_model
from torch.optim.lr_scheduler import *
import csv
import torch.nn.functional as F
from change_detection.utils.domain_genelization_loss import exchange_consistency_loss, jsd_loss, kl_loss
from itertools import chain

# 定义 GRL 层
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_adv * grad_output, None


# 1. BaseCD（保持不变）
class BaseCD(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        if args.resize_size > 1:
            self.example_input_array = torch.randn((2, 6, args.resize_size, args.resize_size))
        else:
            self.example_input_array = torch.randn((2, 6, args.crop_size, args.crop_size))
        # define parameters
        self.save_hyperparameters(args)

        self.hyparams = args
        if self.hyparams.resize_size > 1:
            self.if_slide = False
        else:
            self.if_slide = self.hyparams.src_size > self.hyparams.crop_size
        self.save_test_results = os.path.join(self.hyparams.exp_name+'_TrainingFiles', self.hyparams.save_test_results)

        # model training
        self.change_detection = build_model(self.hyparams.model_name)
        self.val_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=self.hyparams.num_classes)
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=self.hyparams.num_classes)
        if self.hyparams.loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.val_loss_epoch = []
        
        # prepare test output directory once
        self.test_output_dir = os.path.join(f"{self.hyparams.exp_name}_TrainingFiles",
                                            self.hyparams.save_test_results)
        os.makedirs(self.test_output_dir, exist_ok=True)

    def forward(self, x):
        xA, xB = x[:, :3], x[:, 3:]
        out = self.change_detection(xA, xB)
        return out

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x, y = batch['imgAB'], batch['lab']
        outs = self(x)
        loss = self._loss(outs, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, pathA, pathB = batch['imgAB'], batch['lab'], batch['pathA'], batch['pathB']
        logits, val_loss = self._infer(x, y)
        self.val_loss_epoch.append(val_loss)
        self.val_confusion_matrix.update(self._logits2preds(logits), y)

    def on_validation_epoch_end(self):
        cm = self.val_confusion_matrix.compute().cpu().numpy()
        metrics = CM2Metric(cm)
        val_loss_epoch = torch.mean(torch.stack(self.val_loss_epoch))
        self.log('val_loss', val_loss_epoch, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self.log_dict({
            'val_oa': metrics[0],
            'val_iou': metrics[4][1],
            'val_f1': metrics[5][1],
            'val_recall': metrics[6][1],
            'val_precision': metrics[7][1]
        }, prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
        self.val_confusion_matrix.reset()
        self.val_loss_epoch = []

    def test_step(self, batch: dict, batch_idx: int) -> None:
        x, y, pathA, pathB = batch['imgAB'], batch['lab'], batch['pathA'], batch['pathB']
        logits, test_loss = self._infer(x, y)
        self.test_confusion_matrix.update(self._logits2preds(logits), y)

        pred_np = self._logits2preds(logits).cpu().numpy().astype('uint8')
        for p, mask in zip(pathA, pred_np):
            base = os.path.splitext(os.path.basename(p))[0] + '.png'
            out_path = os.path.join(self.test_output_dir, base)
            cv2.imwrite(out_path, (mask * 255).astype('uint8'))

    def on_test_epoch_end(self):
        cm = self.test_confusion_matrix.compute().cpu().numpy()
        metrics = CM2Metric(cm)
        self.log_dict({
            'test_oa': metrics[0],
            'test_iou': metrics[4][1],
            'test_f1': metrics[5][1],
            'test_recall': metrics[6][1],
            'test_precision': metrics[7][1]
        }, prog_bar=True, sync_dist=True)

        names = 'OverallAccuracy, MeanF1, MeanIoU, Kappa, ClassIoU, ClassF1, ClassRecall, ClassPrecision'.split(', ')
        for i in range(len(names)):
            print(names[i]+': '+str(metrics[i]))
        paper_metrics = [metrics[0], metrics[4][1], metrics[5][1], metrics[6][1], metrics[7][1], metrics[1], metrics[2]]
        paper_metrics = [round(value * 100, 2) if isinstance(value, (int, float)) else [round(v * 100, 2) for v in value] for value in paper_metrics]

        names = 'OverallAccuracy, ClassIoU, ClassF1, ClassRecall, ClassPrecision, MeanF1, MeanIoU'.split(', ')
        csv_filename = os.path.join(self.hyparams.exp_name+'_TrainingFiles', 'overall_result.csv')
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(names)
            csv_writer.writerow(paper_metrics)
        print(f'Data saved to {csv_filename}')

        self.test_confusion_matrix.reset()

    def _logits2preds(self, logits):
        if self.hyparams.loss_type == 'ce':
            preds = logits.argmax(dim=1)
        else:
            preds = torch.sigmoid(logits).round()
        return preds

    def _loss(self, outs, y, state='train'):
        if state == 'train':
            if self.hyparams.loss_type == 'ce':
                loss = sum(w * self.criterion(o, y.long()) for w, o in zip(self.hyparams.loss_weights, outs))
            else:
                loss = sum(w * self.criterion(o, y.unsqueeze(1).float()) for w, o in zip(self.hyparams.loss_weights, outs))
        else:
            if self.hyparams.loss_type == 'ce':
                loss = self.criterion(outs, y.long())
            else:
                loss = self.criterion(outs, y.unsqueeze(1).float())
        return loss

    def _infer(self, x, y):
        if self.if_slide:
            logits, val_loss = self._slide_inference(x, y)
            return logits, val_loss
        else:
            outs = self(x)
            val_loss = self._loss(outs, y)
            logits = outs[self.hyparams.pred_idx]
            return logits, val_loss

    def _slide_inference(self, inputs, labels):
        h_crop = w_crop = self.hyparams.crop_size
        h_stride = w_stride = getattr(self.hyparams, "overlap", h_crop // 2)
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.hyparams.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                outs = self(crop_img)
                crop_seg_logit = outs[self.hyparams.pred_idx]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat
        val_loss = self._loss(seg_logits, labels, state='val')
        return seg_logits, val_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hyparams.lr, weight_decay=1e-4)
        
        def lr_lambda(step: int) -> float:
            warmup = self.hyparams.warmup
            power = 3.0
            if step < warmup:
                raw = float(step) / float(max(1, warmup))
            else:
                progress = float(step - warmup) / float(max(1, self.hyparams.max_steps - warmup))
                raw = max(0.0, (1.0 - progress) ** power)
            min_factor = self.hyparams.min_lr / self.hyparams.lr
            return max(raw, min_factor)
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "strict": False,
                "frequency": 1,
                "name": None
            },
        }


# 2. SEED（保持不变）
class SEED(BaseCD):
    def __init__(self, args):
        super().__init__(args)

    def _loss(self, outs, y, state='train'):
        if state == 'train':
            loss = sum(w * self.criterion(o, y.long())
                       for w, o in zip(self.hyparams.loss_weights, outs))
        else:
            loss = self.criterion(outs, y.long())
        return loss / 2.0

    def _infer(self, x, y):
        if self.if_slide and self.hyparams.model_type != 'dgcd':
            logits, val_loss = self._slide_inference(x, y)
            return logits, val_loss
        else:
            outs = self(x)
            val_loss = self._loss(outs, y)
            logits = (outs[0] + outs[1]) / 2.0
            return logits, val_loss

    def _slide_inference(self, inputs, labels):
        h_crop = w_crop = self.hyparams.crop_size
        h_stride = w_stride = getattr(self.hyparams, "overlap", h_crop // 2)
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.hyparams.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                outs = self(crop_img)
                crop_seg_logit = (outs[0] + outs[1]) / 2.0
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat
        val_loss = self._loss(seg_logits, labels, state='val')
        return seg_logits, val_loss



class SEED_DG(BaseCD):
    """
    SEED Lightning 模型：
    - 继承自 BaseCD，复用数据加载、训练/验证/测试流程
    - 只需要在 __init__ 中替换 change_detection 网络即可
    """
    def __init__(self, args):
        # 调用父类，完成超参保存、confusion matrix、criterion 等初始化
        super().__init__(args)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x, y = batch['imgAB'], batch['lab']
        outs = self(x)
        outs2 = self(x)
        loss = self._loss(outs, y)
        loss2 = self._loss(outs2, y)
        pixel_loss = (loss + loss2) / 2.0

        consistency_loss = (kl_loss(outs[0], outs2[0]) + kl_loss(outs[1], outs2[1])) / 2.0

        consistency_weight = 0.05
        rampup_duration = 1000

        progress = self.global_step / rampup_duration
        current_weight = consistency_weight * min(1.0, progress)

        # 4. 计算总损失
        total_loss = pixel_loss + current_weight * consistency_loss

        # 记录训练损失
        self.log('train_pixel_loss', pixel_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_consistency_loss', consistency_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def _loss(self, outs, y, state='train'):
        """Compute the loss for the current batch."""
        if state == 'train':
            loss = sum(w * self.criterion(o, y.long())
                        for w, o in zip(self.hyparams.loss_weights, outs))
        else:
            loss = self.criterion(outs, y.long())
        return loss/2.0

    def _infer(self, x, y):
        """Run either sliding-window or single-pass inference."""

        if self.if_slide and self.hyparams.model_type != 'dgcd':
            logits, val_loss = self._slide_inference(x, y)
            return logits, val_loss
        else:
            outs = self(x)
            val_loss =  self._loss(outs, y)
            logits = (outs[0]+outs[1])/2.0
            return logits, val_loss

    def _slide_inference(self, inputs, labels):

        h_crop = w_crop = self.hyparams.crop_size
        h_stride = w_stride = getattr(self.hyparams, "overlap", h_crop // 2)
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.hyparams.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]

                outs = self(crop_img)
                crop_seg_logit = (outs[0]+outs[1])/2.0

                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        val_loss = self._loss(seg_logits, labels, state='val')

        return seg_logits, val_loss
    


# 3. 新增 BaseCDAdv（基于 BaseCD，添加对抗性风格对齐）
class BaseCDAdv(BaseCD):
    def __init__(self, args):
        super().__init__(args)
        self.automatic_optimization = False
        # 修改 example_input_array 为两个张量
        size = args.resize_size if args.resize_size > 1 else args.crop_size
        self.example_input_array = (torch.randn(2, 3, size, size), torch.randn(2, 3, size, size))
        # 添加对抗性损失权重
        self.lambda_adv = args.lambda_adv if hasattr(args, 'lambda_adv') else 0.1

    def forward(self, x_A, x_B):
        # 返回变化检测输出和多尺度特征
        out, z_A, z_B = self.change_detection(x_A, x_B)  # 假设 build_model 返回 (out, z_A, z_B)
        return out, z_A, z_B

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        opt_F, opt_D = self.optimizers()
        x_A, x_B, y, d_A, d_B = batch['inputA'], batch['inputB'], batch['lab'], batch['d_A'], batch['d_B']
        outs, z_A, z_B = self(x_A, x_B)
        
        # 更新特征提取器和解码器
        opt_F.zero_grad()
        loss_task = self._loss(outs, y)
        loss_adv_F = 0
        for i in range(len(z_A)):
            p_A = self.change_detection.discriminators[i](GRL.apply(z_A[i], self.lambda_adv))
            p_B = self.change_detection.discriminators[i](GRL.apply(z_B[i], self.lambda_adv))
            loss_adv_F += F.binary_cross_entropy_with_logits(p_A, 1 - d_A.unsqueeze(-1)) + F.binary_cross_entropy_with_logits(p_B, 1 - d_B.unsqueeze(-1))
        loss_total = loss_task + self.lambda_adv * loss_adv_F
        self.manual_backward(loss_total)
        opt_F.step()
        
        # 更新域鉴别器
        if batch_idx % 2 == 0:
            opt_D.zero_grad()
            loss_adv_D = 0
            for i in range(len(z_A)):
                p_A = self.change_detection.discriminators[i](z_A[i].detach())
                p_B = self.change_detection.discriminators[i](z_B[i].detach())
                loss_adv_D += F.binary_cross_entropy_with_logits(p_A, d_A.unsqueeze(-1)) + F.binary_cross_entropy_with_logits(p_B, d_B.unsqueeze(-1))
            self.manual_backward(loss_adv_D)
            opt_D.step()
        
        # 日志
        self.log('train_loss_task', loss_task, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_adv_D', loss_adv_D, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_total', loss_total, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return {'loss': loss_total, 'loss_adv_D': loss_adv_D.detach()}


    def validation_step(self, batch, batch_idx):
        x_A, x_B, y, pathA, pathB = batch['inputA'], batch['inputB'], batch['lab'], batch['pathA'], batch['pathB']
        logits, val_loss = self._infer(x_A, x_B, y)
        self.val_loss_epoch.append(val_loss)
        self.val_confusion_matrix.update(self._logits2preds(logits), y)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        x_A, x_B, y, pathA, pathB = batch['inputA'], batch['inputB'], batch['lab'], batch['pathA'], batch['pathB']
        logits, test_loss = self._infer(x_A, x_B, y)
        self.test_confusion_matrix.update(self._logits2preds(logits), y)
        pred_np = self._logits2preds(logits).cpu().numpy().astype('uint8')
        for p, mask in zip(pathA, pred_np):
            base = os.path.splitext(os.path.basename(p))[0] + '.png'
            out_path = os.path.join(self.test_output_dir, base)
            cv2.imwrite(out_path, (mask * 255).astype('uint8'))

    def _infer(self, x_A, x_B, y):
        if self.if_slide:
            logits, val_loss = self._slide_inference(x_A, x_B, y)
            return logits, val_loss
        else:
            outs, _, _ = self(x_A, x_B)
            val_loss = self._loss(outs, y)
            logits = outs[self.hyparams.pred_idx]
            return logits, val_loss

    def _slide_inference(self, x_A, x_B, labels):
        h_crop = w_crop = self.hyparams.crop_size
        h_stride = w_stride = getattr(self.hyparams, "overlap", h_crop // 2)
        batch_size, _, h_img, w_img = x_A.size()
        out_channels = self.hyparams.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = x_A.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = x_A.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img_A = x_A[:, :, y1:y2, x1:x2]
                crop_img_B = x_B[:, :, y1:y2, x1:x2]
                outs, _, _ = self(crop_img_A, crop_img_B)
                crop_seg_logit = outs[self.hyparams.pred_idx]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat
        val_loss = self._loss(seg_logits, labels, state='val')
        return seg_logits, val_loss

    def configure_optimizers(self):
        optimizer_F = optim.AdamW(
            list(self.change_detection.model.parameters()) +
            list(self.change_detection.fpn.parameters()) +
            list(self.change_detection.decode_layersA.parameters()) +
            list(self.change_detection.decode_conv.parameters()) +
            list(self.change_detection.conv_seg.parameters()),
            lr=self.hyparams.lr,
            weight_decay=1e-4
        )
        optimizer_D = optim.AdamW(
            list(self.change_detection.discriminators.parameters()),
            lr=self.hyparams.lr * 0.5,
            weight_decay=1e-4
        )
        
        def lr_lambda(step: int) -> float:
            warmup = self.hyparams.warmup
            power = 3.0
            if step < warmup:
                raw = float(step) / float(max(1, warmup))
            else:
                progress = float(step - warmup) / float(max(1, self.hyparams.max_steps - warmup))
                raw = max(0.0, (1.0 - progress) ** power)
            min_factor = self.hyparams.min_lr / self.hyparams.lr
            return max(raw, min_factor)
        
        scheduler_F = LambdaLR(optimizer_F, lr_lambda)
        scheduler_D = LambdaLR(optimizer_D, lr_lambda)
        
        return [
            {"optimizer": optimizer_F, "lr_scheduler": {"scheduler": scheduler_F, "interval": "step", "strict": False, "frequency": 1}},
            {"optimizer": optimizer_D, "lr_scheduler": {"scheduler": scheduler_D, "interval": "step", "strict": False, "frequency": 1}}
        ]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.lambda_adv = min(0.5, self.lambda_adv + 0.001)


# 4. 新增 SEEDAdv（基于 SEED，添加对抗性风格对齐）
class SEEDAdv(BaseCD):
    def __init__(self, args):
        super().__init__(args)
        self.automatic_optimization = False
        # 修改 example_input_array 为两个张量
        size = args.resize_size if args.resize_size > 1 else args.crop_size
        self.example_input_array = (torch.randn(2, 3, size, size), torch.randn(2, 3, size, size))
        # 添加对抗性损失权重
        self.lambda_adv = args.lambda_adv if hasattr(args, 'lambda_adv') else 0.1

    def forward(self, x_A, x_B):
        # 返回变化检测输出和多尺度特征
        out, z_A, z_B = self.change_detection(x_A, x_B)  # 假设 build_model 返回 (out, z_A, z_B)
        return out, z_A, z_B

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        opt_F, opt_D = self.optimizers()
        x_A, x_B, y, d_A, d_B = batch['inputA'], batch['inputB'], batch['lab'], batch['d_A'], batch['d_B']
        outs, z_A, z_B = self(x_A, x_B)
        
        # 更新特征提取器和解码器
        opt_F.zero_grad()
        loss_task = self._loss(outs, y)
        loss_adv_F = 0
        for i in range(len(z_A)):
            p_A = self.change_detection.discriminators[i](GRL.apply(z_A[i], self.lambda_adv))
            p_B = self.change_detection.discriminators[i](GRL.apply(z_B[i], self.lambda_adv))
            loss_adv_F += F.binary_cross_entropy_with_logits(p_A, 1 - d_A.unsqueeze(-1)) + F.binary_cross_entropy_with_logits(p_B, 1 - d_B.unsqueeze(-1))
        loss_total = loss_task + self.lambda_adv * loss_adv_F
        self.manual_backward(loss_total)
        parameters = chain(
        self.change_detection.model.parameters(),
        self.change_detection.fpn.parameters(),
        self.change_detection.decode_layersA.parameters(),
        self.change_detection.decode_conv.parameters(),
        self.change_detection.conv_seg.parameters()
    )
        torch.nn.utils.clip_grad_norm_(
            parameters=[p for p in parameters if p.requires_grad],
            max_norm=1.0
        )
        opt_F.step()
        
        # 更新域鉴别器
        opt_D.zero_grad()
        loss_adv_D = 0
        for i in range(len(z_A)):
            p_A = self.change_detection.discriminators[i](z_A[i].detach())
            p_B = self.change_detection.discriminators[i](z_B[i].detach())
            loss_adv_D += F.binary_cross_entropy_with_logits(p_A, d_A.unsqueeze(-1)) + F.binary_cross_entropy_with_logits(p_B, d_B.unsqueeze(-1))
        self.manual_backward(loss_adv_D)
        torch.nn.utils.clip_grad_norm_(
        parameters=[p for p in self.change_detection.discriminators.parameters() if p.requires_grad],
        max_norm=1.0
    )
        opt_D.step()
        
        # 日志
        self.log('train_loss_task', loss_task, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_adv_D', loss_adv_D, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_total', loss_total, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return {'loss': loss_total, 'loss_adv_D': loss_adv_D.detach()}

    def validation_step(self, batch, batch_idx):
        x_A, x_B, y, pathA, pathB = batch['inputA'], batch['inputB'], batch['lab'], batch['pathA'], batch['pathB']
        logits, val_loss = self._infer(x_A, x_B, y)
        self.val_loss_epoch.append(val_loss)
        self.val_confusion_matrix.update(self._logits2preds(logits), y)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        x_A, x_B, y, pathA, pathB = batch['inputA'], batch['inputB'], batch['lab'], batch['pathA'], batch['pathB']
        logits, test_loss = self._infer(x_A, x_B, y)
        self.test_confusion_matrix.update(self._logits2preds(logits), y)
        pred_np = self._logits2preds(logits).cpu().numpy().astype('uint8')
        for p, mask in zip(pathA, pred_np):
            base = os.path.splitext(os.path.basename(p))[0] + '.png'
            out_path = os.path.join(self.test_output_dir, base)
            cv2.imwrite(out_path, (mask * 255).astype('uint8'))

    def _loss(self, outs, y, state='train'):
        if state == 'train':
            loss = sum(w * self.criterion(o, y.long())
                       for w, o in zip(self.hyparams.loss_weights, outs))
        else:
            loss = self.criterion(outs, y.long())
        return loss / 2.0

    def _infer(self, x_A, x_B, y):
        if self.if_slide and self.hyparams.model_type != 'dgcd':
            logits, val_loss = self._slide_inference(x_A, x_B, y)
            return logits, val_loss
        else:
            outs, _, _ = self(x_A, x_B)
            val_loss = self._loss(outs, y)
            logits = (outs[0] + outs[1]) / 2.0
            return logits, val_loss

    def _slide_inference(self, x_A, x_B, labels):
        h_crop = w_crop = self.hyparams.crop_size
        h_stride = w_stride = getattr(self.hyparams, "overlap", h_crop // 2)
        batch_size, _, h_img, w_img = x_A.size()
        out_channels = self.hyparams.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = x_A.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = x_A.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img_A = x_A[:, :, y1:y2, x1:x2]
                crop_img_B = x_B[:, :, y1:y2, x1:x2]
                outs, _, _ = self(crop_img_A, crop_img_B)
                crop_seg_logit = (outs[0] + outs[1]) / 2.0
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat
        val_loss = self._loss(seg_logits, labels, state='val')
        return seg_logits, val_loss

    def configure_optimizers(self):
        optimizer_F = optim.AdamW(
            list(self.change_detection.model.parameters()) +
            list(self.change_detection.fpn.parameters()) +
            list(self.change_detection.decode_layersA.parameters()) +
            list(self.change_detection.decode_conv.parameters()) +
            list(self.change_detection.conv_seg.parameters()),
            lr=self.hyparams.lr,
            weight_decay=1e-4
        )
        optimizer_D = optim.AdamW(
            list(self.change_detection.discriminators.parameters()),
            lr=self.hyparams.lr * 0.5,
            weight_decay=1e-4
        )
        
        def lr_lambda(step: int) -> float:
            warmup = self.hyparams.warmup
            power = 3.0
            if step < warmup:
                raw = float(step) / float(max(1, warmup))
            else:
                progress = float(step - warmup) / float(max(1, self.hyparams.max_steps - warmup))
                raw = max(0.0, (1.0 - progress) ** power)
            min_factor = self.hyparams.min_lr / self.hyparams.lr
            return max(raw, min_factor)
        
        scheduler_F = LambdaLR(optimizer_F, lr_lambda)
        scheduler_D = LambdaLR(optimizer_D, lr_lambda)
        
        return [
            {"optimizer": optimizer_F, "lr_scheduler": {"scheduler": scheduler_F, "interval": "step", "strict": False, "frequency": 1}},
            {"optimizer": optimizer_D, "lr_scheduler": {"scheduler": scheduler_D, "interval": "step", "strict": False, "frequency": 1}}
        ]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.lambda_adv = min(0.3, 0.1 + (0.3 - 0.1) * self.global_step / self.hyparams.max_steps)