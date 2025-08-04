import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


def kl_loss(outA, outB):
    # 计算对称 KL
    logpA = F.log_softmax(outA, dim=1)
    pB    = torch.softmax(outB, dim=1)
    logpB = F.log_softmax(outB, dim=1)
    pA    = torch.softmax(outA, dim=1)

    loss_kl = F.kl_div(logpA, pB, reduction='batchmean') \
            + F.kl_div(logpB, pA, reduction='batchmean')
    return loss_kl



def jsd_loss(preds_list, epsilon=1e-12):
    """
    一个“偏执模式”下，为最大化数值稳定性而设计的JSD损失函数。
    """
    all_log_probs = []
    all_probs = []

    # 步骤 1: 稳定地计算每个预测的log-probabilities和probabilities
    for p_logits in preds_list:
        # 使用log_softmax获得log-probabilities，这是数值最稳定的方式
        log_p = F.log_softmax(p_logits, dim=1)
        all_log_probs.append(log_p)
        # 从log-probabilities通过exp获得probabilities
        all_probs.append(torch.exp(log_p))

    # 步骤 2: 计算平均概率分布 M
    M = torch.stack(all_probs, dim=0).mean(dim=0)
    
    # 步骤 3: 偏执地计算log(M)，防止任何可能的log(0)
    # clamp将所有小于epsilon的值强制替换为epsilon
    log_M = torch.clamp(M, min=epsilon).log()

    # 步骤 4: 计算KL散度
    total_kl_divergence = 0.0
    for log_p in all_log_probs:
        # PyTorch的kl_div期望输入是 (log-probabilities, probabilities)
        # 注意这里的参数顺序，第一个是log(M)，第二个是exp(log_p)
        # target需要是概率，所以我们用回 all_probs
        P = torch.exp(log_p)
        kl_div = F.kl_div(log_M, P, reduction='batchmean', log_target=False)
        total_kl_divergence += kl_div

    return total_kl_divergence / len(preds_list)


def exchange_consistency_loss(outs: list[torch.Tensor],
                              outs1: list[torch.Tensor],
                              symmetric: bool = True,
                              reduction: str = 'batchmean') -> torch.Tensor:
    """
    计算两次交换分支的 KL 一致性损失。

    参数:
      outA, outB:   第一次交换后的 logits，形状 [B, C, H, W]
      out1A, out1B: 第二次交换后的 logits，形状同上
      symmetric:    是否使用对称 KL（双向平均），默认为 True
      reduction:    reduction 方式，'batchmean' 或 'sum' / 'mean'

    返回:
      loss: 标量 tensor，交换一致性损失
    """
    # 先把 logits 转为概率分布

    outA, outB = outs
    out1A, out1B = outs1
    pA  = F.softmax(outA,  dim=1).detach()  # detach 掉第一分支，防止双向同时更新
    pB  = F.softmax(outB,  dim=1).detach()
    log1A = F.log_softmax(out1A, dim=1)
    log1B = F.log_softmax(out1B, dim=1)

    # 单向 KL：让第二次分支 向 第一次分支 靠
    kl_A_1A = F.kl_div(log1A, pA, reduction=reduction)
    kl_B_1B = F.kl_div(log1B, pB, reduction=reduction)
    loss = (kl_A_1A + kl_B_1B) * 0.5

    if symmetric:
        # 反向：让第一次分支 向 第二次分支 靠
        logA  = F.log_softmax(outA,  dim=1)
        logB  = F.log_softmax(outB,  dim=1)
        p1A   = F.softmax(out1A, dim=1).detach()
        p1B   = F.softmax(out1B, dim=1).detach()

        kl_1A_A = F.kl_div(logA, p1A, reduction=reduction)
        kl_1B_B = F.kl_div(logB, p1B, reduction=reduction)
        loss = (loss + 0.5*(kl_1A_A + kl_1B_B)) * 0.5

    return loss
