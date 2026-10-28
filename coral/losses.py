"""CORAL 损失（论文式 (1)(2)），以及目标域信息最大化的两项损失。"""
import torch.nn.functional as F


def covariance(x):
    """无偏样本协方差。先减 batch 均值再做 X^T X / (n-1)，与论文式 (2) 的展开写法等价。"""
    xc = x - x.mean(0, keepdim=True)
    return xc.t() @ xc / (x.shape[0] - 1)


def coral_loss(xs, xt):
    """L_CORAL = ||C_S - C_T||_F^2 / (4 d^2)。"""
    d = xs.shape[1]
    return (covariance(xs) - covariance(xt)).pow(2).sum() / (4 * d * d)


def entropy_loss(logits):
    """每个样本预测分布的熵再取平均。越小，目标样本离决策边界越远、预测越确定。"""
    logp = F.log_softmax(logits, 1)
    return -(logp.exp() * logp).sum(1).mean()


def diversity_loss(logits):
    """batch 平均预测分布的负熵。越小，各类别被预测得越均匀，防止熵最小化把样本都推到少数类上。
    与 entropy_loss 相加即信息最大化（单样本熵小、整体预测分布熵大）。
    """
    p = logits.softmax(1).mean(0)
    return (p * (p + 1e-8).log()).sum()
