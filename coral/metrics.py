"""评估指标：准确率、平衡准确率、ECE、预测熵、SND、有效秩、全量与 batch 的 CORAL 距离。"""
import torch
import torch.nn.functional as F

from .losses import coral_loss


def accuracy(logits, labels):
    return float((logits.argmax(1) == labels).float().mean())


def confusion_matrix(pred, labels, num_classes):
    """行为真实类别，列为预测类别。"""
    return torch.bincount(labels * num_classes + pred, minlength=num_classes ** 2).view(num_classes, num_classes)


def per_class_accuracy(cm):
    """各类召回率；某类没有样本时为 nan。"""
    return cm.diag().double() / cm.sum(1).double()


def balanced_accuracy(cm):
    return float(torch.nanmean(per_class_accuracy(cm)))


def ece(probs, labels, n_bins=15):
    """期望校准误差，等宽置信度区间 (m-1)/M < conf <= m/M。"""
    conf, pred = probs.max(1)
    correct = (pred == labels).double()
    bins = ((conf * n_bins).ceil().long() - 1).clamp(0, n_bins - 1)
    total = 0.0
    for m in range(n_bins):
        mask = bins == m
        if mask.any():
            total += float(mask.double().mean() * (correct[mask].mean() - conf[mask].double().mean()).abs())
    return total


def mean_entropy(logits):
    logp = F.log_softmax(logits, 1)
    return float(-(logp.exp() * logp).sum(1).mean())


def snd(logits, temperature=0.05):
    """Soft Neighborhood Density：softmax 输出 L2 归一化后两两内积 / 温度，去掉对角线逐行 softmax，
    对每行求熵再平均。数值越高越好。"""
    p = F.normalize(logits.softmax(1), dim=1)
    sim = p @ p.t() / temperature
    sim.fill_diagonal_(float('-inf'))
    q = sim.softmax(1)
    return float(-torch.special.xlogy(q, q).sum(1).mean())


def effective_rank(x):
    """erank = exp(-Σ p_k log p_k)，p_k 为中心化特征矩阵奇异值的归一化。"""
    s = torch.linalg.svdvals(x - x.mean(0, keepdim=True)).double()
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def coral_distance(fs, ft):
    """用全量特征的协方差算 CORAL 距离（双精度）。"""
    return float(coral_loss(fs.double(), ft.double()))


def batch_coral_distance(fs, ft, batch_size, n_batches, gen):
    """从全量特征中随机抽 n_batches 对 batch，计算平均 CORAL 距离。gen 为 CPU 上的 torch.Generator。"""
    vals = []
    for _ in range(n_batches):
        i = torch.randperm(len(fs), generator=gen)[:batch_size].to(fs.device)
        j = torch.randperm(len(ft), generator=gen)[:batch_size].to(ft.device)
        vals.append(coral_loss(fs[i].double(), ft[j].double()))
    return float(torch.stack(vals).mean())
