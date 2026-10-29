"""CORAL 损失（论文式 (1)(2)）、目标域信息最大化的两项损失、类条件对齐。"""
import torch
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


def mean_loss(xs, xt):
    """||μ_S - μ_T||^2 / (4 d)，与 CORAL 一样按“每个元素差的平方 / 4”归一化。"""
    return (xs.mean(0) - xt.mean(0)).pow(2).sum() / (4 * xs.shape[1])


class ClassConditionalAlignment:
    """按类别对齐源域与目标域特征的均值和 / 或协方差。源域用真实标签，目标域用伪标签。

    batch 128、31 类时每类平均只有约 4 个样本，单靠 batch 估计协方差很不可靠，所以每个域、每个类
    维护一个特征队列（存最近 queue_size 个样本的特征，不回传梯度），与当前 batch 中该类的特征拼起来估计；
    梯度只通过当前 batch 回传。某类在两个域里的样本总数都不少于 min_count、且当前 batch 至少有一个
    该类样本时才计入，损失取所有计入类别的平均。
    """

    def __init__(self, num_classes, dim, device, queue_size=32, min_count=4, terms='both'):
        assert terms in ('both', 'mean', 'cov'), terms
        self.num_classes, self.queue_size, self.min_count, self.terms = num_classes, queue_size, min_count, terms
        self.queues = [[torch.zeros(0, dim, device=device) for _ in range(num_classes)] for _ in range(2)]

    def _class_loss(self, xs, xt):
        loss = 0.0
        if self.terms in ('both', 'mean'):
            loss = loss + mean_loss(xs, xt)
        if self.terms in ('both', 'cov'):
            loss = loss + coral_loss(xs, xt)
        return loss

    def __call__(self, fs, ys, ft, yt):
        """fs / ft：两个域当前 batch 的特征；ys：源域标签；yt：目标域伪标签（不参与的样本为 -1）。
        返回 (损失, 计入的类别数)。调用后把当前 batch 的特征（detach）放进队列。
        """
        losses = []
        for c in range(self.num_classes):
            cur = [fs[ys == c], ft[yt == c]]
            full = [torch.cat([q[c], x]) for q, x in zip(self.queues, cur)]
            if min(len(x) for x in full) >= self.min_count and max(len(x) for x in cur) > 0:
                losses.append(self._class_loss(*full))
            for q, x in zip(self.queues, cur):
                if len(x):
                    q[c] = torch.cat([q[c], x.detach()])[-self.queue_size:]
        if not losses:
            return fs.sum() * 0.0, 0
        return torch.stack(losses).mean(), len(losses)
