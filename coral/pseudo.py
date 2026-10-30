"""不依赖 softmax 置信度的目标域伪标签：特征空间聚类（SHOT 的做法）。"""
import torch
import torch.nn.functional as F


def centroid_labels(feats, logits, rounds=1):
    """按特征空间里最近的类中心给每个目标样本打伪标签。

    1. 特征做 L2 归一化（用余弦相似度）；
    2. 用 softmax 概率加权平均得到各类初始中心；
    3. 每个样本分给最近的中心，再用这些硬标签重算中心、重新分配，共 rounds 轮。
    某类没有分到样本时保留它上一轮的中心。只用特征和预测，不用任何标签。
    """
    f = F.normalize(feats.float(), dim=1)
    p = logits.float().softmax(1)
    centers = F.normalize(p.t() @ f, dim=1)
    labels = (f @ centers.t()).argmax(1)
    for _ in range(rounds):
        onehot = F.one_hot(labels, p.shape[1]).float()
        counts = onehot.sum(0)
        new = F.normalize(onehot.t() @ f, dim=1)
        centers = torch.where(counts[:, None] > 0, new, centers)
        labels = (f @ centers.t()).argmax(1)
    return labels
