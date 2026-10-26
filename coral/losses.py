"""CORAL 损失（论文式 (1)(2)）。"""


def covariance(x):
    """无偏样本协方差。先减 batch 均值再做 X^T X / (n-1)，与论文式 (2) 的展开写法等价。"""
    xc = x - x.mean(0, keepdim=True)
    return xc.t() @ xc / (x.shape[0] - 1)


def coral_loss(xs, xt):
    """L_CORAL = ||C_S - C_T||_F^2 / (4 d^2)。"""
    d = xs.shape[1]
    return (covariance(xs) - covariance(xt)).pow(2).sum() / (4 * d * d)
