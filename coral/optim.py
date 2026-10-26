"""Caffe 写法的 SGD。"""
import torch


class CaffeSGD(torch.optim.Optimizer):
    """Caffe 的 SGD 把学习率乘进动量：v = m·v + lr·(g + wd·w)，w ← w − v。

    torch.optim.SGD 是 v = m·v + (g + wd·w)，w ← w − lr·v。学习率不变时两者完全等价；
    学习率阶梯下降时，Caffe 的旧动量仍带着原来的大步长逐渐衰减，PyTorch 则立刻按新学习率缩小。
    原文用 step 调度（每 500 步 ×0.1），所以对齐原文时用这个版本。
    """

    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d = p.grad.add(p, alpha=group['weight_decay']) if group['weight_decay'] else p.grad
                state = self.state[p]
                if 'history' not in state:
                    state['history'] = torch.zeros_like(p)
                history = state['history']
                history.mul_(group['momentum']).add_(d, alpha=group['lr'])
                p.sub_(history)
