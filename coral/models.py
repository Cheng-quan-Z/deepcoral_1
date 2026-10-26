"""AlexNet 骨干：可从 conv5 / fc6 / fc7 / fc8 任意层取特征。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

LAYERS = ('conv5', 'fc6', 'fc7', 'fc8')


class AlexNet(nn.Module):
    """torchvision AlexNet（ImageNet 预训练），分类头按 CaffeNet 的顺序排列：
    pool5 → fc6 → ReLU → Dropout → fc7 → ReLU → Dropout → fc8。

    torchvision 原版把 Dropout 放在 fc6、fc7 之前且 fc8 前没有 Dropout，这里改回 Caffe 的位置。
    fc8 换成 num_classes 维，权重 N(0, 0.005)，偏置 0。
    """

    def __init__(self, num_classes=31, pretrained=True, dropout=0.5):
        super().__init__()
        weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.alexnet(weights=weights)
        self.features = net.features  # 最后一层是 pool5
        self.fc6 = net.classifier[1]
        self.fc7 = net.classifier[4]
        self.fc8 = nn.Linear(4096, num_classes)
        nn.init.normal_(self.fc8.weight, 0.0, 0.005)
        nn.init.zeros_(self.fc8.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """返回各层特征的字典。conv5 为 ReLU 后全局平均池化的 256 维；
        fc6、fc7 取 ReLU 之后、Dropout 之前；fc8 为 softmax 之前的 logits。
        """
        conv5 = self.features[:-1](x)
        pool5 = torch.flatten(self.features[-1](conv5), 1)
        fc6 = F.relu(self.fc6(pool5))
        fc7 = F.relu(self.fc7(self.dropout(fc6)))
        fc8 = self.fc8(self.dropout(fc7))
        return {'conv5': conv5.mean((2, 3)), 'fc6': fc6, 'fc7': fc7, 'fc8': fc8}


def param_groups(model, lr, weight_decay, new_layers=('fc8',), new_lr_mult=10.0):
    """Caffe 微调惯例：新层学习率 ×10；偏置学习率再 ×2 且不做权重衰减。

    每组记录 lr_mult，学习率调度时按 lr_t × lr_mult 设置。
    """
    groups = {}
    for name, p in model.named_parameters():
        new = name.split('.')[0] in new_layers
        bias = name.endswith('bias')
        if (new, bias) not in groups:
            mult = (new_lr_mult if new else 1.0) * (2.0 if bias else 1.0)
            groups[(new, bias)] = {'params': [], 'lr': lr * mult, 'lr_mult': mult,
                                   'weight_decay': 0.0 if bias else weight_decay}
        groups[(new, bias)]['params'].append(p)
    return list(groups.values())


@torch.no_grad()
def extract_features(model, bank, idx, layers, batch_size=500):
    """评估模式下（关闭 Dropout、中心裁剪）提取 idx 对应图像的指定层特征。"""
    was_training = model.training
    model.eval()
    out = {l: [] for l in layers}
    for s in range(0, len(idx), batch_size):
        feats = model(bank.center(idx[s:s + batch_size]))
        for l in layers:
            out[l].append(feats[l].float())
    model.train(was_training)
    return {l: torch.cat(v) for l, v in out.items()}
