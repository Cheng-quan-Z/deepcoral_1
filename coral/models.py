"""骨干网络：torchvision AlexNet 与 BVLC Reference CaffeNet，都可从 conv5 / fc6 / fc7 / fc8 任意层取特征。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

LAYERS = ('conv5', 'fc6', 'fc7', 'fc8')
BACKBONES = ('caffenet', 'torchvision')


class _AlexNetFamily(nn.Module):
    """共用的分类头，按 CaffeNet 的顺序排列：pool5 → fc6 → ReLU → Dropout → fc7 → ReLU → Dropout → fc8。

    子类只需提供以 pool5 结尾的 self.features，以及预训练的 self.fc6、self.fc7。
    fc8 换成 num_classes 维，权重 N(0, 0.005)，偏置 0。
    """

    def _init_head(self, num_classes, dropout):
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


class AlexNet(_AlexNetFamily):
    """torchvision AlexNet（ImageNet 预训练）。torchvision 原版把 Dropout 放在 fc6、fc7 之前且 fc8 前没有，
    这里改回 Caffe 的位置。输入为 RGB、ImageNet 均值方差归一化。
    """

    def __init__(self, num_classes=31, pretrained=True, dropout=0.5):
        super().__init__()
        weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.alexnet(weights=weights)
        self.features = net.features  # 最后一层是 pool5
        self.fc6 = net.classifier[1]
        self.fc7 = net.classifier[4]
        self._init_head(num_classes, dropout)


class CaffeNet(_AlexNetFamily):
    """BVLC Reference CaffeNet，原文所用网络。与 torchvision AlexNet 的区别：conv1 为 96 通道且无 padding，
    先池化再做 LRN，conv2 / conv4 / conv5 分两组，池化按 Caffe 的向上取整。
    输入为 227×227、BGR、0–255 减均值（见 coral.data 的 caffe 预处理）。
    预训练权重由 tools/convert_caffenet.py 从官方 caffemodel 转换得到。
    """

    def __init__(self, num_classes=31, weights=None, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, ceil_mode=True), nn.LocalResponseNorm(5, 1e-4, 0.75, 1.0),
            nn.Conv2d(96, 256, 5, padding=2, groups=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, ceil_mode=True), nn.LocalResponseNorm(5, 1e-4, 0.75, 1.0),
            nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1, groups=2), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1, groups=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, ceil_mode=True),
        )
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self._init_head(num_classes, dropout)
        if weights:
            state = torch.load(weights, map_location='cpu')['state_dict']
            missing, unexpected = self.load_state_dict(state, strict=False)
            if unexpected or set(missing) != {'fc8.weight', 'fc8.bias'}:
                raise RuntimeError(f'CaffeNet 权重不匹配：缺少 {missing}，多余 {unexpected}')


# CaffeNet 各层在 self.features 中的下标，供权重转换使用
CAFFENET_CONV_INDEX = {'conv1': 0, 'conv2': 4, 'conv3': 8, 'conv4': 10, 'conv5': 12}


def build_model(backbone, num_classes, pretrained=True, caffenet_weights=None):
    if backbone == 'torchvision':
        return AlexNet(num_classes, pretrained=pretrained)
    if backbone == 'caffenet':
        return CaffeNet(num_classes, weights=caffenet_weights if pretrained else None)
    raise ValueError(f'未知骨干 {backbone!r}')


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
def extract_features(model, bank, idx, layers, batch_size=500, mirror_gen=None):
    """评估模式下（关闭 Dropout、中心裁剪）提取 idx 对应图像的指定层特征。
    给出 mirror_gen 时每张图随机水平翻转，对应 Caffe 测试阶段 mirror: true 的行为。
    """
    was_training = model.training
    model.eval()
    out = {l: [] for l in layers}
    for s in range(0, len(idx), batch_size):
        feats = model(bank.center(idx[s:s + batch_size], mirror_gen))
        for l in layers:
            out[l].append(feats[l].float())
    model.train(was_training)
    return {l: torch.cat(v) for l, v in out.items()}
