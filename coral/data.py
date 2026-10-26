"""Office-31 数据：缓存读取、源域验证集切分、GPU 端数据增强与双流采样。

prepare_data.py 先把每张图缩放到 256x256 存成 uint8 数组。训练时整个域常驻显存，
随机裁剪、水平翻转和归一化都在 GPU 上完成，效果等价于 torchvision 的
Resize((256, 256)) → RandomCrop(crop) → RandomHorizontalFlip → Normalize，
测试时等价于 Resize((256, 256)) → CenterCrop(crop) → Normalize。
"""
import json
import os

import numpy as np
import torch

DOMAINS = {'A': 'amazon', 'D': 'dslr', 'W': 'webcam'}
EXPECTED_SIZES = {'amazon': 2817, 'dslr': 498, 'webcam': 795}
RESIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_task(task):
    """'AW' -> ('amazon', 'webcam')。"""
    if len(task) != 2 or task[0] not in DOMAINS or task[1] not in DOMAINS or task[0] == task[1]:
        raise ValueError(f'非法任务名 {task!r}，应为 AW、DA 这样的两个不同字母')
    return DOMAINS[task[0]], DOMAINS[task[1]]


def load_classes(data_dir):
    with open(os.path.join(data_dir, 'classes.json'), encoding='utf-8') as f:
        return json.load(f)


def load_domain(data_dir, domain):
    """返回 (图像 N×256×256×3 uint8, 标签 int64)。"""
    x = np.load(os.path.join(data_dir, f'{domain}_x.npy'))
    y = np.load(os.path.join(data_dir, f'{domain}_y.npy')).astype(np.int64)
    return x, y


def stratified_split(labels, val_frac, seed=0):
    """按类别分层切出验证集，返回 (train_idx, val_idx)。val_frac = 0 时验证集为空。"""
    rng = np.random.RandomState(seed)
    val = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        val.extend(idx[:int(round(len(idx) * val_frac))])
    val = np.sort(np.asarray(val, dtype=np.int64))
    train = np.setdiff1d(np.arange(len(labels)), val)
    return train, val


class ImageBank:
    """一个域的全部图像，常驻 device。"""

    def __init__(self, images, device, crop=224):
        assert images.shape[1:] == (RESIZE, RESIZE, 3), images.shape
        self.x = torch.from_numpy(images).to(device)
        self.device = device
        self.crop = crop
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1) * 255

    def __len__(self):
        return self.x.shape[0]

    def _normalize(self, patch):
        return (patch.permute(0, 3, 1, 2).float() - self.mean) / self.std

    def augment(self, idx, gen):
        """训练增强：每张图独立的随机裁剪位置和随机水平翻转。gen 为 CPU 上的 torch.Generator。"""
        n, c = idx.numel(), self.crop
        oy = torch.randint(0, RESIZE - c + 1, (n, 1), generator=gen)
        ox = torch.randint(0, RESIZE - c + 1, (n, 1), generator=gen)
        flip = torch.rand(n, 1, generator=gen) < 0.5
        ar = torch.arange(c)
        rows = oy + ar
        cols = torch.where(flip, ox + c - 1 - ar, ox + ar)
        idx, rows, cols = (t.to(self.device) for t in (idx, rows, cols))
        return self._normalize(self.x[idx[:, None, None], rows[:, :, None], cols[:, None, :]])

    def center(self, idx):
        """测试预处理：中心裁剪。"""
        o, c = (RESIZE - self.crop) // 2, self.crop
        return self._normalize(self.x[idx.to(self.device), o:o + c, o:o + c])


class _Stream:
    """无限循环的随机索引流：每个 epoch 重新打乱，丢弃不满的 batch，保证每步样本数恒定。"""

    def __init__(self, indices, batch_size, gen):
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        if len(self.indices) < batch_size:
            raise ValueError(f'样本数 {len(self.indices)} 小于 batch 大小 {batch_size}')
        self.batch_size = batch_size
        self.gen = gen
        self.order, self.pos = None, 0

    def _next_idx(self):
        if self.order is None or self.pos + self.batch_size > len(self.indices):
            self.order = self.indices[torch.randperm(len(self.indices), generator=self.gen)]
            self.pos = 0
        batch = self.order[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        return batch


class LabeledStream(_Stream):
    """源域训练流：返回 (图像, 标签)。"""

    def __init__(self, bank, labels, indices, batch_size, gen):
        super().__init__(indices, batch_size, gen)
        self.bank = bank
        self.labels = torch.as_tensor(labels, dtype=torch.long).to(bank.device)

    def next(self):
        idx = self._next_idx()
        return self.bank.augment(idx, self.gen), self.labels[idx.to(self.bank.device)]


class UnlabeledStream(_Stream):
    """目标域训练流：构造时不接收标签，只返回图像，从结构上杜绝训练读到目标标签。"""

    def __init__(self, bank, indices, batch_size, gen):
        super().__init__(indices, batch_size, gen)
        self.bank = bank

    def next(self):
        return self.bank.augment(self._next_idx(), self.gen)
