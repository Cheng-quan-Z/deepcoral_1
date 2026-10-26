"""检查转换后的 CaffeNet 权重与 Caffe 预处理是否正确：用 ImageNet 的 1000 类分类头直接预测
Office-31 中有对应 ImageNet 类别的图片，统计 top-5 命中率。

python tools/check_caffenet.py --data_dir data_cache

权重损坏或输入尺度、均值用错时命中率会接近 0；正确时平均约 0.8。
通道顺序（BGR）对这些白底商品图影响很小，由 tests/smoke_test.py 的逐像素核对来保证。
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coral import data  # noqa: E402
from coral.models import CaffeNet, extract_features  # noqa: E402

# Office-31 类名 → 对应的 ImageNet 类别下标
IMAGENET_MATCH = {
    'back_pack': {414},             # backpack
    'bike': {444, 671},             # bicycle-built-for-two, mountain bike
    'bike_helmet': {518},           # crash helmet
    'desk_lamp': {846},             # table lamp
    'keyboard': {508},              # computer keyboard
    'laptop_computer': {620, 681},  # laptop, notebook
    'mobile_phone': {487},          # cellular telephone
    'mouse': {673},                 # computer mouse
    'mug': {504},                   # coffee mug
    'printer': {742},               # printer
    'projector': {745},             # projector
}


def hit_rates(model, x, y, classes, device):
    bank = data.ImageBank(x, device, crop=227, preprocess='caffe')
    logits = extract_features(model, bank, torch.arange(len(y)), ['fc8'])['fc8']
    top5 = logits.topk(5, 1).indices.cpu().numpy()
    rates = {}
    for c, targets in IMAGENET_MATCH.items():
        if c in classes:
            rows = np.flatnonzero(y == classes.index(c))
            rates[c] = np.mean([bool(set(top5[r]) & targets) for r in rows])
    return rates


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data_dir', default='data_cache')
    p.add_argument('--weights', default='weights/caffenet_imagenet.pth')
    p.add_argument('--domain', default='amazon')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.weights, map_location='cpu')
    model = CaffeNet(num_classes=1000)
    model.load_state_dict({**ckpt['state_dict'], 'fc8.weight': ckpt['imagenet_fc8']['weight'],
                           'fc8.bias': ckpt['imagenet_fc8']['bias']})
    model.to(device)
    classes = data.load_classes(args.data_dir)
    x, y = data.load_domain(args.data_dir, args.domain)
    keep = np.isin(y, [classes.index(c) for c in IMAGENET_MATCH if c in classes])
    x, y = x[keep], y[keep]

    rates = hit_rates(model, x, y, classes, device)
    print(f'{"类别":16s} top-5 命中率')
    for c, r in rates.items():
        print(f'{c:16s} {r:.2f}')
    mean = np.mean(list(rates.values()))
    print(f'{"平均":16s} {mean:.2f}')
    print('检查通过' if mean >= 0.6 else '[警告] 命中率偏低（正常约 0.8），权重或预处理可能有问题')


if __name__ == '__main__':
    main()
