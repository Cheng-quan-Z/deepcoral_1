"""把 Office-31 三个域的图像统一缩放到 256x256，存成 uint8 数组缓存。只需运行一次。

python prepare_data.py --root /path/to/office31 --out data_cache

--root 下应有 amazon / dslr / webcam 三个目录，每个目录下是 images/<类名>/*.jpg 或直接 <类名>/*.jpg。
三个域按类名排序统一类别编号。
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from coral.data import EXPECTED_SIZES, RESIZE

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def domain_dir(root, domain):
    d = os.path.join(root, domain)
    return os.path.join(d, 'images') if os.path.isdir(os.path.join(d, 'images')) else d


def load_image(path):
    with Image.open(path) as img:
        return np.asarray(img.convert('RGB').resize((RESIZE, RESIZE), Image.BILINEAR), dtype=np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--root', required=True, help='Office-31 根目录')
    p.add_argument('--out', default='data_cache', help='缓存输出目录')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    class_sets = {}
    for domain in EXPECTED_SIZES:
        d = domain_dir(args.root, domain)
        class_sets[domain] = sorted(c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c)))
    classes = class_sets['amazon']
    for domain, cs in class_sets.items():
        if cs != classes:
            raise SystemExit(f'{domain} 的类别与 amazon 不一致：{sorted(set(cs) ^ set(classes))}')
    if len(classes) != 31:
        print(f'[警告] 类别数为 {len(classes)}，Office-31 应为 31')

    os.makedirs(args.out, exist_ok=True)
    meta = {}
    for domain, expected in EXPECTED_SIZES.items():
        d = domain_dir(args.root, domain)
        paths, labels = [], []
        for y, c in enumerate(classes):
            files = sorted(f for f in os.listdir(os.path.join(d, c)) if f.lower().endswith(IMG_EXT))
            paths += [os.path.join(d, c, f) for f in files]
            labels += [y] * len(files)
        with ProcessPoolExecutor(args.workers) as ex:
            x = np.stack(list(ex.map(load_image, paths, chunksize=32)))
        np.save(os.path.join(args.out, f'{domain}_x.npy'), x)
        np.save(os.path.join(args.out, f'{domain}_y.npy'), np.asarray(labels, dtype=np.int16))
        meta[domain] = {'n': len(paths), 'files': [os.path.relpath(q, d).replace(os.sep, '/') for q in paths]}
        flag = '' if len(paths) == expected else f'  [警告] 应为 {expected} 张'
        print(f'{domain}: {len(paths)} 张，{x.nbytes / 2 ** 20:.0f} MB{flag}')

    with open(os.path.join(args.out, 'classes.json'), 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f'缓存已写入 {args.out}')


if __name__ == '__main__':
    main()
