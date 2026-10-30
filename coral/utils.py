"""随机种子、代码版本、硬件信息与 JSON 落盘。"""
import glob
import hashlib
import json
import os
import platform
import random

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def code_version():
    """coral/*.py 与 train.py 内容的 md5 前 12 位，用来标识产生某条日志的代码版本。
    换行符统一成 LF 再计算，Windows 上 Git 检出为 CRLF 时版本号不变。
    """
    h = hashlib.md5()
    for path in sorted(glob.glob(os.path.join(ROOT, 'coral', '*.py'))) + [os.path.join(ROOT, 'train.py')]:
        with open(path, 'rb') as f:
            h.update(f.read().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


def hardware_info(device):
    info = {'python': platform.python_version(), 'torch': torch.__version__, 'platform': platform.platform()}
    if device.type == 'cuda':
        info.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version())
    return info


def save_json(obj, path):
    """先写临时文件再改名，中途中断不会留下半个 JSON（续跑时靠 JSON 是否存在判断运行是否完成）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)
