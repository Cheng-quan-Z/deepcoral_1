"""把官方 bvlc_reference_caffenet.caffemodel 转成 PyTorch 权重，不需要安装 Caffe。

python tools/convert_caffenet.py --caffemodel weights/bvlc_reference_caffenet.caffemodel --out weights/caffenet_imagenet.pth

caffemodel 是 protobuf 序列化的 NetParameter，这里直接按 protobuf 线格式解析，
兼容新版（layer，字段 100）和旧版（layers，字段 2，V1LayerParameter）两种写法。
输出文件含 state_dict（conv1–5、fc6、fc7，键名与 coral.models.CaffeNet 一致）和
ImageNet 的 fc8（只给 tools/check_caffenet.py 做正确性检查用）。
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coral.models import CAFFENET_CONV_INDEX, CaffeNet  # noqa: E402

OFFICIAL_SHA1 = '4c8d77deb20ea792f84eb5e6d0a11ca0a8660a46'


def read_varint(buf, pos):
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if b < 0x80:
            return result, pos
        shift += 7


def iter_fields(buf):
    """逐个产出 (字段号, 线类型, 值)。长度分隔字段返回 memoryview 切片，不复制数据。"""
    pos, end = 0, len(buf)
    while pos < end:
        key, pos = read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            val, pos = read_varint(buf, pos)
        elif wire == 1:
            val, pos = buf[pos:pos + 8], pos + 8
        elif wire == 2:
            n, pos = read_varint(buf, pos)
            val, pos = buf[pos:pos + n], pos + n
        elif wire == 5:
            val, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f'不支持的 protobuf 线类型 {wire}')
        yield field, wire, val


def parse_blob(buf):
    """BlobProto：data 为字段 5（float，通常是 packed），double_data 为字段 8。"""
    chunks = []
    for field, wire, val in iter_fields(buf):
        if field == 5:
            chunks.append(np.frombuffer(val, '<f4') if wire == 2 else np.frombuffer(val, '<f4', 1))
        elif field == 8:
            chunks.append(np.frombuffer(val, '<f8').astype(np.float32) if wire == 2
                          else np.frombuffer(val, '<f8', 1).astype(np.float32))
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def parse_caffemodel(path):
    """返回 {层名: [blob 数组, ...]}，只保留带参数的层。"""
    with open(path, 'rb') as f:
        buf = memoryview(f.read())
    layers = {}
    for field, wire, val in iter_fields(buf):
        if wire != 2 or field not in (2, 100):
            continue
        name_field, blob_field = (1, 7) if field == 100 else (4, 6)  # 新版 LayerParameter / 旧版 V1LayerParameter
        name, blobs = None, []
        for f2, w2, v2 in iter_fields(val):
            if f2 == name_field and w2 == 2:
                name = bytes(v2).decode('utf-8')
            elif f2 == blob_field and w2 == 2:
                blobs.append(parse_blob(v2))
        if name and blobs:
            layers[name] = blobs
    return layers


def sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def convert(layers):
    """按 CaffeNet 的参数形状重排 blob，返回 (state_dict, imagenet_fc8)。"""
    model = CaffeNet(num_classes=1000)
    targets = {f'features.{i}': name for name, i in CAFFENET_CONV_INDEX.items()}
    targets.update(fc6='fc6', fc7='fc7', fc8='fc8')
    ref = model.state_dict()
    state = {}
    for prefix, name in targets.items():
        if name not in layers or len(layers[name]) != 2:
            raise SystemExit(f'caffemodel 中缺少层 {name} 的权重和偏置，现有层：{sorted(layers)}')
        for blob, suffix in zip(layers[name], ('weight', 'bias')):
            key = f'{prefix}.{suffix}'
            if blob.size != ref[key].numel():
                raise SystemExit(f'{name}.{suffix} 元素数 {blob.size}，应为 {ref[key].numel()}')
            state[key] = torch.from_numpy(blob.copy()).view_as(ref[key])
    fc8 = {'weight': state.pop('fc8.weight'), 'bias': state.pop('fc8.bias')}
    return state, fc8


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--caffemodel', default='weights/bvlc_reference_caffenet.caffemodel')
    p.add_argument('--out', default='weights/caffenet_imagenet.pth')
    args = p.parse_args()

    digest = sha1(args.caffemodel)
    print(f'SHA1 {digest}' + ('（与官方一致）' if digest == OFFICIAL_SHA1 else f'  [警告] 官方为 {OFFICIAL_SHA1}'))
    try:
        layers = parse_caffemodel(args.caffemodel)
    except (IndexError, ValueError) as e:
        raise SystemExit(f'解析失败（{e}），文件很可能没下完整：大小应为 243,862,418 字节，请续传后重试')
    state, fc8 = convert(layers)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({'state_dict': state, 'imagenet_fc8': fc8, 'source_sha1': digest}, args.out)
    for k, v in state.items():
        print(f'  {k:20s} {tuple(v.shape)}')
    print(f'已写入 {args.out}')


if __name__ == '__main__':
    main()
