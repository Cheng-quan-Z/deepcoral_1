"""冒烟测试：数值核对 + 用假数据从数据缓存到阶段一汇总完整跑一遍。CPU 上约 3–5 分钟，不需要下载任何文件。

python tests/smoke_test.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import convert_caffenet  # noqa: E402
from coral import data, metrics  # noqa: E402
from coral.losses import coral_loss, covariance  # noqa: E402
from coral.models import CAFFENET_CONV_INDEX, AlexNet, CaffeNet, param_groups  # noqa: E402
from train import lr_at, parse_args  # noqa: E402


def check_covariance():
    x = np.random.randn(50, 7)
    assert np.allclose(covariance(torch.tensor(x)).numpy(), np.cov(x, rowvar=False))
    # 论文式 (2) 的展开写法
    n = len(x)
    one = np.ones((1, n))
    c = (x.T @ x - (one @ x).T @ (one @ x) / n) / (n - 1)
    assert np.allclose(covariance(torch.tensor(x)).numpy(), c)


def check_gradient():
    """自动求导结果与论文式 (4)(5) 一致。"""
    ns, nt, d = 40, 30, 6
    ds = torch.randn(ns, d, dtype=torch.float64, requires_grad=True)
    dt = torch.randn(nt, d, dtype=torch.float64, requires_grad=True)
    gs, gt = torch.autograd.grad(coral_loss(ds, dt), (ds, dt))
    diff = (covariance(ds) - covariance(dt)).detach()
    for x, g, n, sign in ((ds.detach(), gs, ns, 1), (dt.detach(), gt, nt, -1)):
        one = torch.ones(1, n, dtype=torch.float64)
        formula = sign / (d * d * (n - 1)) * (x.t() - (one @ x).t() @ one / n).t() @ diff
        assert torch.allclose(g, formula), (g - formula).abs().max()


def check_augment():
    images = np.random.randint(0, 256, (5, 256, 256, 3), dtype=np.uint8)
    bank = data.ImageBank(images, torch.device('cpu'), crop=224)
    idx = torch.tensor([3, 0, 3])
    out = bank.augment(idx, torch.Generator().manual_seed(7))
    gen = torch.Generator().manual_seed(7)
    oy = torch.randint(0, 33, (3, 1), generator=gen)[:, 0]
    ox = torch.randint(0, 33, (3, 1), generator=gen)[:, 0]
    flip = (torch.rand(3, 1, generator=gen) < 0.5)[:, 0]
    mean, std = np.array(data.IMAGENET_MEAN) * 255, np.array(data.IMAGENET_STD) * 255
    for k in range(3):
        patch = images[idx[k], oy[k]:oy[k] + 224, ox[k]:ox[k] + 224].astype(np.float64)
        if flip[k]:
            patch = patch[:, ::-1]
        ref = ((patch - mean) / std).transpose(2, 0, 1)
        assert np.allclose(out[k].numpy(), ref, atol=1e-4)
    center = bank.center(torch.tensor([1]))[0].numpy()
    ref = ((images[1, 16:240, 16:240] - mean) / std).transpose(2, 0, 1)
    assert np.allclose(center, ref, atol=1e-4)

    # Caffe 预处理：227 中心裁剪（偏移 14），BGR，0–255 减均值
    caffe = data.ImageBank(images, torch.device('cpu'), crop=227, preprocess='caffe')
    ref = (images[2, 14:241, 14:241, ::-1] - np.array(data.CAFFE_MEAN_BGR)).transpose(2, 0, 1)
    assert np.allclose(caffe.center(torch.tensor([2]))[0].numpy(), ref, atol=1e-4)
    # 测试时随机翻转：翻转的那些恰好是中心裁剪的左右镜像
    plain = caffe.center(torch.arange(5))
    mirrored = caffe.center(torch.arange(5), torch.Generator().manual_seed(3))
    flip = torch.rand(5, generator=torch.Generator().manual_seed(3)) < 0.5
    for k in range(5):
        assert torch.allclose(mirrored[k], plain[k].flip(2) if flip[k] else plain[k])


def check_streams():
    bank = data.ImageBank(np.zeros((10, 256, 256, 3), np.uint8), torch.device('cpu'))
    tgt = data.UnlabeledStream(bank, np.arange(10), 4, torch.Generator().manual_seed(0))
    assert not hasattr(tgt, 'labels') and torch.is_tensor(tgt.next())
    src = data.LabeledStream(bank, np.arange(10), np.arange(10), 4, torch.Generator().manual_seed(0))
    seen = torch.cat([src.next()[1] for _ in range(2)])
    assert len(set(seen.tolist())) == 8  # 一个 epoch 内不重复，10 张只用 8 张（丢弃不满的 batch）
    tr, va = data.stratified_split(np.repeat(np.arange(3), 10), 0.1)
    assert len(va) == 3 and len(np.intersect1d(tr, va)) == 0


def check_model():
    for model in (AlexNet(31, pretrained=False), CaffeNet(31)):
        for crop in (224, 227):
            out = model(torch.randn(2, 3, crop, crop))
            assert {k: v.shape[1] for k, v in out.items()} == {'conv5': 256, 'fc6': 4096, 'fc7': 4096, 'fc8': 31}
        mults = sorted((g['lr_mult'], g['weight_decay']) for g in param_groups(model, 1e-3, 5e-4))
        assert mults == [(1.0, 5e-4), (2.0, 0.0), (10.0, 5e-4), (20.0, 0.0)]
    # CaffeNet 各层参数形状与 Caffe 的 bvlc_reference_caffenet 一致（分组卷积的输入通道减半）
    shapes = {k: tuple(v.shape) for k, v in CaffeNet(31).state_dict().items() if k.endswith('weight')}
    assert shapes['features.0.weight'] == (96, 3, 11, 11) and shapes['features.4.weight'] == (256, 48, 5, 5)
    assert shapes['features.10.weight'] == (384, 192, 3, 3) and shapes['fc6.weight'] == (4096, 9216)


def check_lr_and_preset():
    args = parse_args(['--task', 'AW', '--name', 'x', '--out_dir', 'x', '--preset', 'caffe', '--no_pretrained'])
    assert (args.backbone, args.crop, args.lr_schedule, args.test_mirror, args.preprocess) == \
        ('caffenet', 227, 'step', True, 'caffe')
    assert [lr_at(args, it) for it in (0, 499, 500, 1999)] == [1e-3, 1e-3, 1e-3 * 0.1, 1e-3 * 0.1 ** 3]
    args = parse_args(['--task', 'AW', '--name', 'x', '--out_dir', 'x', '--preset', 'caffe', '--no_pretrained',
                       '--lr_schedule', 'const', '--no-test_mirror'])  # 单独给出的参数覆盖预设
    assert args.lr_schedule == 'const' and not args.test_mirror


def _varint(n):
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(num, wire, payload):
    key = _varint(num << 3 | wire)
    return key + (_varint(len(payload)) + payload if wire == 2 else payload)


def _caffemodel(params, v1):
    """按 caffe.proto 编码一个 NetParameter。v1=True 为旧版 layers / V1LayerParameter 写法。"""
    net = _field(1, 2, b'CaffeNet')
    for name, blobs in params.items():
        enc = b''.join(_field(6 if v1 else 7, 2, _field(5, 2, b.astype('<f4').tobytes())) for b in blobs)
        if v1:
            layer = _field(4, 2, name.encode()) + _field(5, 0, _varint(4)) + enc
            net += _field(2, 2, layer)
        else:
            layer = _field(1, 2, name.encode()) + _field(2, 2, b'Convolution') + enc
            net += _field(100, 2, layer)
    net += _field(2 if v1 else 100, 2, _field(4 if v1 else 1, 2, b'relu1'))  # 无参数的层应被忽略
    return net


def check_converter(tmp):
    """用随机 CaffeNet 参数编码出假的 caffemodel，转换后应逐元素还原。"""
    ref = CaffeNet(num_classes=1000).state_dict()
    names = {**{n: f'features.{i}' for n, i in CAFFENET_CONV_INDEX.items()}, 'fc6': 'fc6', 'fc7': 'fc7', 'fc8': 'fc8'}
    params = {n: [ref[f'{k}.weight'].numpy(), ref[f'{k}.bias'].numpy()] for n, k in names.items()}
    path = os.path.join(tmp, 'fake.caffemodel')
    with open(path, 'wb') as f:
        f.write(_caffemodel(params, v1=True))
    state, fc8 = convert_caffenet.convert(convert_caffenet.parse_caffemodel(path))
    assert all(torch.equal(state[k], ref[k]) for k in state) and len(state) == 14  # conv1–5、fc6、fc7 的权重和偏置
    assert torch.equal(fc8['weight'], ref['fc8.weight'])
    small = {'conv1': [np.arange(6, dtype=np.float32), np.ones(2, np.float32)]}
    with open(path, 'wb') as f:
        f.write(_caffemodel(small, v1=False))
    layers = convert_caffenet.parse_caffemodel(path)
    assert list(layers) == ['conv1'] and np.array_equal(layers['conv1'][0], small['conv1'][0])
    unpacked = b''.join(_field(5, 5, np.float32(v).tobytes()) for v in (1.5, -2.0))  # 非 packed 写法
    assert np.array_equal(convert_caffenet.parse_blob(unpacked), [1.5, -2.0])
    # 转换结果能被 CaffeNet 直接加载
    weights = os.path.join(tmp, 'caffenet.pth')
    torch.save({'state_dict': state, 'imagenet_fc8': fc8}, weights)
    assert torch.equal(CaffeNet(31, weights=weights).fc6.weight, ref['fc6.weight'])


def check_metrics():
    logits = torch.randn(200, 31)
    labels = torch.randint(0, 31, (200,))
    cm = metrics.confusion_matrix(logits.argmax(1), labels, 31)
    assert cm.sum() == 200 and abs(cm.diag().sum() / 200 - metrics.accuracy(logits, labels)) < 1e-6
    assert 0 <= metrics.ece(logits.softmax(1), labels) <= 1
    x = torch.randn(100, 3) @ torch.randn(3, 50)  # 秩 3
    assert metrics.effective_rank(x) <= 3 + 1e-6


def check_calibration_rounds():
    """结束时损失比超出 2 倍时，用新比值再训练恰好一次。用假的 train 代替真实训练。"""
    sys.path.insert(0, os.path.join(ROOT, 'stage1_reproduction'))
    import calibrate_lambda as cl

    calls = []

    def fake_train(task, name, lam, args, extra):
        calls.append(lam)
        coral = 0.1 if lam == 1.0 else 0.01  # λ=1 → λ*=10 → 比值 10 超限 → λ**=100 → 比值 1
        return {'name': name, 'final': {}, 'trace': {'cls_loss': [1.0] * 5, 'coral_loss': [coral] * 5}}

    cl.train = fake_train
    args = type('A', (), {'prefix': 's1', 'calib_exp': 'calib', 'seed': 1})()
    entry = cl.calibrate('AW', args, [])
    assert np.allclose(calls, [1, 10, 100]) and np.isclose(entry['lam_star'], 100) and entry['accepted']
    assert entry['final_run'] == 's1-calib-AW-lamstar2-seed1' and len(entry['rounds']) == 2


def run(*cmd):
    print('>>', ' '.join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], check=True, cwd=ROOT)


def load_log(out, name):
    with open(os.path.join(out, 'logs', f'{name}.json'), encoding='utf-8') as f:
        return json.load(f)


def check_pipeline(tmp):
    raw, cache, cache_pil, out = (os.path.join(tmp, d) for d in ('office31', 'cache', 'cache_pil', 'results'))
    rng = np.random.RandomState(0)
    for domain in data.EXPECTED_SIZES:
        for c in range(31):
            d = os.path.join(raw, domain, 'images', f'class{c:02d}')
            os.makedirs(d)
            for k in range(4):
                Image.fromarray(rng.randint(0, 256, (60, 80, 3), dtype=np.uint8)).save(os.path.join(d, f'{k}.jpg'))
    run('prepare_data.py', '--root', raw, '--out', cache, '--interp', 'caffe', '--workers', '2')
    run('prepare_data.py', '--root', raw, '--out', cache_pil, '--interp', 'pil', '--workers', '2')
    a, b = np.load(os.path.join(cache, 'amazon_x.npy')), np.load(os.path.join(cache_pil, 'amazon_x.npy'))
    assert a.shape == b.shape and not np.array_equal(a, b)  # 两种缩放方式确实不同

    fast = ['--iters', '20', '--eval_every', '10', '--batch_size', '16', '--eval_batches', '3',
            '--no_pretrained', '--device', 'cpu']
    runner = ['stage1_reproduction/run_stage1.py']
    common = ['--data_dir', cache, '--data_dir_pil', cache_pil, '--out_dir', out,
              '--tasks', 'AW', '--seeds', '1', '--attr_seeds', '1', *fast]
    run(*runner, 'baseline', *common)
    run(*runner, 'coral', *common)
    # 随机初始化的网络在噪声图上 CORAL 损失约 1e-12，标定出的 λ 极大会发散；lr = 0 让两次运行损失相同，只测流程
    run(*runner, 'rule', *common, '--lr', '0')
    run(*runner, 'attribution', *common)
    run('train.py', '--task', 'DA', '--seed', '2', '--lam', '1', '--name', 's9-misc-DA-all-seed2', '--out_dir', out,
        '--align_layers', 'conv5,fc6,fc7,fc8', '--lr_schedule', 'inv', '--src_val_frac', '0.25', '--crop', '227',
        '--data_dir', cache, *fast)
    run('train.py', '--task', 'WD', '--lam', '1e20', '--lr', '10', '--name', 's9-diverge-WD-big-seed1', '--out_dir', out,
        '--data_dir', cache, *fast)
    run('stage1_reproduction/analyze_stage1.py', '--out_dir', out)

    log = load_log(out, 's1-coral-AW-lamoff-seed1')
    cfg = log['config']
    assert (cfg['backbone'], cfg['crop'], cfg['lr_schedule'], cfg['test_mirror'], cfg['lam'], cfg['cache_interp']) == \
        ('caffenet', 227, 'step', True, 0.8, 'caffe')
    for key in ('tgt_acc', 'tgt_bal_acc', 'coral_full_fc8', 'coral_evalbatch_fc8', 'erank_fc7_tgt', 'tgt_snd', 'tgt_ece'):
        assert key in log['final'], key
    log = load_log(out, 's1-coralrule-AW-lamstar-seed1')
    assert log['promoted_from'] and len(log['trace']['cls_loss']) == 20 and len(log['history']) == 3
    feats = np.load(os.path.join(out, 'features', 's1-coralrule-AW-lamstar-seed1.npz'))
    assert feats['tgt_fc7'].shape == (124, 4096) and feats['src_fc8'].shape == (124, 31)
    tvall = load_log(out, 's1-attr-AW-tvall-seed1')['config']
    assert (tvall['backbone'], tvall['crop'], tvall['lr_schedule'], tvall['test_mirror'], tvall['cache_interp']) == \
        ('torchvision', 224, 'const', False, 'pil')
    log = load_log(out, 's9-misc-DA-all-seed2')
    assert log['config']['n_src_val'] == 31 and 'src_val_acc' in log['final'] and 'coral_full_conv5' in log['final']
    with open(os.path.join(out, 'stage1_summary.md'), encoding='utf-8') as f:
        summary = f.read()
    for section in ('主结果', '与原作者训练日志对照', '规则标定', '差距归因'):
        assert section in summary, section
    for name in ('stage1_runs.csv', 'figures/fig2_AW.png'):
        assert os.path.exists(os.path.join(out, name)), name


def main():
    for check in (check_covariance, check_gradient, check_augment, check_streams, check_model, check_lr_and_preset,
                  check_metrics, check_calibration_rounds):
        check()
        print(f'[ok] {check.__name__}', flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        check_converter(tmp)
        print('[ok] check_converter', flush=True)
        check_pipeline(tmp)
    print('[ok] 全部通过')


if __name__ == '__main__':
    main()
