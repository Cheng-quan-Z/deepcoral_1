"""冒烟测试：数值核对 + 用假数据从数据缓存到阶段一汇总完整跑一遍。CPU 上约 1–2 分钟。

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

from coral import data, metrics  # noqa: E402
from coral.losses import coral_loss, covariance  # noqa: E402
from coral.models import AlexNet, param_groups  # noqa: E402


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
    model = AlexNet(31, pretrained=False)
    for crop in (224, 227):
        out = model(torch.randn(2, 3, crop, crop))
        assert {k: v.shape[1] for k, v in out.items()} == {'conv5': 256, 'fc6': 4096, 'fc7': 4096, 'fc8': 31}
    mults = sorted((g['lr_mult'], g['weight_decay']) for g in param_groups(model, 1e-3, 5e-4))
    assert mults == [(1.0, 5e-4), (2.0, 0.0), (10.0, 5e-4), (20.0, 0.0)]


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


def check_pipeline(tmp):
    raw, cache, out = (os.path.join(tmp, d) for d in ('office31', 'cache', 'results'))
    rng = np.random.RandomState(0)
    for domain in data.EXPECTED_SIZES:
        for c in range(31):
            d = os.path.join(raw, domain, 'images', f'class{c:02d}')
            os.makedirs(d)
            for k in range(4):
                Image.fromarray(rng.randint(0, 256, (60, 80, 3), dtype=np.uint8)).save(os.path.join(d, f'{k}.jpg'))
    run('prepare_data.py', '--root', raw, '--out', cache, '--workers', '2')

    fast = ['--data_dir', cache, '--iters', '20', '--eval_every', '10', '--batch_size', '16',
            '--eval_batches', '3', '--no_pretrained', '--device', 'cpu']
    run('train.py', '--task', 'AW', '--seed', '1', '--lam', '0', '--name', 's1-base-AW-lam0-seed1',
        '--out_dir', out, '--save_features', '--save_ckpt', *fast)
    # 随机初始化的网络在噪声图上 CORAL 损失约 1e-12，标定出的 λ 极大会发散；lr = 0 让两次运行损失相同，只测流程
    run('stage1_reproduction/calibrate_lambda.py', '--tasks', 'AW', '--seed', '1', '--out_dir', out,
        '--save_features', *fast, '--lr', '0')
    run('train.py', '--task', 'DA', '--seed', '2', '--lam', '1', '--name', 's9-misc-DA-all-seed2', '--out_dir', out,
        '--align_layers', 'conv5,fc6,fc7,fc8', '--lr_schedule', 'inv', '--src_val_frac', '0.25', '--crop', '227', *fast)
    run('train.py', '--task', 'WD', '--lam', '1e20', '--lr', '10', '--name', 's9-diverge-WD-big-seed1', '--out_dir', out, *fast)
    run('stage1_reproduction/analyze_stage1.py', '--out_dir', out)

    with open(os.path.join(out, 'logs', 's1-coral-AW-lamstar-seed1.json'), encoding='utf-8') as f:
        log = json.load(f)
    assert log['promoted_from'] and len(log['trace']['cls_loss']) == 20 and len(log['history']) == 3
    for key in ('tgt_acc', 'tgt_bal_acc', 'coral_full_fc8', 'coral_evalbatch_fc8', 'erank_fc7_tgt', 'tgt_snd', 'tgt_ece'):
        assert key in log['final'], key
    feats = np.load(os.path.join(out, 'features', 's1-coral-AW-lamstar-seed1.npz'))
    assert feats['tgt_fc7'].shape == (124, 4096) and feats['src_fc8'].shape == (124, 31)
    with open(os.path.join(out, 'logs', 's9-misc-DA-all-seed2.json'), encoding='utf-8') as f:
        log = json.load(f)
    assert log['config']['n_src_val'] == 31 and 'src_val_acc' in log['final'] and 'coral_full_conv5' in log['final']
    for name in ('stage1_summary.md', 'stage1_runs.csv', 'figures/fig2_AW.png'):
        assert os.path.exists(os.path.join(out, name)), name


def main():
    for check in (check_covariance, check_gradient, check_augment, check_streams, check_model, check_metrics,
                  check_calibration_rounds):
        check()
        print(f'[ok] {check.__name__}', flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        check_pipeline(tmp)
    print('[ok] 全部通过')


if __name__ == '__main__':
    main()
