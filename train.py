"""Deep CORAL 统一训练入口：各阶段的实验都通过命令行参数切换配置，每次运行写一个 JSON 日志。

例：python train.py --task AW --lam 0 --seed 1 --name s1-base-AW-lam0-seed1 --out_dir stage1_reproduction/results

输出：
  <out_dir>/logs/<name>.json       配置、每 eval_every 次迭代的指标、最终指标、逐迭代的两项损失
  <out_dir>/features/<name>.npz    --save_features：最后一轮评估模式下的全量 fc7 / fc8 特征与标签
  <out_dir>/ckpt/<name>.pt         --save_ckpt：最后一轮的模型权重
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from coral import data, metrics
from coral.losses import coral_loss
from coral.models import LAYERS, AlexNet, extract_features, param_groups
from coral.utils import code_version, hardware_info, save_json, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--task', required=True, help='迁移任务，如 AW 表示 Amazon→Webcam')
    p.add_argument('--name', required=True, help='运行名，按“阶段-实验-任务-配置-种子”命名')
    p.add_argument('--out_dir', required=True, help='结果目录')
    p.add_argument('--data_dir', default='data_cache', help='prepare_data.py 生成的缓存目录')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--iters', type=int, default=2000)
    p.add_argument('--eval_every', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=128, help='源域、目标域各取这么多')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr_schedule', choices=['const', 'inv'], default='const',
                   help='const：固定学习率；inv：lr0 / (1 + 10p)^0.75，p 为训练进度')
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--lam', type=float, default=0.0, help='CORAL 损失权重，0 即无适应基线')
    p.add_argument('--align_layers', default='fc8', help='施加 CORAL 的层，逗号分隔，可选 ' + ','.join(LAYERS))
    p.add_argument('--crop', type=int, default=224, help='裁剪尺寸，CaffeNet 为 227')
    p.add_argument('--src_val_frac', type=float, default=0.0, help='从源域按类分层留出、不参与训练的验证集比例')
    p.add_argument('--split_seed', type=int, default=0, help='验证集切分的种子，所有运行共用同一切分')
    p.add_argument('--eval_batches', type=int, default=50, help='评估模式下随机抽多少对 batch 算平均 batch CORAL 距离')
    p.add_argument('--save_features', action='store_true')
    p.add_argument('--save_ckpt', action='store_true')
    p.add_argument('--no_pretrained', action='store_true', help='不加载 ImageNet 预训练权重（仅用于冒烟测试）')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--overwrite', action='store_true', help='日志已存在时也重新训练')
    args = p.parse_args(argv)
    args.align_layers = [l for l in args.align_layers.split(',') if l]
    bad = set(args.align_layers) - set(LAYERS)
    if bad or not args.align_layers:
        p.error(f'--align_layers 非法：{args.align_layers}')
    return args


def lr_at(args, it):
    if args.lr_schedule == 'const':
        return args.lr
    return args.lr / (1 + 10 * it / args.iters) ** 0.75


@torch.no_grad()
def evaluate(model, eval_sets, layers, align_layers, batch_size, n_batches, gen):
    """评估模式下（关闭 Dropout、中心裁剪）提取全量特征并计算各项指标。

    eval_sets: {'src': (bank, idx, labels), 'tgt': ..., 可选 'src_val': ...}。目标标签只在这里被读取。
    返回 (指标字典, 各集合的特征, 目标域混淆矩阵)。特征出现 NaN / Inf 时只返回准确率并标记 diverged。
    """
    feats = {k: extract_features(model, bank, idx, layers) for k, (bank, idx, _) in eval_sets.items()}
    rec = {f'{k}_acc': metrics.accuracy(feats[k]['fc8'], y) for k, (_, _, y) in eval_sets.items()}

    fs, ft = feats['src'], feats['tgt']
    yt = eval_sets['tgt'][2]
    logits = ft['fc8']
    cm = metrics.confusion_matrix(logits.argmax(1), yt, logits.shape[1])
    rec['tgt_bal_acc'] = metrics.balanced_accuracy(cm)
    rec['diverged'] = not all(bool(torch.isfinite(v).all()) for f in feats.values() for v in f.values())
    if rec['diverged']:
        return rec, feats, cm
    rec['tgt_ece'] = metrics.ece(logits.softmax(1), yt)
    rec['tgt_nll'] = float(F.cross_entropy(logits, yt))
    rec['tgt_entropy'] = metrics.mean_entropy(logits)
    rec['tgt_snd'] = metrics.snd(logits)
    for l in layers:
        rec[f'coral_full_{l}'] = metrics.coral_distance(fs[l], ft[l])
    for l in dict.fromkeys(['fc8'] + align_layers):
        rec[f'coral_evalbatch_{l}'] = metrics.batch_coral_distance(fs[l], ft[l], batch_size, n_batches, gen)
    for l in ('fc7', 'fc8'):
        rec[f'erank_{l}_src'] = metrics.effective_rank(fs[l])
        rec[f'erank_{l}_tgt'] = metrics.effective_rank(ft[l])
    return rec, feats, cm


def sig(v):
    """保留 6 位有效数字，压缩 JSON 体积。"""
    return float(f'{v:.6g}')


def main(argv=None):
    args = parse_args(argv)
    log_path = os.path.join(args.out_dir, 'logs', f'{args.name}.json')
    if os.path.exists(log_path) and not args.overwrite:
        print(f'[skip] {log_path} 已存在')
        return
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    src_name, tgt_name = data.parse_task(args.task)
    classes = data.load_classes(args.data_dir)
    xs, ys = data.load_domain(args.data_dir, src_name)
    xt, yt = data.load_domain(args.data_dir, tgt_name)
    src_train, src_val = data.stratified_split(ys, args.src_val_frac, args.split_seed)
    src_bank = data.ImageBank(xs, device, args.crop)
    tgt_bank = data.ImageBank(xt, device, args.crop)
    del xs, xt

    # 三个独立的生成器：源域顺序与增强、目标域顺序与增强、评估抽样。
    # 同一种子下不同 λ 的运行，fc8 初始化、数据顺序、增强、Dropout 掩码都相同，便于配对比较。
    gens = [torch.Generator().manual_seed(args.seed * 1000 + k) for k in range(3)]
    src_stream = data.LabeledStream(src_bank, ys, src_train, args.batch_size, gens[0])
    tgt_stream = data.UnlabeledStream(tgt_bank, np.arange(len(tgt_bank)), args.batch_size, gens[1])

    def eval_set(bank, idx, labels):
        return bank, torch.as_tensor(idx, dtype=torch.long), torch.as_tensor(labels[idx], dtype=torch.long).to(device)

    eval_sets = {'src': eval_set(src_bank, src_train, ys), 'tgt': eval_set(tgt_bank, np.arange(len(yt)), yt)}
    if len(src_val):
        eval_sets['src_val'] = eval_set(src_bank, src_val, ys)
    layers = [l for l in LAYERS if l in {'fc7', 'fc8', *args.align_layers}]

    model = AlexNet(len(classes), pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.SGD(param_groups(model, args.lr, args.weight_decay), lr=args.lr, momentum=args.momentum)

    history, cls_trace, coral_trace, steps = [], [], [], []
    t0 = time.time()
    for it in range(args.iters + 1):
        if it % args.eval_every == 0 or it == args.iters:
            rec, feats, cm = evaluate(model, eval_sets, layers, args.align_layers,
                                      args.batch_size, args.eval_batches, gens[2])
            if steps:
                s = torch.stack(steps).cpu()
                cls_trace += [sig(v) for v in s[:, 0].tolist()]
                coral_trace += [sig(v) for v in s[:, 1].tolist()]
                interval = {'cls_loss': float(s[:, 0].mean()), 'coral_loss': float(s[:, 1].mean()),
                            'src_batch_acc': float(s[:, 2].mean())}
            else:
                interval = {'cls_loss': None, 'coral_loss': None, 'src_batch_acc': None}
            steps = []
            history.append({'iter': it, 'lam': args.lam, 'lr': lr_at(args, it), **interval, **rec})
            print(f'[{args.name}] it {it:5d} | cls {interval["cls_loss"] or 0:.4f} '
                  f'coral {interval["coral_loss"] or 0:.3e} | src {rec["src_acc"]:.4f} '
                  f'tgt {rec["tgt_acc"]:.4f} | {time.time() - t0:.0f}s', flush=True)
        if rec['diverged'] or not np.isfinite(interval['cls_loss'] or 0) or not np.isfinite(interval['coral_loss'] or 0):
            history[-1]['diverged'] = True
            print(f'[{args.name}] 第 {it} 步出现 NaN / Inf，提前停止（λ 可能过大）', flush=True)
            break
        if it == args.iters:
            break

        lr = lr_at(args, it)
        for g in opt.param_groups:
            g['lr'] = lr * g['lr_mult']
        xs_b, ys_b = src_stream.next()
        xt_b = tgt_stream.next()
        out_s, out_t = model(xs_b), model(xt_b)
        loss_cls = F.cross_entropy(out_s['fc8'], ys_b)
        loss_coral = sum(coral_loss(out_s[l], out_t[l]) for l in args.align_layers)
        loss = loss_cls + args.lam * loss_coral if args.lam else loss_cls
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        steps.append(torch.stack([loss_cls.detach(), loss_coral.detach(),
                                  (out_s['fc8'].argmax(1) == ys_b).float().mean()]))

    config = dict(vars(args), src_domain=src_name, tgt_domain=tgt_name, n_src_train=len(src_train),
                  n_src_val=len(src_val), n_tgt=len(yt), classes=classes)
    final = dict(history[-1], tgt_per_class_acc=metrics.per_class_accuracy(cm).tolist(), tgt_confusion=cm.tolist())
    log = {'name': args.name, 'config': config, 'code_version': code_version(), 'hardware': hardware_info(device),
           'time_sec': time.time() - t0, 'history': history, 'final': final,
           'trace': {'cls_loss': cls_trace, 'coral_loss': coral_trace}}

    if args.save_features:
        arrays = {}
        for k, f in feats.items():
            for l, v in f.items():
                arrays[f'{k}_{l}'] = v.cpu().numpy().astype(np.float16 if v.shape[1] > 256 else np.float32)
            arrays[f'{k}_y'] = eval_sets[k][2].cpu().numpy()
        os.makedirs(os.path.join(args.out_dir, 'features'), exist_ok=True)
        np.savez(os.path.join(args.out_dir, 'features', f'{args.name}.npz'), **arrays)
    if args.save_ckpt:
        os.makedirs(os.path.join(args.out_dir, 'ckpt'), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.out_dir, 'ckpt', f'{args.name}.pt'))
    save_json(log, log_path)  # 最后写日志：日志存在即代表整个运行（含特征、权重）已完成


if __name__ == '__main__':
    main()
