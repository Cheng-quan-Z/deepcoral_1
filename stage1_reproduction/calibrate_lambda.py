"""阶段一 1.3：按“训练结束时分类损失与 CORAL 损失大致相等”的规则标定 λ*，全程不使用目标标签。

每个任务：
  1. λ = 1 训练一次，λ* = 最后 200 次迭代的平均分类损失 / 平均（未乘 λ 的）CORAL 损失；
  2. 用 λ* 训练，检查最后 200 次迭代的 L_cls / (λ*·L_coral)；若超出 [1/2, 2]，用这次的新比值再训练一次；
  3. 最后那次运行与正式实验中同种子的 Deep CORAL 运行配置完全相同，直接登记为正式结果，省一次训练。

结果写入 --lambda_file（默认 <out_dir>/lambda_star.json）。未识别的参数原样透传给 train.py。
例：python stage1_reproduction/calibrate_lambda.py --tasks AW DA --out_dir stage1_reproduction/results --data_dir data_cache
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW = 200
TOL = 2.0
SAVE_FLAGS = ('--save_features', '--save_ckpt')


def load_log(out_dir, name):
    with open(os.path.join(out_dir, 'logs', f'{name}.json'), encoding='utf-8') as f:
        return json.load(f)


def loss_means(log, window=WINDOW):
    """最后 window 次迭代的平均分类损失与平均（未加权）CORAL 损失。"""
    if log['final'].get('diverged'):
        raise SystemExit(f'{log["name"]}：训练发散，无法标定 λ')
    cls = log['trace']['cls_loss'][-window:]
    coral = log['trace']['coral_loss'][-window:]
    cls_mean, coral_mean = sum(cls) / len(cls), sum(coral) / len(coral)
    if not coral_mean > 0:
        raise SystemExit(f'{log["name"]}：CORAL 损失均值为 {coral_mean}，无法标定 λ')
    return cls_mean, coral_mean


def train(task, name, lam, args, extra):
    cmd = [sys.executable, os.path.join(ROOT, 'train.py'), '--task', task, '--seed', str(args.seed),
           '--lam', repr(lam), '--name', name, '--out_dir', args.out_dir, *extra]
    print('>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return load_log(args.out_dir, name)


def promote(out_dir, src, dst):
    """把标定中的最终运行登记为正式运行 dst：复制日志，移动特征与权重。"""
    log = load_log(out_dir, src)
    log['name'] = log['config']['name'] = dst
    log['promoted_from'] = src
    for sub, ext in (('features', '.npz'), ('ckpt', '.pt')):
        path = os.path.join(out_dir, sub, src + ext)
        if os.path.exists(path):
            shutil.move(path, os.path.join(out_dir, sub, dst + ext))
    with open(os.path.join(out_dir, 'logs', f'{dst}.json'), 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False)


def calibrate(task, args, extra):
    tag = f'{args.prefix}-{args.calib_exp}-{task}'
    no_save = [a for a in extra if a not in SAVE_FLAGS]
    cls, coral = loss_means(train(task, f'{tag}-lam1-seed{args.seed}', 1.0, args, no_save))
    lam = cls / coral
    entry = {'seed': args.seed, 'ratio_lam1': lam, 'rounds': []}
    for r in (1, 2):
        name = f'{tag}-lamstar{r}-seed{args.seed}'
        cls, coral = loss_means(train(task, name, lam, args, extra))
        check = cls / (lam * coral)
        entry['rounds'].append({'lam': lam, 'run': name, 'cls_over_weighted_coral': check})
        print(f'[{task}] 第 {r} 轮 λ = {lam:.4g}，结束时 L_cls / (λ·L_coral) = {check:.3f}', flush=True)
        if 1 / TOL <= check <= TOL or r == 2:
            break
        lam = cls / coral  # 超出 2 倍：用新比值再训练一次（方案规定只再迭代一次）
    entry.update(lam_star=lam, final_run=name, accepted=1 / TOL <= check <= TOL)
    return entry


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--tasks', nargs='+', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--seed', type=int, default=1, help='标定所用种子，同时也是被登记的正式运行的种子')
    p.add_argument('--prefix', default='s1')
    p.add_argument('--exp', default='coral', help='正式 Deep CORAL 运行的实验名')
    p.add_argument('--calib_exp', default='calib', help='标定运行的实验名')
    p.add_argument('--lambda_file', default=None, help='默认 <out_dir>/lambda_star.json')
    args, extra = p.parse_known_args()
    lambda_file = args.lambda_file or os.path.join(args.out_dir, 'lambda_star.json')

    table = {}
    if os.path.exists(lambda_file):
        with open(lambda_file, encoding='utf-8') as f:
            table = json.load(f)
    for task in args.tasks:
        if task not in table:
            table[task] = calibrate(task, args, extra)
            os.makedirs(os.path.dirname(lambda_file), exist_ok=True)
            with open(lambda_file, 'w', encoding='utf-8') as f:
                json.dump(table, f, ensure_ascii=False, indent=1)
        main_name = f'{args.prefix}-{args.exp}-{task}-lamstar-seed{args.seed}'
        if not os.path.exists(os.path.join(args.out_dir, 'logs', f'{main_name}.json')):
            promote(args.out_dir, table[task]['final_run'], main_name)
        print(f'[{task}] λ* = {table[task]["lam_star"]:.4g}', flush=True)


if __name__ == '__main__':
    main()
