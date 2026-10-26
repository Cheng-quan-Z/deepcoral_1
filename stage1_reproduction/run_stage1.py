"""阶段一：忠实复现。Windows（PowerShell）和 WSL / Linux 通用。

python stage1_reproduction/run_stage1.py <阶段> --data_dir <caffe 缩放的缓存>

阶段：
  baseline     无适应基线（λ = 0），6 任务 × 5 种子。跑完先看 results/stage1_summary.md 的“基线验收”
  coral        Deep CORAL，使用原作者日志中每个任务的 λ（主结果），6 任务 × 5 种子
  rule         按方案 1.3 的规则（两项损失相等）不用目标标签标定 λ*，再跑 6 任务 × 5 种子
  shared       所有任务共用 A→W 标定出的 λ*（方案 1.3 第 4 步，需先跑 rule）
  attribution  差距归因：每次只把一项设置换回 torchvision 风格，跑无适应基线，6 任务 × 3 种子
  analyze      只做汇总
  all          依次 baseline、coral、rule

所有运行都使用 train.py --preset caffe（CaffeNet、BGR 减均值、227 裁剪、step 学习率、测试随机翻转）。
已完成的运行会被跳过，中断后重跑同一命令即可续跑。未识别的参数原样传给 train.py。
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OFFICIAL = os.path.join(HERE, 'official_reference.json')
TASKS = ['AW', 'AD', 'DA', 'DW', 'WA', 'WD']
SAVE = ['--save_features', '--save_ckpt']  # 后续阶段离线分析要用；权重每个约 230 MB

# 差距归因：在原文配置上每次只改一项（--attr_pil 需要 PIL 缩放的缓存）
ATTRIBUTION = {
    'const': ['--lr_schedule', 'const'],
    'pil': [],  # 只换数据缓存
    'tv': ['--backbone', 'torchvision', '--crop', '224'],
    'tvall': ['--backbone', 'torchvision', '--crop', '224', '--lr_schedule', 'const', '--no-test_mirror'],
}


def train(args, extra, task, seed, lam, name, *more, data_dir=None):
    cmd = [sys.executable, os.path.join(ROOT, 'train.py'), '--preset', 'caffe', '--task', task, '--seed', str(seed),
           '--lam', repr(lam), '--name', name, '--out_dir', args.out_dir,
           '--data_dir', data_dir or args.data_dir, *more, *extra]
    print('>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def official_lambda():
    with open(OFFICIAL, encoding='utf-8') as f:
        return {t: v['lam'] for t, v in json.load(f)['tasks'].items()}


def lambda_star(args, task):
    with open(os.path.join(args.out_dir, 'lambda_star.json'), encoding='utf-8') as f:
        return json.load(f)[task]['lam_star']


def warn_if_baseline_failed(args):
    sys.path.insert(0, HERE)
    from analyze_stage1 import PAPER_CNN, final_acc, load_runs

    runs = load_runs(os.path.join(args.out_dir, 'logs'))
    tasks = [t for t in args.tasks if runs.get(('base', t))]
    if not tasks:
        print('[提醒] 还没有基线结果，建议先跑 baseline 并通过验收', flush=True)
        return
    ours = sum(sum(final_acc(l) for l in runs[('base', t)].values()) / len(runs[('base', t)]) for t in tasks) / len(tasks)
    paper = sum(PAPER_CNN[t] for t in tasks) / len(tasks)
    if abs(ours - paper) > 2:
        print(f'[提醒] 基线平均 {ours:.1f}，原文 {paper:.1f}，未通过验收；方案建议先排查基线再跑 CORAL', flush=True)


def run_baseline(args, extra):
    for task in args.tasks:
        for seed in args.seeds:
            train(args, extra, task, seed, 0.0, f's1-base-{task}-lam0-seed{seed}', *SAVE)


def run_coral(args, extra):
    warn_if_baseline_failed(args)
    lam = official_lambda()
    for task in args.tasks:
        for seed in args.seeds:
            train(args, extra, task, seed, lam[task], f's1-coral-{task}-lamoff-seed{seed}', *SAVE)


def run_rule(args, extra):
    warn_if_baseline_failed(args)
    cmd = [sys.executable, os.path.join(HERE, 'calibrate_lambda.py'), '--tasks', *args.tasks,
           '--seed', str(args.calib_seed), '--out_dir', args.out_dir, '--exp', 'coralrule',
           '--data_dir', args.data_dir, '--preset', 'caffe', *SAVE, *extra]
    print('>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    for task in args.tasks:
        for seed in args.seeds:
            train(args, extra, task, seed, lambda_star(args, task), f's1-coralrule-{task}-lamstar-seed{seed}', *SAVE)


def run_shared(args, extra):
    lam = lambda_star(args, 'AW')
    for task in args.tasks:
        if task != 'AW':  # A→W 上两种做法相同
            for seed in args.seeds:
                train(args, extra, task, seed, lam, f's1-sharedlam-{task}-lamAW-seed{seed}')


def run_attribution(args, extra):
    for cfg, more in ATTRIBUTION.items():
        if cfg in ('pil', 'tvall') and not args.data_dir_pil:
            print(f'[跳过] 归因配置 {cfg} 需要 --data_dir_pil（prepare_data.py --interp pil 生成的缓存）', flush=True)
            continue
        data_dir = args.data_dir_pil if cfg in ('pil', 'tvall') else args.data_dir
        for task in args.tasks:
            for seed in args.attr_seeds:
                train(args, extra, task, seed, 0.0, f's1-attr-{task}-{cfg}-seed{seed}', *more, data_dir=data_dir)


PHASES = {'baseline': [run_baseline], 'coral': [run_coral], 'rule': [run_rule], 'shared': [run_shared],
          'attribution': [run_attribution], 'analyze': [], 'all': [run_baseline, run_coral, run_rule]}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split('\n', 2)[2])
    p.add_argument('phase', choices=list(PHASES))
    p.add_argument('--data_dir', default='data_cache', help='prepare_data.py --interp caffe 生成的缓存')
    p.add_argument('--data_dir_pil', default=None, help='prepare_data.py --interp pil 生成的缓存，只有 attribution 用')
    p.add_argument('--out_dir', default=os.path.join(HERE, 'results'))
    p.add_argument('--tasks', nargs='+', default=TASKS)
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--attr_seeds', nargs='+', type=int, default=[1, 2, 3])
    p.add_argument('--calib_seed', type=int, default=1)
    args, extra = p.parse_known_args()
    for step in PHASES[args.phase]:
        step(args, extra)
    subprocess.run([sys.executable, os.path.join(HERE, 'analyze_stage1.py'), '--out_dir', args.out_dir], check=True)


if __name__ == '__main__':
    main()
