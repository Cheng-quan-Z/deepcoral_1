"""阶段三 改进 2：类条件 CORAL，加在 CORAL + 信息最大化之上。Windows 和 WSL / Linux 通用。

python stage3_class_conditional/run_cc.py <阶段> --data_dir <caffe 缩放的缓存>

阶段：
  main         主实验：CORAL + IM + 类条件对齐（均值 + 协方差），γ = 0.3，4 任务 × 3 种子，共 12 次
  ablation     消融：类条件只对齐均值 / 只对齐协方差，4 任务 × 1 种子，共 8 次
  sensitivity  γ 敏感性：γ = 0.1、1.0，4 任务 × 1 种子，共 8 次
  analyze      只做汇总
  all          依次 main、ablation、sensitivity

事先定好、不用目标标签挑选的设置：
  - 在 fc8 上按类对齐；源域用真实标签，目标域用当前 batch 自身预测中置信度 ≥ 0.9 的样本作伪标签；
  - 前 30% 迭代（600 步）不启用；每个域、每个类维护 32 个样本的特征队列，某类两域样本都 ≥ 4 个才计入；
  - 权重 γ = 0.3；CORAL 的 λ 沿用原作者取值，信息最大化 β = 0.3（与阶段二相同）。
对照组复用之前同种子的运行：CORAL + IM 来自阶段二，基线和 Deep CORAL 来自阶段一。
所有运行使用 train.py --preset caffe。已完成的运行会被跳过；未识别的参数原样传给 train.py。
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OFFICIAL = os.path.join(ROOT, 'stage1_reproduction', 'official_reference.json')
TASKS = ['AW', 'AD', 'DA', 'WA']
BETA = 0.3
MAIN_GAMMA = 0.3
SENSITIVITY_GAMMAS = [0.1, 1.0]
ABLATION = {'ccmean': 'mean', 'cccov': 'cov'}


def tag(gamma):
    """γ 写进运行名：0.3 → g0p3，1.0 → g1。"""
    return 'g' + f'{gamma:g}'.replace('.', 'p')


def official_lambda():
    with open(OFFICIAL, encoding='utf-8') as f:
        return {t: v['lam'] for t, v in json.load(f)['tasks'].items()}


def train(args, extra, task, seed, name, gamma, terms):
    cmd = [sys.executable, os.path.join(ROOT, 'train.py'), '--preset', 'caffe', '--task', task, '--seed', str(seed),
           '--lam', repr(official_lambda()[task]), '--ent_weight', repr(BETA), '--div_weight', repr(BETA),
           '--cc_weight', repr(gamma), '--cc_terms', terms, '--name', name,
           '--out_dir', args.out_dir, '--data_dir', args.data_dir, *extra]
    print('>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_main(args, extra):
    for task in args.tasks:
        for seed in args.seeds:
            train(args, extra, task, seed, f's3-cc-{task}-{tag(MAIN_GAMMA)}-seed{seed}', MAIN_GAMMA, 'both')


def run_ablation(args, extra):
    for exp, terms in ABLATION.items():
        for task in args.tasks:
            for seed in args.extra_seeds:
                train(args, extra, task, seed, f's3-{exp}-{task}-{tag(MAIN_GAMMA)}-seed{seed}', MAIN_GAMMA, terms)


def run_sensitivity(args, extra):
    for gamma in SENSITIVITY_GAMMAS:
        for task in args.tasks:
            for seed in args.extra_seeds:
                train(args, extra, task, seed, f's3-cc-{task}-{tag(gamma)}-seed{seed}', gamma, 'both')


PHASES = {'main': [run_main], 'ablation': [run_ablation], 'sensitivity': [run_sensitivity], 'analyze': [],
          'all': [run_main, run_ablation, run_sensitivity]}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split('\n', 2)[2])
    p.add_argument('phase', choices=list(PHASES))
    p.add_argument('--data_dir', default='data_cache', help='prepare_data.py --interp caffe 生成的缓存')
    p.add_argument('--out_dir', default=os.path.join(HERE, 'results'))
    p.add_argument('--stage1_dir', default=os.path.join(ROOT, 'stage1_reproduction', 'results'))
    p.add_argument('--stage2_dir', default=os.path.join(ROOT, 'stage2_improvement', 'results'))
    p.add_argument('--tasks', nargs='+', default=TASKS)
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3], help='主实验的种子')
    p.add_argument('--extra_seeds', nargs='+', type=int, default=[1], help='消融和敏感性实验的种子')
    args, extra = p.parse_known_args()
    for step in PHASES[args.phase]:
        step(args, extra)
    subprocess.run([sys.executable, os.path.join(HERE, 'analyze_cc.py'), '--out_dir', args.out_dir,
                    '--stage1_dir', args.stage1_dir, '--stage2_dir', args.stage2_dir], check=True)


if __name__ == '__main__':
    main()
