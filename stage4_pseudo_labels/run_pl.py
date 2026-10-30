"""阶段四 改进 3：用特征空间类中心生成伪标签，替换阶段三中基于置信度的伪标签。Windows 和 WSL / Linux 通用。

python stage4_pseudo_labels/run_pl.py <阶段> --data_dir <caffe 缩放的缓存>

阶段：
  main      主实验：CORAL + IM + 类条件对齐，伪标签改用类中心，4 任务 × 3 种子，共 12 次
  ablation  消融（种子 1）：只保留类中心与预测一致的样本（4 次）；类中心伪标签上再加自训练（4 次）
  analyze   只做汇总
  all       依次 main、ablation

阶段三发现：信息最大化让模型过度自信，置信度 ≥ 0.9 的筛选几乎筛不掉错误伪标签。类中心伪标签不看置信度：
每次评估（每 100 步）用全部目标图像的 fc7 特征，先按预测概率加权得到各类中心，再把每个样本分给最近的中心
（余弦距离），用硬标签重算一次中心后重新分配。

事先定好、不用目标标签挑选的设置：类条件对齐与阶段三相同（fc8、均值 + 协方差、γ = 0.3、第 600 步启用、
队列 32、每类至少 4 个样本）；自训练权重 0.3；CORAL 的 λ 用原作者取值，信息最大化 β = 0.3。
对照组复用之前同种子的运行：阶段三的类条件（置信度伪标签）、阶段二的 CORAL + IM、阶段一的基线和 Deep CORAL。
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
GAMMA = 0.3
ST_WEIGHT = 0.3


def official_lambda():
    with open(OFFICIAL, encoding='utf-8') as f:
        return {t: v['lam'] for t, v in json.load(f)['tasks'].items()}


def train(args, extra, task, seed, name, pl_source, st_weight=0.0):
    cmd = [sys.executable, os.path.join(ROOT, 'train.py'), '--preset', 'caffe', '--task', task, '--seed', str(seed),
           '--lam', repr(official_lambda()[task]), '--ent_weight', repr(BETA), '--div_weight', repr(BETA),
           '--cc_weight', repr(GAMMA), '--pl_source', pl_source, '--st_weight', repr(st_weight), '--name', name,
           '--out_dir', args.out_dir, '--data_dir', args.data_dir, *extra]
    print('>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_main(args, extra):
    for task in args.tasks:
        for seed in args.seeds:
            train(args, extra, task, seed, f's4-cc-{task}-cent-seed{seed}', 'centroid')


def run_ablation(args, extra):
    for task in args.tasks:
        for seed in args.extra_seeds:
            train(args, extra, task, seed, f's4-cc-{task}-agree-seed{seed}', 'agree')
            train(args, extra, task, seed, f's4-ccst-{task}-cent-seed{seed}', 'centroid', ST_WEIGHT)


PHASES = {'main': [run_main], 'ablation': [run_ablation], 'analyze': [], 'all': [run_main, run_ablation]}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split('\n', 2)[2])
    p.add_argument('phase', choices=list(PHASES))
    p.add_argument('--data_dir', default='data_cache', help='prepare_data.py --interp caffe 生成的缓存')
    p.add_argument('--out_dir', default=os.path.join(HERE, 'results'))
    p.add_argument('--stage1_dir', default=os.path.join(ROOT, 'stage1_reproduction', 'results'))
    p.add_argument('--stage2_dir', default=os.path.join(ROOT, 'stage2_improvement', 'results'))
    p.add_argument('--stage3_dir', default=os.path.join(ROOT, 'stage3_class_conditional', 'results'))
    p.add_argument('--tasks', nargs='+', default=TASKS)
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3], help='主实验的种子')
    p.add_argument('--extra_seeds', nargs='+', type=int, default=[1], help='消融实验的种子')
    args, extra = p.parse_known_args()
    for step in PHASES[args.phase]:
        step(args, extra)
    subprocess.run([sys.executable, os.path.join(HERE, 'analyze_pl.py'), '--out_dir', args.out_dir,
                    '--stage1_dir', args.stage1_dir, '--stage2_dir', args.stage2_dir,
                    '--stage3_dir', args.stage3_dir], check=True)


if __name__ == '__main__':
    main()
