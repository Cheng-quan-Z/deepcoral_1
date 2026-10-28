"""阶段二 改进 1：目标域信息最大化（IM），单独使用以及与 Deep CORAL 组合。Windows 和 WSL / Linux 通用。

python stage2_improvement/run_improve.py <阶段> --data_dir <caffe 缩放的缓存>

阶段：
  main         主实验：IM、CORAL + IM，β = 0.3，4 任务 × 3 种子，共 24 次
  sensitivity  β 敏感性：CORAL + IM 取 β = 0.1、1.0，4 任务 × 1 种子，共 8 次
  ablation     消融：CORAL + 只做熵最小化（不加多样性项），4 任务 × 1 种子，共 4 次
  analyze      只做汇总
  all          依次 main、sensitivity、ablation

信息最大化 = β·(目标样本平均预测熵) − β·(目标 batch 平均预测分布的熵)。
β = 0.3 是事先定好的主设置，没有用目标标签挑选；敏感性实验只用来报告结果对 β 的依赖程度。
CORAL 的 λ 沿用阶段一的原作者取值。基线和 Deep CORAL 直接复用阶段一同种子的运行，不重跑。
D→W、W→D 已接近 99%，没有提升空间，默认不跑。
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
MAIN_BETA = 0.3
SENSITIVITY_BETAS = [0.1, 1.0]


def tag(beta):
    """β 写进运行名：0.3 → b0p3，1.0 → b1。"""
    return 'b' + f'{beta:g}'.replace('.', 'p')


def official_lambda():
    with open(OFFICIAL, encoding='utf-8') as f:
        return {t: v['lam'] for t, v in json.load(f)['tasks'].items()}


def train(args, extra, task, seed, name, lam, ent, div):
    cmd = [sys.executable, os.path.join(ROOT, 'train.py'), '--preset', 'caffe', '--task', task, '--seed', str(seed),
           '--lam', repr(lam), '--ent_weight', repr(ent), '--div_weight', repr(div), '--name', name,
           '--out_dir', args.out_dir, '--data_dir', args.data_dir, *extra]
    print('>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_main(args, extra):
    lam = official_lambda()
    for task in args.tasks:
        for seed in args.seeds:
            train(args, extra, task, seed, f's2-im-{task}-{tag(MAIN_BETA)}-seed{seed}', 0.0, MAIN_BETA, MAIN_BETA)
            train(args, extra, task, seed, f's2-coralim-{task}-{tag(MAIN_BETA)}-seed{seed}',
                  lam[task], MAIN_BETA, MAIN_BETA)


def run_sensitivity(args, extra):
    lam = official_lambda()
    for beta in SENSITIVITY_BETAS:
        for task in args.tasks:
            for seed in args.extra_seeds:
                train(args, extra, task, seed, f's2-coralim-{task}-{tag(beta)}-seed{seed}', lam[task], beta, beta)


def run_ablation(args, extra):
    lam = official_lambda()
    for task in args.tasks:
        for seed in args.extra_seeds:
            train(args, extra, task, seed, f's2-coralent-{task}-{tag(MAIN_BETA)}-seed{seed}', lam[task], MAIN_BETA, 0.0)


PHASES = {'main': [run_main], 'sensitivity': [run_sensitivity], 'ablation': [run_ablation], 'analyze': [],
          'all': [run_main, run_sensitivity, run_ablation]}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split('\n', 2)[2])
    p.add_argument('phase', choices=list(PHASES))
    p.add_argument('--data_dir', default='data_cache', help='prepare_data.py --interp caffe 生成的缓存')
    p.add_argument('--out_dir', default=os.path.join(HERE, 'results'))
    p.add_argument('--stage1_dir', default=os.path.join(ROOT, 'stage1_reproduction', 'results'),
                   help='阶段一结果目录，汇总时从这里读基线和 Deep CORAL')
    p.add_argument('--tasks', nargs='+', default=TASKS)
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3], help='主实验的种子')
    p.add_argument('--extra_seeds', nargs='+', type=int, default=[1], help='敏感性和消融实验的种子')
    args, extra = p.parse_known_args()
    for step in PHASES[args.phase]:
        step(args, extra)
    subprocess.run([sys.executable, os.path.join(HERE, 'analyze_improve.py'), '--out_dir', args.out_dir,
                    '--stage1_dir', args.stage1_dir], check=True)


if __name__ == '__main__':
    main()
