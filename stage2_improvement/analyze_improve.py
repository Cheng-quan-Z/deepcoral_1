"""阶段二 改进 1 的汇总：主结果（与阶段一同种子的基线、Deep CORAL 配对比较）、β 敏感性、
多样性项消融、目标准确率与预测熵曲线。

python stage2_improvement/analyze_improve.py

输出 <out_dir>/improve_summary.md、<out_dir>/improve_runs.csv、<out_dir>/figures/curves.png。
部分运行缺失时只汇总已有的部分。
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'stage1_reproduction'))
from analyze_stage1 import ARROW, NAME_RE, ci, curve, final_acc, fmt_ci, ms, paired_diffs  # noqa: E402

TASKS = ['AW', 'AD', 'DA', 'WA']
MAIN = 'b0p3'
METHODS = [('base', '基线'), ('coral', 'Deep CORAL'), ('im', 'IM'), ('coralim', 'CORAL + IM')]
METHOD_CFG = {'base': 'lam0', 'coral': 'lamoff', 'im': MAIN, 'coralim': MAIN}
COMPARE = [('coralim', 'coral', 'CORAL+IM − CORAL'), ('im', 'base', 'IM − 基线'), ('coralim', 'im', 'CORAL+IM − IM')]


def load_runs(log_dir, exps=None):
    """{(实验名, 配置, 任务): {种子: 日志}}。与阶段一不同，配置也作为键，同一实验不同 β 的运行不会互相覆盖。"""
    runs = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(log_dir, '*.json'))):
        with open(path, encoding='utf-8') as f:
            log = json.load(f)
        m = NAME_RE.match(log['name'])
        if m and (exps is None or m['exp'] in exps):
            runs[(m['exp'], m['cfg'], m['task'])][int(m['seed'])] = log
    return runs


def select(runs, exp, task, cfg=None):
    return runs.get((exp, cfg or METHOD_CFG[exp], task), {})


def method_runs(runs, task):
    return {exp: select(runs, exp, task) for exp, _ in METHODS}


def main_table(runs):
    lines = ['## 主结果（目标域准确率 %，最后一轮；β = 0.3）', '',
             '各方法只统计四种方法共有的种子，配对差值也按同种子计算。基线和 Deep CORAL 来自阶段一。', '',
             '| 任务 | ' + ' | '.join(n for _, n in METHODS) + ' | ' + ' | '.join(c for *_, c in COMPARE) + ' |',
             '|---|' + '---|' * (len(METHODS) + len(COMPARE))]
    per_task, gains_by_seed = {}, {c: {} for *_, c in COMPARE}
    for t in TASKS:
        m = method_runs(runs, t)
        seeds = sorted(set.intersection(*(set(v) for v in m.values())))
        if not seeds:
            continue
        m = {e: {s: v[s] for s in seeds} for e, v in m.items()}
        per_task[t] = {e: np.mean([final_acc(l) for l in v.values()]) for e, v in m.items()}
        cells = [ms([final_acc(l) for l in m[e].values()]) for e, _ in METHODS]
        for a, b, name in COMPARE:
            diffs = paired_diffs(m[b], m[a])
            gain, sig = fmt_ci(ci(list(diffs.values())))
            cells.append(f'{gain}{"" if sig in ("是", "—") else "，不显著"}')
            for s, d in diffs.items():
                gains_by_seed[name].setdefault(s, []).append(d)
        lines.append(f'| {ARROW[t]} | ' + ' | '.join(cells) + ' |')
    if per_task:
        k = len(per_task)
        avg = [f'{np.mean([v[e] for v in per_task.values()]):.1f}' for e, _ in METHODS]
        for *_, name in COMPARE:
            full = [np.mean(g) for g in gains_by_seed[name].values() if len(g) == k]
            avg.append(fmt_ci(ci(full))[0])
        lines.append(f'| 平均（{k} 个任务） | ' + ' | '.join(avg) + ' |')
    return lines + ['', '“不显著”指 95% 置信区间跨 0。', '']


def sensitivity_table(runs):
    cfgs = {cfg: l['config']['ent_weight'] for (exp, cfg, _), by_seed in runs.items() if exp == 'coralim'
            for l in by_seed.values()}
    if len(cfgs) < 2:
        return []
    cfgs = sorted(cfgs, key=cfgs.get)
    lines = ['## β 敏感性（CORAL + IM，种子 1，目标域准确率 %）', '',
             '| 任务 | Deep CORAL | ' + ' | '.join(f'β = {c[1:].replace("p", ".")}' for c in cfgs) + ' |',
             '|---|---|' + '---|' * len(cfgs)]
    for t in TASKS:
        logs = [select(runs, 'coralim', t, c).get(1) for c in cfgs]
        if any(logs):
            coral = select(runs, 'coral', t).get(1)
            lines.append(f'| {ARROW[t]} | {f"{final_acc(coral):.1f}" if coral else "—"} | '
                         + ' | '.join(f'{final_acc(l):.1f}' if l else '—' for l in logs) + ' |')
    return lines + ['', 'β = 0.3 是事先定好的主设置；这张表只说明结果对 β 的敏感程度，不用来挑 β。', '']


def ablation_table(runs):
    rows = [t for t in TASKS if select(runs, 'coralent', t, MAIN)]
    if not rows:
        return []
    lines = ['## 消融：去掉多样性项（种子 1，目标域准确率 %；括号内为目标预测覆盖的类别数 / 31）', '',
             '| 任务 | Deep CORAL | CORAL + 只做熵最小化 | CORAL + IM |', '|---|---|---|---|']

    def cell(log):
        if not log:
            return '—'
        covered = sum(1 for col in np.array(log['final']['tgt_confusion']).T if col.sum() > 0)
        return f'{final_acc(log):.1f}（{covered}）'

    for t in rows:
        lines.append(f'| {ARROW[t]} | {cell(select(runs, "coral", t).get(1))} | '
                     f'{cell(select(runs, "coralent", t, MAIN).get(1))} | {cell(select(runs, "coralim", t, MAIN).get(1))} |')
    return lines + ['', '只做熵最小化时，预测容易塌到少数类，覆盖的类别数会减少；多样性项就是为防止这一点。', '']


def plot_curves(runs, path):
    tasks = [t for t in TASKS if any(method_runs(runs, t).values())]
    if not tasks:
        return False
    fig, axes = plt.subplots(2, len(tasks), figsize=(4.5 * len(tasks), 7), squeeze=False)
    styles = {'base': ('C7', '--'), 'coral': ('C0', '-'), 'im': ('C2', '--'), 'coralim': ('C3', '-')}
    labels = {'base': 'no adaptation', 'coral': 'Deep CORAL', 'im': 'IM', 'coralim': 'CORAL + IM'}
    for j, t in enumerate(tasks):
        for exp, by_seed in method_runs(runs, t).items():
            if not by_seed:
                continue
            color, style = styles[exp]
            for i, key in enumerate(('tgt_acc', 'tgt_entropy')):
                it, m = curve(by_seed.values(), key)
                axes[i, j].plot(it, 100 * m if key == 'tgt_acc' else m, style, color=color, label=labels[exp])
        axes[0, j].set(title=f'{t}: target accuracy (%)', xlabel='iteration')
        axes[1, j].set(title=f'{t}: mean target prediction entropy', xlabel='iteration')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def write_csv(runs, path):
    keys = ['tgt_acc', 'tgt_bal_acc', 'src_acc', 'tgt_ece', 'tgt_nll', 'tgt_entropy', 'tgt_snd', 'coral_full_fc8']
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['name', 'exp', 'task', 'seed', 'lam', 'ent_weight', 'div_weight', *keys])
        for (exp, _, task), by_seed in sorted(runs.items()):
            for seed, log in sorted(by_seed.items()):
                cfg = log['config']
                w.writerow([log['name'], exp, task, seed, cfg['lam'], cfg.get('ent_weight', 0), cfg.get('div_weight', 0),
                            *[log['final'].get(k, '') for k in keys]])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out_dir', default=os.path.join(HERE, 'results'))
    p.add_argument('--stage1_dir', default=os.path.join(ROOT, 'stage1_reproduction', 'results'))
    args = p.parse_args()
    runs = load_runs(os.path.join(args.stage1_dir, 'logs'), exps=('base', 'coral'))
    runs.update(load_runs(os.path.join(args.out_dir, 'logs')))

    lines = ['# 阶段二 改进 1：目标域信息最大化（IM）', '']
    lines += main_table(runs)
    lines += sensitivity_table(runs)
    lines += ablation_table(runs)
    fig_dir = os.path.join(args.out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    if plot_curves(runs, os.path.join(fig_dir, 'curves.png')):
        lines += ['## 训练曲线（种子均值）', '', '![curves](figures/curves.png)', '']

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv({k: v for k, v in runs.items() if k[0] not in ('base', 'coral')},
              os.path.join(args.out_dir, 'improve_runs.csv'))
    text = '\n'.join(lines)
    with open(os.path.join(args.out_dir, 'improve_summary.md'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
