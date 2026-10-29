"""阶段三 改进 2 的汇总：主结果（与之前同种子的基线、Deep CORAL、CORAL + IM 配对比较）、伪标签质量、
均值 / 协方差消融、γ 敏感性、训练曲线。

python stage3_class_conditional/analyze_cc.py

输出 <out_dir>/cc_summary.md、<out_dir>/cc_runs.csv、<out_dir>/figures/curves.png。
伪标签准确率要用目标标签计算，只用于分析，训练中没有使用。部分运行缺失时只汇总已有的部分。
"""
import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'stage1_reproduction'))
sys.path.insert(0, os.path.join(ROOT, 'stage2_improvement'))
from analyze_improve import load_runs  # noqa: E402
from analyze_stage1 import ARROW, ci, curve, final_acc, fmt_ci, ms, paired_diffs  # noqa: E402

TASKS = ['AW', 'AD', 'DA', 'WA']
MAIN = 'g0p3'
METHODS = [('base', 'lam0', '基线'), ('coral', 'lamoff', 'Deep CORAL'), ('coralim', 'b0p3', 'CORAL + IM'),
           ('cc', MAIN, 'CORAL + IM + 类条件')]
COMPARE = [('cc', 'coralim', '类条件 − (CORAL+IM)'), ('cc', 'coral', '类条件 − Deep CORAL')]


def select(runs, exp, task, cfg=None):
    return runs.get((exp, cfg or dict((e, c) for e, c, _ in METHODS).get(exp, MAIN), task), {})


def pct(v):
    return f'{100 * v:.1f}' if v is not None and np.isfinite(v) else '—'


def main_table(runs):
    lines = ['## 主结果（目标域准确率 %，最后一轮；γ = 0.3）', '',
             '各方法只统计共有的种子，配对差值按同种子计算。基线、Deep CORAL 来自阶段一，CORAL + IM 来自阶段二。', '',
             '| 任务 | ' + ' | '.join(n for *_, n in METHODS) + ' | ' + ' | '.join(c for *_, c in COMPARE) +
             ' | 伪标签占比 / 准确率 |', '|---|' + '---|' * (len(METHODS) + len(COMPARE) + 1)]
    per_task, gains_by_seed = {}, {c: {} for *_, c in COMPARE}
    for t in TASKS:
        m = {e: select(runs, e, t) for e, *_ in METHODS}
        seeds = sorted(set.intersection(*(set(v) for v in m.values())))
        if not seeds:
            continue
        m = {e: {s: v[s] for s in seeds} for e, v in m.items()}
        per_task[t] = {e: np.mean([final_acc(l) for l in v.values()]) for e, v in m.items()}
        cells = [ms([final_acc(l) for l in m[e].values()]) for e, *_ in METHODS]
        for a, b, name in COMPARE:
            diffs = paired_diffs(m[b], m[a])
            gain, sig = fmt_ci(ci(list(diffs.values())))
            cells.append(f'{gain}{"" if sig in ("是", "—") else "，不显著"}')
            for s, d in diffs.items():
                gains_by_seed[name].setdefault(s, []).append(d)
        cc = list(m['cc'].values())
        cells.append(f'{pct(np.mean([l["final"]["tgt_conf_frac"] for l in cc]))} / '
                     f'{pct(np.mean([l["final"]["tgt_conf_acc"] for l in cc]))}')
        lines.append(f'| {ARROW[t]} | ' + ' | '.join(cells) + ' |')
    if per_task:
        k = len(per_task)
        avg = [f'{np.mean([v[e] for v in per_task.values()]):.1f}' for e, *_ in METHODS]
        for *_, name in COMPARE:
            avg.append(fmt_ci(ci([np.mean(g) for g in gains_by_seed[name].values() if len(g) == k]))[0])
        lines.append(f'| 平均（{k} 个任务） | ' + ' | '.join(avg) + ' | |')
    return lines + ['', '“不显著”指 95% 置信区间跨 0。伪标签占比 / 准确率：最后一轮评估时置信度 ≥ 0.9 的目标样本'
                    '占全部目标样本的比例，以及这些样本的预测准确率（用目标标签计算，仅用于分析）。', '']


def pseudo_label_table(runs):
    """伪标签准确率在训练中是上升还是下降：下降说明错误被不断强化（确认偏差）。"""
    rows = [t for t in TASKS if select(runs, 'cc', t)]
    if not rows:
        return []
    lines = ['## 伪标签质量随训练的变化（CORAL + IM + 类条件，种子均值，%）', '',
             '| 任务 | 启用时占比 / 准确率 | 最后一轮占比 / 准确率 | 准确率变化 |', '|---|---|---|---|']
    for t in rows:
        logs = list(select(runs, 'cc', t).values())
        _, frac = curve(logs, 'tgt_conf_frac')
        it, acc = curve(logs, 'tgt_conf_acc')
        i600 = min(int(np.searchsorted(it, logs[0]['config']['cc_start'])), len(it) - 1)
        lines.append(f'| {ARROW[t]} | {pct(frac[i600])} / {pct(acc[i600])} | {pct(frac[-1])} / {pct(acc[-1])} | '
                     f'{100 * (acc[-1] - acc[i600]):+.1f} |')
    return lines + ['', '第 600 步是类条件对齐开始启用的时刻。准确率下降说明伪标签错误被不断强化（确认偏差）。', '']


def one_seed_table(runs, title, cols, note):
    """种子 1 上的若干配置对比；cols 为 [(实验名, 配置, 列名)]。"""
    if not any(select(runs, e, t, c).get(1) for e, c, _ in cols[1:] for t in TASKS):
        return []
    lines = [f'## {title}（种子 1，目标域准确率 %）', '', '| 任务 | ' + ' | '.join(n for *_, n in cols) + ' |',
             '|---|' + '---|' * len(cols)]
    for t in TASKS:
        logs = [select(runs, e, t, c).get(1) for e, c, _ in cols]
        if any(logs[1:]):
            lines.append(f'| {ARROW[t]} | ' + ' | '.join(f'{final_acc(l):.1f}' if l else '—' for l in logs) + ' |')
    return lines + ['', note, '']


def plot_curves(runs, path):
    tasks = [t for t in TASKS if select(runs, 'cc', t)]
    if not tasks:
        return False
    fig, axes = plt.subplots(2, len(tasks), figsize=(4.5 * len(tasks), 7), squeeze=False)
    styles = {'coral': ('C0', '-', 'Deep CORAL'), 'coralim': ('C3', '--', 'CORAL + IM'),
              'cc': ('C4', '-', 'CORAL + IM + class-conditional')}
    for j, t in enumerate(tasks):
        for exp, (color, style, label) in styles.items():
            by_seed = select(runs, exp, t)
            if by_seed:
                it, m = curve(by_seed.values(), 'tgt_acc')
                axes[0, j].plot(it, 100 * m, style, color=color, label=label)
        logs = list(select(runs, 'cc', t).values())
        for key, color, label in (('tgt_conf_acc', 'C4', 'pseudo-label accuracy'),
                                  ('tgt_conf_frac', 'C7', 'fraction with confidence >= 0.9')):
            it, m = curve(logs, key)
            axes[1, j].plot(it, 100 * m, color=color, label=label)
        axes[1, j].axvline(600, color='gray', ls=':', lw=1)
        axes[0, j].set(title=f'{t}: target accuracy (%)', xlabel='iteration')
        axes[1, j].set(title=f'{t}: pseudo labels (%)', xlabel='iteration')
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def write_csv(runs, path):
    keys = ['tgt_acc', 'tgt_bal_acc', 'src_acc', 'tgt_ece', 'tgt_entropy', 'tgt_conf_frac', 'tgt_conf_acc', 'coral_full_fc8']
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['name', 'exp', 'task', 'seed', 'lam', 'ent_weight', 'cc_weight', 'cc_terms', *keys])
        for (exp, _, task), by_seed in sorted(runs.items()):
            for seed, log in sorted(by_seed.items()):
                cfg = log['config']
                w.writerow([log['name'], exp, task, seed, cfg['lam'], cfg.get('ent_weight', 0), cfg.get('cc_weight', 0),
                            cfg.get('cc_terms', ''), *[log['final'].get(k, '') for k in keys]])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out_dir', default=os.path.join(HERE, 'results'))
    p.add_argument('--stage1_dir', default=os.path.join(ROOT, 'stage1_reproduction', 'results'))
    p.add_argument('--stage2_dir', default=os.path.join(ROOT, 'stage2_improvement', 'results'))
    args = p.parse_args()
    runs = load_runs(os.path.join(args.stage1_dir, 'logs'), exps=('base', 'coral'))
    runs.update(load_runs(os.path.join(args.stage2_dir, 'logs'), exps=('coralim',)))
    own = load_runs(os.path.join(args.out_dir, 'logs'))
    runs.update(own)

    lines = ['# 阶段三 改进 2：类条件 CORAL', '']
    lines += main_table(runs)
    lines += pseudo_label_table(runs)
    lines += one_seed_table(
        runs, '消融：类条件项对齐什么',
        [('coralim', 'b0p3', 'CORAL + IM'), ('ccmean', MAIN, '+ 只对齐类均值'), ('cccov', MAIN, '+ 只对齐类协方差'),
         ('cc', MAIN, '+ 均值和协方差')],
        '三种类条件变体都加在 CORAL + IM 之上，γ = 0.3。')
    lines += one_seed_table(
        runs, 'γ 敏感性',
        [('coralim', 'b0p3', 'CORAL + IM（γ = 0）'), ('cc', 'g0p1', 'γ = 0.1'), ('cc', MAIN, 'γ = 0.3'), ('cc', 'g1', 'γ = 1')],
        'γ = 0.3 是事先定好的主设置；这张表只说明结果对 γ 的敏感程度，不用来挑 γ。')
    fig_dir = os.path.join(args.out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    if plot_curves(runs, os.path.join(fig_dir, 'curves.png')):
        lines += ['## 训练曲线（种子均值；灰色虚线为类条件对齐开始的第 600 步）', '', '![curves](figures/curves.png)', '']

    write_csv(own, os.path.join(args.out_dir, 'cc_runs.csv'))
    text = '\n'.join(lines)
    with open(os.path.join(args.out_dir, 'cc_summary.md'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
