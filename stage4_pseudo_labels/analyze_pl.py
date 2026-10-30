"""阶段四 改进 3 的汇总：主结果（与之前各阶段同种子配对比较）、三种伪标签的质量对比、消融、训练曲线。

python stage4_pseudo_labels/analyze_pl.py

输出 <out_dir>/pl_summary.md、<out_dir>/pl_runs.csv、<out_dir>/figures/curves.png。
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
# (键, 实验名, 配置, 列名)；s3cc 与 s4cc 的实验名都是 cc，按配置区分
METHODS = [('base', 'base', 'lam0', '基线'), ('coral', 'coral', 'lamoff', 'Deep CORAL'),
           ('coralim', 'coralim', 'b0p3', 'CORAL + IM'), ('s3cc', 'cc', 'g0p3', '+ 类条件（置信度伪标签）'),
           ('s4cc', 'cc', 'cent', '+ 类条件（类中心伪标签）')]
COMPARE = [('s4cc', 'coralim', '类中心 − (CORAL+IM)'), ('s4cc', 's3cc', '类中心 − 置信度'),
           ('s4cc', 'coral', '类中心 − Deep CORAL')]
ABLATION = METHODS[2:] + [('agree', 'cc', 'agree', '+ 类条件（一致样本）'), ('st', 'ccst', 'cent', '+ 类条件 + 自训练')]


def select(runs, key, task):
    _, exp, cfg, _ = next(m for m in METHODS + ABLATION if m[0] == key)
    return runs.get((exp, cfg, task), {})


def pct(v):
    return f'{100 * v:.1f}' if v is not None and np.isfinite(v) else '—'


def main_table(runs):
    lines = ['## 主结果（目标域准确率 %，最后一轮）', '',
             '各方法只统计共有的种子，配对差值按同种子计算。基线、Deep CORAL 来自阶段一，CORAL + IM 来自阶段二，'
             '置信度伪标签的类条件来自阶段三。', '',
             '| 任务 | ' + ' | '.join(m[3] for m in METHODS) + ' | ' + ' | '.join(c for *_, c in COMPARE) + ' |',
             '|---|' + '---|' * (len(METHODS) + len(COMPARE))]
    per_task, gains_by_seed = {}, {c: {} for *_, c in COMPARE}
    for t in TASKS:
        m = {k: select(runs, k, t) for k, *_ in METHODS}
        seeds = sorted(set.intersection(*(set(v) for v in m.values())))
        if not seeds:
            continue
        m = {k: {s: v[s] for s in seeds} for k, v in m.items()}
        per_task[t] = {k: np.mean([final_acc(l) for l in v.values()]) for k, v in m.items()}
        cells = [ms([final_acc(l) for l in m[k].values()]) for k, *_ in METHODS]
        for a, b, name in COMPARE:
            diffs = paired_diffs(m[b], m[a])
            gain, sig = fmt_ci(ci(list(diffs.values())))
            cells.append(f'{gain}{"" if sig in ("是", "—") else "，不显著"}')
            for s, d in diffs.items():
                gains_by_seed[name].setdefault(s, []).append(d)
        lines.append(f'| {ARROW[t]} | ' + ' | '.join(cells) + ' |')
    if per_task:
        k = len(per_task)
        avg = [f'{np.mean([v[key] for v in per_task.values()]):.1f}' for key, *_ in METHODS]
        for *_, name in COMPARE:
            avg.append(fmt_ci(ci([np.mean(g) for g in gains_by_seed[name].values() if len(g) == k]))[0])
        lines.append(f'| 平均（{k} 个任务） | ' + ' | '.join(avg) + ' |')
    return lines + ['', '“不显著”指 95% 置信区间跨 0。', '']


def quality_table(runs):
    """同一批模型上三种伪标签的覆盖率和准确率：置信度 ≥ 0.9、类中心、类中心与预测一致。"""
    rows = [t for t in TASKS if select(runs, 's4cc', t)]
    if not rows:
        return []
    lines = ['## 伪标签质量（类中心伪标签主实验的模型，种子均值，覆盖率 / 准确率 %）', '',
             '| 任务 | 时刻 | 预测准确率 | 置信度 ≥ 0.9 | 类中心 | 类中心与预测一致 |', '|---|---|---|---|---|---|']
    for t in rows:
        logs = list(select(runs, 's4cc', t).values())
        start = logs[0]['config']['cc_start']
        for label, pick in ((f'第 {start} 步', lambda h: next((x for x in h if x['iter'] >= start), h[-1])),
                            ('最后一轮', lambda h: h[-1])):
            hs = [pick(l['history']) for l in logs]

            def avg(key):
                vals = [h.get(key) for h in hs if h.get(key) is not None]
                return np.mean(vals) if vals else None

            lines.append(f'| {ARROW[t]} | {label} | {pct(avg("tgt_acc"))} | {pct(avg("tgt_conf_frac"))} / '
                         f'{pct(avg("tgt_conf_acc"))} | 100.0 / {pct(avg("tgt_pl_acc"))} | '
                         f'{pct(avg("tgt_pl_agree_frac"))} / {pct(avg("tgt_pl_agree_acc"))} |')
    return lines + ['', '类中心伪标签覆盖全部目标样本，所以它的准确率要和“预测准确率”比；置信度筛选和“一致”筛选'
                    '只保留部分样本，看的是保留下来的那部分是否更准。', '']


def ablation_table(runs):
    if not any(select(runs, k, t).get(1) for k in ('agree', 'st') for t in TASKS):
        return []
    lines = ['## 消融（种子 1，目标域准确率 %）', '', '| 任务 | ' + ' | '.join(m[3] for m in ABLATION) + ' |',
             '|---|' + '---|' * len(ABLATION)]
    for t in TASKS:
        logs = [select(runs, m[0], t).get(1) for m in ABLATION]
        if any(logs):
            lines.append(f'| {ARROW[t]} | ' + ' | '.join(f'{final_acc(l):.1f}' if l else '—' for l in logs) + ' |')
    return lines + ['', '类条件各列都在 CORAL + IM 之上，γ = 0.3；自训练权重 0.3。', '']


def plot_curves(runs, path):
    tasks = [t for t in TASKS if select(runs, 's4cc', t)]
    if not tasks:
        return False
    fig, axes = plt.subplots(2, len(tasks), figsize=(4.5 * len(tasks), 7), squeeze=False)
    styles = {'coralim': ('C3', '--', 'CORAL + IM'), 's3cc': ('C4', ':', '+ class-cond. (confidence labels)'),
              's4cc': ('C2', '-', '+ class-cond. (centroid labels)')}
    for j, t in enumerate(tasks):
        for key, (color, style, label) in styles.items():
            by_seed = select(runs, key, t)
            if by_seed:
                it, m = curve(by_seed.values(), 'tgt_acc')
                axes[0, j].plot(it, 100 * m, style, color=color, label=label)
        logs = list(select(runs, 's4cc', t).values())
        for key, color, label in (('tgt_acc', 'C7', 'all predictions'), ('tgt_conf_acc', 'C4', 'confidence >= 0.9'),
                                  ('tgt_pl_acc', 'C2', 'centroid labels'), ('tgt_pl_agree_acc', 'C1', 'centroid = prediction')):
            it, m = curve(logs, key)
            axes[1, j].plot(it, 100 * m, color=color, label=label)
        axes[1, j].axvline(logs[0]['config']['cc_start'], color='gray', ls=':', lw=1)
        axes[0, j].set(title=f'{t}: target accuracy (%)', xlabel='iteration')
        axes[1, j].set(title=f'{t}: pseudo-label accuracy (%)', xlabel='iteration')
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def write_csv(runs, path):
    keys = ['tgt_acc', 'tgt_bal_acc', 'tgt_ece', 'tgt_conf_frac', 'tgt_conf_acc', 'tgt_pl_acc', 'tgt_pl_agree_frac',
            'tgt_pl_agree_acc']
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['name', 'exp', 'task', 'seed', 'pl_source', 'st_weight', *keys])
        for (exp, _, task), by_seed in sorted(runs.items()):
            for seed, log in sorted(by_seed.items()):
                cfg = log['config']
                w.writerow([log['name'], exp, task, seed, cfg.get('pl_source', 'pred'), cfg.get('st_weight', 0),
                            *[log['final'].get(k, '') for k in keys]])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out_dir', default=os.path.join(HERE, 'results'))
    p.add_argument('--stage1_dir', default=os.path.join(ROOT, 'stage1_reproduction', 'results'))
    p.add_argument('--stage2_dir', default=os.path.join(ROOT, 'stage2_improvement', 'results'))
    p.add_argument('--stage3_dir', default=os.path.join(ROOT, 'stage3_class_conditional', 'results'))
    args = p.parse_args()
    runs = load_runs(os.path.join(args.stage1_dir, 'logs'), exps=('base', 'coral'))
    runs.update(load_runs(os.path.join(args.stage2_dir, 'logs'), exps=('coralim',)))
    runs.update(load_runs(os.path.join(args.stage3_dir, 'logs'), exps=('cc',)))
    own = load_runs(os.path.join(args.out_dir, 'logs'))
    runs.update(own)

    lines = ['# 阶段四 改进 3：类中心伪标签', '']
    lines += main_table(runs)
    lines += quality_table(runs)
    lines += ablation_table(runs)
    fig_dir = os.path.join(args.out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    if plot_curves(runs, os.path.join(fig_dir, 'curves.png')):
        lines += ['## 训练曲线（种子均值；灰色虚线为类条件对齐开始的步数）', '', '![curves](figures/curves.png)', '']

    write_csv(own, os.path.join(args.out_dir, 'pl_runs.csv'))
    text = '\n'.join(lines)
    with open(os.path.join(args.out_dir, 'pl_summary.md'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
