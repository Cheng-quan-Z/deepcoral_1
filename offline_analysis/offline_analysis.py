"""离线分析：只用各阶段已有的训练日志，不需要再跑实验。

python offline_analysis/offline_analysis.py

  1. 选点规则（方案 2.4）：不用目标标签的停止点规则，与用目标标签挑出的最优点相差多少；
  2. 无监督选 λ、选方法（方案 4.4）：只看无标签指标，能否挑出目标准确率最高的 λ / 方法；
  3. batch 与全量 CORAL 距离（方案 3.2 前半）：batch 估计和 Dropout 各让距离偏大多少；
  4. 有效秩（方案 3.3）：λ 增大时特征是否退化；
  5. 校准性（方案 4.3）：各方法的 ECE、NLL、预测熵。

无标签指标：目标预测平均熵（越小越好）、SND（越大越好）、全量 fc8 CORAL 距离（越小越好）。
目标准确率只用于评价这些规则，不参与选择。
输出 offline_analysis/results/offline_summary.md 和 figures/。
"""
import argparse
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
sys.path.insert(0, os.path.join(ROOT, 'stage2_improvement'))
from analyze_improve import load_runs  # noqa: E402
from analyze_stage1 import ARROW, curve  # noqa: E402

STAGE_DIRS = ['stage1_reproduction', 'stage2_improvement', 'stage3_class_conditional', 'stage4_pseudo_labels']
SPLIT_NOTE = '共用 λ 在 A→W 上沿用规则 λ 的运行（两者 λ 相同）。'
TASKS6 = ['AD', 'AW', 'DA', 'DW', 'WA', 'WD']
TASKS4 = ['AW', 'AD', 'DA', 'WA']
# 主配置：(实验名, 配置, 名称)
METHODS = [('base', 'lam0', '基线'), ('coral', 'lamoff', 'Deep CORAL（原作者 λ）'),
           ('coralrule', 'lamstar', 'Deep CORAL（规则 λ）'), ('sharedlam', 'lamAW', 'Deep CORAL（共用 λ）'),
           ('im', 'b0p3', 'IM'), ('coralim', 'b0p3', 'CORAL + IM'), ('cc', 'g0p3', '+ 类条件（置信度伪标签）'),
           ('cc', 'cent', '+ 类条件（类中心伪标签）')]
# 无标签指标：(键, 名称, 越大越好)
METRICS = [('tgt_entropy', '最小预测熵', False), ('tgt_snd', '最大 SND', True), ('coral_full_fc8', '最小 CORAL 距离', False)]


def name_of(exp, cfg):
    return next(n for e, c, n in METHODS if (e, c) == (exp, cfg))


def runs_of(runs, exp, cfg, tasks):
    """[(任务, 种子, 日志)]"""
    return [(t, s, l) for t in tasks for s, l in sorted(runs.get((exp, cfg, t), {}).items())]


def pick(history, key, larger):
    """按某个无标签指标在训练过程中选一个评估点（不含第 0 步）。"""
    hs = [h for h in history if h['iter'] > 0 and h.get(key) is not None]
    return (max if larger else min)(hs, key=lambda h: h[key])


def stopping_table(runs):
    lines = ['## 1. 选点规则：训练过程中在哪一步停（方案 2.4）', '',
             '每格为“目标准确率 %（平均选中步数）”，最后一列为最优点与“最后一轮”的差。最优点用目标标签挑出，只作上界。', '',
             '| 方法 | 运行数 | 最后一轮 | ' + ' | '.join(n for _, n, _ in METRICS) + ' | 最优点（用目标标签） | 最优 − 最后一轮 |',
             '|---|---|---|' + '---|' * len(METRICS) + '---|---|']
    for exp, cfg, name in METHODS:
        rs = runs_of(runs, exp, cfg, TASKS6)
        if not rs:
            continue
        hists = [l['history'] for *_, l in rs]
        last = np.mean([100 * h[-1]['tgt_acc'] for h in hists])
        best = np.mean([100 * max(x['tgt_acc'] for x in h) for h in hists])
        cells = []
        for key, _, larger in METRICS:
            chosen = [pick(h, key, larger) for h in hists]
            cells.append(f'{np.mean([100 * c["tgt_acc"] for c in chosen]):.1f}（{np.mean([c["iter"] for c in chosen]):.0f}）')
        lines.append(f'| {name} | {len(rs)} | {last:.1f} | ' + ' | '.join(cells) + f' | {best:.1f} | {best - last:+.1f} |')
    return lines + ['', '各规则只在每 100 步一次的评估点中选择，不含第 0 步。', '']


def selection_rows(runs, candidates, tasks, seeds):
    """对每个（任务, 种子），只看无标签指标从候选中挑一个，返回 {规则: [(挑中的准确率, 最优准确率, 挑中的候选)]}。"""
    out = defaultdict(list)
    for t in tasks:
        for s in seeds:
            cand = [(c, runs.get((c[0], c[1], t), {}).get(s)) for c in candidates]
            cand = [(c, l) for c, l in cand if l]
            if len(cand) < 2:
                continue
            oracle = max(100 * l['final']['tgt_acc'] for _, l in cand)
            for key, name, larger in METRICS:
                c, l = (max if larger else min)(cand, key=lambda x: x[1]['final'][key])
                out[name].append((100 * l['final']['tgt_acc'], oracle, c))
            for c, l in cand:
                out[f'固定用 {name_of(*c)}'].append((100 * l['final']['tgt_acc'], oracle, c))
    return out


def selection_table(title, note, rows, candidates):
    lines = [title, '', '| 规则 | 选中的平均准确率 % | 与最优候选的差（遗憾值） | ' +
             ' | '.join(f'选中 {name_of(*c)} 的次数' for c in candidates) + ' |',
             '|---|---|---|' + '---|' * len(candidates)]
    for rule, vals in rows.items():
        if not rule.startswith('固定'):
            counts = [sum(1 for *_, c in vals if c == cand) for cand in candidates]
            lines.append(f'| {rule} | {np.mean([v[0] for v in vals]):.1f} | {np.mean([v[1] - v[0] for v in vals]):.1f} | '
                         + ' | '.join(map(str, counts)) + ' |')
    for rule, vals in rows.items():
        if rule.startswith('固定'):
            lines.append(f'| {rule} | {np.mean([v[0] for v in vals]):.1f} | {np.mean([v[1] - v[0] for v in vals]):.1f} | '
                         + ' | '.join('' for _ in candidates) + ' |')
    return lines + ['', note, '']


def lambda_selection(runs):
    cands = [('base', 'lam0'), ('coral', 'lamoff'), ('coralrule', 'lamstar'), ('sharedlam', 'lamAW')]
    rows = selection_rows(runs, cands, TASKS6, range(1, 6))
    if not rows:
        return []
    return selection_table(
        '## 2a. 无监督选 λ（方案 4.4）：在 λ = 0、原作者 λ、规则 λ、共用 λ 中挑一个',
        '对每个任务的每个种子单独挑选（6 任务 × 5 种子）。' + SPLIT_NOTE, rows, cands)


def method_selection(runs):
    cands = [(e, c) for e, c, _ in METHODS if e != 'sharedlam']
    rows = selection_rows(runs, cands, TASKS4, range(1, 4))
    if not rows:
        return []
    return selection_table(
        '## 2b. 无监督选方法：在全部主配置中挑一个', '对每个任务的每个种子单独挑选（4 任务 × 3 种子）。注意：带 IM 的方法直接最小化预测熵，'
        '用“最小预测熵”挑选天然偏向它们。', rows, cands)


def batch_vs_full(runs):
    lines = ['## 3. batch 上对齐了 ≠ 整个域对齐了（方案 3.2 前半，最后一轮，fc8）', '',
             '训练 batch：训练模式（有 Dropout）下每步 batch 的 CORAL 损失（最后 100 步平均）；评估 batch：评估模式下随机抽 50 对 '
             '128 张的 batch 求平均；全量：全部源域、目标域图像的协方差。', '',
             '| 任务 | 方法 | 训练 batch | 评估 batch | 全量 | 训练 batch / 全量 | 其中 Dropout 造成（训练 / 评估 batch） | '
             '估计误差造成（评估 batch / 全量） | fc7 全量距离 |', '|---|---|---|---|---|---|---|---|---|']
    for t in TASKS6:
        for exp, cfg in (('base', 'lam0'), ('coral', 'lamoff')):
            rs = runs_of(runs, exp, cfg, [t])
            if not rs:
                continue
            tr = np.mean([l['history'][-1]['coral_loss'] for *_, l in rs])
            eb = np.mean([l['final']['coral_evalbatch_fc8'] for *_, l in rs])
            full = np.mean([l['final']['coral_full_fc8'] for *_, l in rs])
            f7 = np.mean([l['final']['coral_full_fc7'] for *_, l in rs])
            lines.append(f'| {ARROW[t]} | {name_of(exp, cfg)} | {tr:.3g} | {eb:.3g} | {full:.3g} | {tr / full:.1f} 倍 | '
                         f'{tr / eb:.2f} 倍 | {eb / full:.2f} 倍 | {f7:.2e} |')
    return lines + ['', '有限样本估计的协方差距离天然偏大（即使两域分布相同，batch 估计的距离也大于 0），'
                    '真实距离越小，这部分偏差占比越大。', '']


def erank_table(runs):
    cols = [('base', 'lam0'), ('coral', 'lamoff'), ('coralrule', 'lamstar'), ('coralim', 'b0p3'), ('cc', 'cent')]
    lines = ['## 4. 有效秩：λ 增大时特征是否退化（方案 3.3，目标域，最后一轮）', '',
             '每格为“fc8 有效秩 / fc7 有效秩，目标准确率 %”。fc8 有效秩最大为 31；fc7 有效秩受样本数限制，只在同一任务内比较。', '',
             '| 任务 | 原作者 λ | 规则 λ | ' + ' | '.join(name_of(*c) for c in cols) + ' |', '|---|---|---|' + '---|' * len(cols)]
    for t in TASKS6:
        if not any(runs_of(runs, exp, cfg, [t]) for exp, cfg in cols):
            continue
        cells, lams = [], {}
        for exp, cfg in cols:
            rs = runs_of(runs, exp, cfg, [t])
            if not rs:
                cells.append('—')
                continue
            lams[exp] = rs[0][2]['config']['lam']
            f = [l['final'] for *_, l in rs]
            cells.append(f'{np.mean([x["erank_fc8_tgt"] for x in f]):.1f} / {np.mean([x["erank_fc7_tgt"] for x in f]):.0f}，'
                         f'{100 * np.mean([x["tgt_acc"] for x in f]):.1f}')
        lam_cells = [f'{lams[e]:.3g}' if e in lams else '—' for e in ('coral', 'coralrule')]
        lines.append(f'| {ARROW[t]} | ' + ' | '.join(lam_cells + cells) + ' |')
    return lines + ['', '原文把 λ 过大时性能下降归因于“退化特征”；若成立，同一任务内 λ 越大，有效秩应越低。', '']


def calibration_table(runs):
    lines = ['## 5. 校准性（方案 4.3，A→W、A→D、D→A、W→A 四个任务、种子 1–3 平均，最后一轮）', '',
             '| 方法 | 目标准确率 % | ECE | NLL | 平均预测熵 |', '|---|---|---|---|---|']
    for exp, cfg, name in METHODS:
        per_task = defaultdict(list)
        for t, s, l in runs_of(runs, exp, cfg, TASKS4):
            if s <= 3:
                per_task[t].append(l['final'])
        if len(per_task) < len(TASKS4):
            continue

        def avg(key, scale=1.0):
            return np.mean([np.mean([scale * f[key] for f in v]) for v in per_task.values()])

        lines.append(f'| {name} | {avg("tgt_acc", 100):.1f} | {avg("tgt_ece"):.3f} | {avg("tgt_nll"):.2f} | '
                     f'{avg("tgt_entropy"):.2f} |')
    return lines + ['', 'ECE 越小，预测置信度与实际准确率越一致。先对每个任务内的种子取平均，再对任务取平均。'
                    '方案中的温度缩放需要源域验证集上的 logits，现有运行没有留验证集、也没有上传特征，这里没有做。', '']


def plot(runs, path):
    fig, axes = plt.subplots(2, len(TASKS4), figsize=(4.5 * len(TASKS4), 7), squeeze=False)
    styles = [('base', 'lam0', 'C7', '--', 'no adaptation'), ('coral', 'lamoff', 'C0', '-', 'Deep CORAL (authors\' lambda)'),
              ('coralrule', 'lamstar', 'C1', '-', 'Deep CORAL (rule lambda)'), ('coralim', 'b0p3', 'C3', '-', 'CORAL + IM')]
    for j, t in enumerate(TASKS4):
        for exp, cfg, color, style, label in styles:
            logs = [l for *_, l in runs_of(runs, exp, cfg, [t])]
            if not logs:
                continue
            for i, key in enumerate(('erank_fc8_tgt', 'tgt_ece')):
                it, m = curve(logs, key)
                axes[i, j].plot(it, m, style, color=color, label=label)
        axes[0, j].set(title=f'{t}: effective rank of target fc8', xlabel='iteration')
        axes[1, j].set(title=f'{t}: target ECE', xlabel='iteration')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--log_dirs', nargs='+', default=[os.path.join(ROOT, d, 'results', 'logs') for d in STAGE_DIRS],
                   help='各阶段的日志目录，默认为仓库里四个阶段的 results/logs')
    p.add_argument('--out_dir', default=os.path.join(HERE, 'results'))
    args = p.parse_args()
    runs = {}
    for d in args.log_dirs:
        runs.update(load_runs(d))
    # 共用 λ 就是 A→W 上标定出的 λ*，在 A→W 上与规则 λ 完全相同，当时没有重复跑，这里直接沿用
    if ('coralrule', 'lamstar', 'AW') in runs:
        runs.setdefault(('sharedlam', 'lamAW', 'AW'), runs[('coralrule', 'lamstar', 'AW')])

    lines = ['# 离线分析（只用已有日志，不需要新实验）', '']
    lines += stopping_table(runs)
    lines += lambda_selection(runs)
    lines += method_selection(runs)
    lines += batch_vs_full(runs)
    lines += erank_table(runs)
    lines += calibration_table(runs)
    os.makedirs(os.path.join(args.out_dir, 'figures'), exist_ok=True)
    plot(runs, os.path.join(args.out_dir, 'figures', 'erank_ece.png'))
    lines += ['## 有效秩与 ECE 随训练的变化（种子均值）', '', '![erank_ece](figures/erank_ece.png)', '']
    text = '\n'.join(lines)
    with open(os.path.join(args.out_dir, 'offline_summary.md'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
