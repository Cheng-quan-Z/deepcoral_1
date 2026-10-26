"""阶段一结果汇总：主结果表（配对差值 + 95% CI）、1.5 验收检查、λ 标定表、两组对照、图 2 复现曲线。

python stage1_reproduction/analyze_stage1.py --out_dir stage1_reproduction/results

输出 <out_dir>/stage1_summary.md、<out_dir>/stage1_runs.csv、<out_dir>/figures/fig2_<任务>.png。
部分运行缺失时只汇总已有的部分。
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

TASKS = ['AD', 'AW', 'DA', 'DW', 'WA', 'WD']  # 原文表 1 的列顺序
ARROW = {t: f'{t[0]}→{t[1]}' for t in TASKS}
PAPER_CNN = {'AD': 63.8, 'AW': 61.6, 'DA': 51.1, 'DW': 95.4, 'WA': 49.8, 'WD': 99.0}
PAPER_CORAL = {'AD': 66.8, 'AW': 66.4, 'DA': 52.8, 'DW': 95.7, 'WA': 51.5, 'WD': 99.2}
NAME_RE = re.compile(r'^(?P<stage>s\d+)-(?P<exp>[a-z]+)-(?P<task>[ADW]{2})-(?P<cfg>[^-]+)-seed(?P<seed>\d+)$')


def load_runs(log_dir):
    """{(实验名, 任务): {种子: 日志}}"""
    runs = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(log_dir, '*.json'))):
        with open(path, encoding='utf-8') as f:
            log = json.load(f)
        m = NAME_RE.match(log['name'])
        if m:
            runs[(m['exp'], m['task'])][int(m['seed'])] = log
    return runs


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def final_acc(log):
    return 100 * log['final']['tgt_acc']


def best_acc(log):
    return 100 * max(h['tgt_acc'] for h in log['history'])


def ms(vals):
    """“均值 ± 标准差（种子数）”"""
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return '—'
    sd = f'{vals.std(ddof=1):.1f}' if len(vals) > 1 else '—'
    return f'{vals.mean():.1f} ± {sd} ({len(vals)})'


def ci(diffs):
    """配对差值的均值与 95% 置信区间（t 分布，n = 5 时临界值 2.776）。"""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n < 2:
        return d.mean() if n else np.nan, np.nan, np.nan, n
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return d.mean(), d.mean() - h, d.mean() + h, n


def paired_diffs(a, b, key=final_acc):
    """同种子配对：b − a，返回 {种子: 差值}。"""
    return {s: key(b[s]) - key(a[s]) for s in sorted(set(a) & set(b))}


def fmt_ci(c):
    m, lo, hi, n = c
    if n == 0:
        return '—', '—'
    if n < 2:
        return f'{m:+.1f} (1)', '—'
    sig = '是' if lo > 0 or hi < 0 else '否（跨 0）'
    return f'{m:+.1f} [{lo:+.1f}, {hi:+.1f}] ({n})', sig


def main_table(runs, lam_table):
    lines = ['## 主结果（目标域准确率 %，最后一轮）', '',
             '| 任务 | 原文 CNN | 基线 | 原文 D-CORAL | Deep CORAL | 原文增益 | 配对增益 [95% CI] | 显著 | D-CORAL 最佳一轮（仅参考） | λ* |',
             '|---|---|---|---|---|---|---|---|---|---|']
    base_means, coral_means, gains_by_seed = {}, {}, defaultdict(list)
    for t in TASKS:
        base, cor = runs.get(('base', t), {}), runs.get(('coral', t), {})
        if not base and not cor:
            continue
        diffs = paired_diffs(base, cor)
        gain, sig = fmt_ci(ci(list(diffs.values())))
        lam = f'{lam_table[t]["lam_star"]:.3g}' if t in lam_table else '—'
        lines.append(f'| {ARROW[t]} | {PAPER_CNN[t]} | {ms([final_acc(l) for l in base.values()])} | {PAPER_CORAL[t]} | '
                     f'{ms([final_acc(l) for l in cor.values()])} | {PAPER_CORAL[t] - PAPER_CNN[t]:+.1f} | {gain} | {sig} | '
                     f'{ms([best_acc(l) for l in cor.values()])} | {lam} |')
        if base and cor:
            base_means[t] = np.mean([final_acc(l) for l in base.values()])
            coral_means[t] = np.mean([final_acc(l) for l in cor.values()])
            for s, d in diffs.items():
                gains_by_seed[s].append(d)
    # 平均行只统计两种方法都有结果的任务；配对增益先在每个种子内对任务求平均，再对种子求 CI
    done = list(base_means)
    if done:
        full_seeds = [s for s, g in gains_by_seed.items() if len(g) == len(done)]
        gain, sig = fmt_ci(ci([np.mean(gains_by_seed[s]) for s in full_seeds]))
        avg = lambda d: np.mean([d[t] for t in done])  # noqa: E731
        lines.append(f'| 平均（{len(done)} 个任务） | {avg(PAPER_CNN):.1f} | {avg(base_means):.1f} | {avg(PAPER_CORAL):.1f} | '
                     f'{avg(coral_means):.1f} | {avg(PAPER_CORAL) - avg(PAPER_CNN):+.1f} | {gain} | {sig} | | |')
    lines += ['', '“最佳一轮”是用目标标签挑出的迭代步，相当于用目标标签选模型，只作参考上界。', '']
    return lines


def curve(logs, key):
    """多个种子同一指标按迭代对齐后取均值，跳过缺失值（如第 0 步没有训练损失、发散后提前停止）。"""
    logs = list(logs)
    its = sorted({h['iter'] for l in logs for h in l['history'] if h.get(key) is not None})
    pos = {it: i for i, it in enumerate(its)}
    vals = np.full((len(logs), len(its)), np.nan)
    for r, l in enumerate(logs):
        for h in l['history']:
            if h.get(key) is not None:
                vals[r, pos[h['iter']]] = h[key]
    return np.array(its), np.nanmean(vals, 0)


def acceptance(runs):
    lines = ['## 验收检查（方案 1.5）', '']
    base_tasks = [t for t in TASKS if runs.get(('base', t))]
    if base_tasks:
        avg = np.mean([np.mean([final_acc(l) for l in runs[('base', t)].values()]) for t in base_tasks])
        paper = np.mean([PAPER_CNN[t] for t in base_tasks])
        ok = abs(avg - paper) <= 2
        lines.append(f'1. 基线：{len(base_tasks)} 个任务平均 {avg:.1f}，原文对应任务平均 {paper:.1f}，'
                     f'差 {avg - paper:+.1f} → {"通过" if ok else "未通过：先排查网络与预处理（见 1.6），不要去调 CORAL"}')

    aw_base, aw_cor = runs.get(('base', 'AW'), {}), runs.get(('coral', 'AW'), {})
    if aw_base and aw_cor:
        lam = next(iter(aw_cor.values()))['config']['lam']
        it, cls = curve(aw_cor.values(), 'cls_loss')
        _, cor = curve(aw_cor.values(), 'coral_loss')
        _, full_b = curve(aw_base.values(), 'coral_full_fc8')
        _, full_c = curve(aw_cor.values(), 'coral_full_fc8')
        early, late = lam * cor[0] / cls[0], lam * cor[-1] / cls[-1]
        lines.append(f'2. 图 2 现象（A→W，供人工判断）：第 {it[0]} 步 λ·L_coral / L_cls = {early:.3f}（应远小于 1），'
                     f'结束时 = {late:.3f}（应接近 1）；λ = 0 时全量 CORAL 距离从 {full_b[0]:.3e} 到 {full_b[-1]:.3e}'
                     f'（应持续增大），结束时是 Deep CORAL 的 {full_b[-1] / full_c[-1]:.1f} 倍（应显著大于 1）')

    gains = {t: ci(list(paired_diffs(runs.get(('base', t), {}), runs.get(('coral', t), {})).values())) for t in TASKS}
    gains = {t: g for t, g in gains.items() if g[3] > 0}
    if gains:
        parts = [f'{ARROW[t]} {gains[t][0]:+.1f}（{"为正" if gains[t][0] > 0 else "不为正"}）' for t in ('AW', 'AD') if t in gains]
        lines.append(f'3. 增益方向：{"；".join(parts) or "A→W、A→D 尚无结果"}')
        if len(gains) == len(TASKS):
            avg = np.mean([g[0] for g in gains.values()])
            lines.append(f'   6 任务平均增益 {avg:+.1f}，原文 +2.0 → {"在 ±1 以内，复现成功" if abs(avg - 2.0) <= 1 else "超出 ±1"}')
        ns = [ARROW[t] for t, g in gains.items() if g[3] >= 2 and g[1] <= 0 <= g[2]]
        lines.append(f'4. 增益不显著（95% CI 跨 0）的任务：{"、".join(ns) if ns else "无"}')
    return lines + ['']


def calib_table(lam_table, title):
    if not lam_table:
        return []
    lines = [f'## λ 标定：{title}', '', '| 任务 | λ=1 时的损失比 | 各轮 λ 与结束时 L_cls/(λ·L_coral) | 最终 λ* | 在 2 倍以内 |',
             '|---|---|---|---|---|']
    for t in TASKS:
        if t in lam_table:
            e = lam_table[t]
            rounds = '；'.join(f'{r["lam"]:.3g} → {r["cls_over_weighted_coral"]:.2f}' for r in e['rounds'])
            lines.append(f'| {ARROW[t]} | {e["ratio_lam1"]:.3g} | {rounds} | {e["lam_star"]:.3g} | {"是" if e["accepted"] else "否"} |')
    return lines + ['']


def shared_table(runs, lam_table):
    rows = [t for t in TASKS if runs.get(('sharedlam', t))]
    if not rows:
        return []
    lam_aw = lam_table.get('AW', {}).get('lam_star', np.nan)
    lines = [f'## 对照：所有任务共用 A→W 的 λ*（{lam_aw:.3g}）', '',
             '| 任务 | 自身 λ* | 增益（自身 λ*） | 增益（共用 λ） | 共用 − 自身 [95% CI] | 显著 |', '|---|---|---|---|---|---|']
    for t in rows:
        base, own, shared = runs.get(('base', t), {}), runs.get(('coral', t), {}), runs[('sharedlam', t)]
        g_own = fmt_ci(ci(list(paired_diffs(base, own).values())))[0]
        g_sh = fmt_ci(ci(list(paired_diffs(base, shared).values())))[0]
        d, sig = fmt_ci(ci(list(paired_diffs(own, shared).values())))
        lam = f'{lam_table[t]["lam_star"]:.3g}' if t in lam_table else '—'
        lines.append(f'| {ARROW[t]} | {lam} | {g_own} | {g_sh} | {d} | {sig} |')
    return lines + ['']


def inv_table(runs, lam_inv):
    rows = [t for t in TASKS if runs.get(('baseinv', t)) or runs.get(('coralinv', t))]
    if not rows:
        return []
    lines = ['## 对照：inv 学习率调度 lr0 / (1 + 10p)^0.75', '',
             '| 任务 | 基线（固定） | 基线（inv） | D-CORAL（固定） | D-CORAL（inv） | 配对增益（inv） | 显著 | λ*（inv） |',
             '|---|---|---|---|---|---|---|---|']
    for t in rows:
        b, c = runs.get(('baseinv', t), {}), runs.get(('coralinv', t), {})
        gain, sig = fmt_ci(ci(list(paired_diffs(b, c).values())))
        lam = f'{lam_inv[t]["lam_star"]:.3g}' if t in lam_inv else '—'
        lines.append(f'| {ARROW[t]} | {ms([final_acc(l) for l in runs.get(("base", t), {}).values()])} | '
                     f'{ms([final_acc(l) for l in b.values()])} | {ms([final_acc(l) for l in runs.get(("coral", t), {}).values()])} | '
                     f'{ms([final_acc(l) for l in c.values()])} | {gain} | {sig} | {lam} |')
    return lines + ['']


def plot_fig2(task, base, cor, path):
    """复现原文图 2：(a) 源 / 目标准确率，(b) 两项损失，(c) CORAL 距离（λ = 0 即原文 2(c)）。"""
    lam = next(iter(cor.values()))['config']['lam']
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    ax = axes[0]
    for logs, style, method in ((cor, '-', 'D-CORAL'), (base, '--', 'no adaptation')):
        for key, color, what in (('src_acc', 'C0', 'source'), ('tgt_acc', 'C1', 'target')):
            it, m = curve(logs.values(), key)
            ax.plot(it, 100 * m, style, color=color, label=f'{what}, {method}')
    ax.set(xlabel='iteration', ylabel='accuracy (%)', title=f'(a) {task}: accuracy (eval mode)')
    ax.legend(fontsize=8)

    ax = axes[1]
    it, cls = curve(cor.values(), 'cls_loss')
    _, c = curve(cor.values(), 'coral_loss')
    ax.plot(it, cls, label='classification loss')
    ax.plot(it, lam * c, label=f'CORAL loss x lambda ({lam:.3g})')
    ax.set(xlabel='iteration', ylabel='loss (mean over 100 iters)', title='(b) D-CORAL losses')
    ax.legend(fontsize=8)

    ax = axes[2]
    for logs, style, method in ((cor, '-', 'D-CORAL'), (base, '--', 'lambda=0')):
        for key, color, what in (('coral_loss', 'C2', 'batch, train mode'), ('coral_evalbatch_fc8', 'C4', 'batch, eval mode'),
                                 ('coral_full_fc8', 'C3', 'full data, eval mode')):
            it, m = curve(logs.values(), key)
            ax.plot(it, m, style, color=color, label=f'{what}, {method}')
    ax.set(xlabel='iteration', ylabel='CORAL distance (fc8)', yscale='log', title='(c) CORAL distance')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_csv(runs, path):
    keys = ['tgt_acc', 'tgt_bal_acc', 'src_acc', 'src_val_acc', 'tgt_ece', 'tgt_nll', 'tgt_entropy', 'tgt_snd',
            'coral_full_fc8', 'coral_full_fc7', 'erank_fc7_src', 'erank_fc7_tgt', 'erank_fc8_src', 'erank_fc8_tgt']
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['name', 'exp', 'task', 'seed', 'lam', 'lr_schedule', 'best_tgt_acc', 'best_iter', *keys])
        for (exp, task), by_seed in sorted(runs.items()):
            for seed, log in sorted(by_seed.items()):
                best = max(log['history'], key=lambda h: h['tgt_acc'])
                w.writerow([log['name'], exp, task, seed, log['config']['lam'], log['config']['lr_schedule'],
                            best['tgt_acc'], best['iter'], *[log['final'].get(k, '') for k in keys]])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out_dir', default='stage1_reproduction/results')
    args = p.parse_args()
    runs = load_runs(os.path.join(args.out_dir, 'logs'))
    lam_main = load_json(os.path.join(args.out_dir, 'lambda_star.json'))
    lam_inv = load_json(os.path.join(args.out_dir, 'lambda_star_inv.json'))

    lines = ['# 阶段一结果汇总', '']
    lines += main_table(runs, lam_main)
    lines += acceptance(runs)
    lines += calib_table(lam_main, '主实验（固定学习率）')
    lines += calib_table(lam_inv, 'inv 学习率调度')
    lines += shared_table(runs, lam_main)
    lines += inv_table(runs, lam_inv)

    fig_dir = os.path.join(args.out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    figs = []
    for t in TASKS:
        if runs.get(('base', t)) and runs.get(('coral', t)):
            plot_fig2(t, runs[('base', t)], runs[('coral', t)], os.path.join(fig_dir, f'fig2_{t}.png'))
            figs.append(f'![{ARROW[t]}](figures/fig2_{t}.png)')
    if figs:
        lines += ['## 图 2 复现（种子均值）', ''] + figs + ['']

    write_csv(runs, os.path.join(args.out_dir, 'stage1_runs.csv'))
    text = '\n'.join(lines)
    with open(os.path.join(args.out_dir, 'stage1_summary.md'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
