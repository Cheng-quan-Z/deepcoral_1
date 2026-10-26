"""从原作者发布的 Caffe 训练日志（VisionLearningGroup/CORAL 仓库 code/D-CORAL_Office31.zip）提取参考曲线。

python tools/parse_official_logs.py --zip D-CORAL_Office31.zip --out stage1_reproduction/official_reference.json

输出每个任务的 CORAL 权重 λ，以及 5 次运行每 50 步的目标准确率、每 20 步的分类损失与 CORAL 损失。
仓库里已附带生成好的 official_reference.json，一般不需要重新运行。
"""
import argparse
import json
import re
import zipfile
from collections import defaultdict

TASK_DIRS = {'a_d': 'AD', 'a_w': 'AW', 'd_a': 'DA', 'd_w': 'DW', 'w_a': 'WA', 'w_d': 'WD'}


def parse_log(text):
    test_iters = [int(x) for x in re.findall(r'Iteration (\d+), Testing net', text)]
    test_acc = [float(x) for x in re.findall(r'Test net output #0: accuracy = ([\d.]+)', text)]
    train_iters = [int(x) for x in re.findall(r'Iteration (\d+), loss = ', text)]
    cls = [float(x) for x in re.findall(r'Train net output #1: classification_loss = ([\d.e+-]+)', text)]
    coral = [float(x) for x in re.findall(r'Train net output #2: coral_fc8 = ([\d.e+-]+)', text)]
    n = min(len(train_iters), len(cls), len(coral))
    return {'test_iter': test_iters, 'tgt_acc': test_acc,
            'train_iter': train_iters[:n], 'cls_loss': cls[:n], 'coral_loss': coral[:n]}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--zip', required=True)
    p.add_argument('--out', default='stage1_reproduction/official_reference.json')
    args = p.parse_args()

    runs = defaultdict(list)
    lam = {}
    with zipfile.ZipFile(args.zip) as z:
        for name in sorted(z.namelist()):
            m = re.match(r'D-CORAL_Office31/(\w_\w)/coral_\w_\w_log_fc8_loss_([\d.]+)_full_source_iter_\d+\.txt$', name)
            if m:
                task = TASK_DIRS[m.group(1)]
                lam[task] = float(m.group(2))
                runs[task].append(parse_log(z.read(name).decode('utf-8', 'replace')))
    ref = {'source': 'https://github.com/VisionLearningGroup/CORAL/blob/master/code/D-CORAL_Office31.zip',
           'tasks': {t: {'lam': lam[t], 'runs': runs[t]} for t in sorted(runs)}}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(ref, f)
    for t, v in ref['tasks'].items():
        finals = [100 * r['tgt_acc'][-1] for r in v['runs']]
        print(f'{t}: λ = {v["lam"]}，{len(finals)} 次运行，最后一轮平均 {sum(finals) / len(finals):.1f}')


if __name__ == '__main__':
    main()
