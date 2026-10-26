# Deep CORAL 复现与拓展

## 1. 克隆仓库（WSL）

```bash
git clone https://github.com/Cheng-quan-Z/deepcoral_1.git
cd deepcoral_1
```

## 2. 安装环境

```bash
conda create -n deepcoral python=3.10 -y
conda activate deepcoral
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 3. 生成数据缓存

缓存放在 WSL 自己的文件系统里，读取比 /mnt/e 快得多。

```bash
python prepare_data.py --root /path/to/office31 --out ~/data_cache
```

## 4. 冒烟测试

```bash
python tests/smoke_test.py
```

## 5. 阶段一：无适应基线

跑完先看 `stage1_reproduction/results/stage1_summary.md` 里的“基线验收”。

```bash
DATA_DIR=~/data_cache bash stage1_reproduction/run_stage1.sh baseline
```

## 6. 阶段一：标定 λ* 并跑 Deep CORAL

```bash
DATA_DIR=~/data_cache bash stage1_reproduction/run_stage1.sh coral
```

## 7. 阶段一：对照组（可选）

```bash
DATA_DIR=~/data_cache bash stage1_reproduction/run_stage1_controls.sh
```

## 8. 重新汇总

```bash
bash stage1_reproduction/run_stage1.sh analyze
```
