# Deep CORAL 复现与拓展

下面的命令在 Windows PowerShell 和 WSL / Linux 里都能用；两边写法不同的地方分开给出。所有命令都在仓库根目录 `deepcoral_1` 下执行。

## 1. 克隆仓库

```bash
git clone https://github.com/Cheng-quan-Z/deepcoral_1.git
cd deepcoral_1
```

已经克隆过的，拉取最新代码：

```bash
git pull
```

## 2. 安装环境

```bash
conda create -n deepcoral python=3.11 -y
conda activate deepcoral
```

先装 CUDA 版 PyTorch。CUDA 版本不能高于 `nvidia-smi` 右上角显示的版本；RTX 50 系列显卡需要 cu128 或更高。具体命令以 https://pytorch.org/get-started/locally/ 为准，下面以 cu128 为例：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

确认 GPU 可用，应输出 `True`：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 3. 下载 CaffeNet 预训练权重

原文用的是 Caffe 官方的 BVLC Reference CaffeNet。权重文件约 233 MB，不放在仓库里，需要自己下载到 `weights/` 目录。

- 下载地址：http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
- 完整大小：243,862,418 字节
- SHA1：`4c8d77deb20ea792f84eb5e6d0a11ca0a8660a46`

这个服务器速度较慢，而且经常中途断开，下载命令有时会“正常结束”但文件不完整。下面的命令都支持断点续传：文件大小不到 243,862,418 字节时，重新运行同一条命令即可接着下载，直到大小对上。

WSL / Linux：

```bash
mkdir -p weights
wget -c -O weights/bvlc_reference_caffenet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
```

Windows PowerShell（注意是 `curl.exe`，不是 PowerShell 自带的 `curl` 别名）：

```powershell
mkdir weights -Force
curl.exe -L -C - -o weights/bvlc_reference_caffenet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
```

也可以用浏览器打开下载地址，把文件保存为 `weights/bvlc_reference_caffenet.caffemodel`。

查看文件大小（WSL 用第一条，PowerShell 用第二条）：

```bash
stat -c %s weights/bvlc_reference_caffenet.caffemodel
```

```powershell
(Get-Item weights/bvlc_reference_caffenet.caffemodel).Length
```

## 4. 转换权重

把 Caffe 格式转成 PyTorch 格式，不需要安装 Caffe：

```bash
python tools/convert_caffenet.py
```

第一行应显示 `SHA1 4c8d77de...（与官方一致）`。如果显示警告或报错，说明文件没下完整，回到第 3 步重新运行下载命令续传。成功后生成 `weights/caffenet_imagenet.pth`。

## 5. 生成数据缓存

Office-31 已在仓库的 `data/office31` 里。主实验用和 Caffe 一样的缩放方式：

```bash
python prepare_data.py --interp caffe --out data_cache
```

第 12 步的差距归因还需要一份 PIL 缩放的缓存（不做归因可以跳过）：

```bash
python prepare_data.py --interp pil --out data_cache_pil
```

三个域应分别显示 2817、498、795 张，且没有“警告”字样。

## 6. 检查权重与预处理

用 ImageNet 分类头直接识别 Office-31 里的背包、键盘、鼠标等图片：

```bash
python tools/check_caffenet.py --data_dir data_cache
```

最后一行应为“检查通过”，平均命中率约 0.8。

## 7. 冒烟测试

用假数据把整个流程跑一遍，约 3–5 分钟，最后一行应为 `[ok] 全部通过`：

```bash
python tests/smoke_test.py
```

## 8. 阶段一：无适应基线

6 个任务 × 5 个种子，共 30 次，每次约 4 分钟：

```bash
python stage1_reproduction/run_stage1.py baseline --data_dir data_cache
```

跑完打开 `stage1_reproduction/results/stage1_summary.md`，看“验收检查”第 1 条。显示“通过”再继续第 9 步；未通过就先停下，把结果按第 13 步上传。

## 9. 阶段一：Deep CORAL（原作者的 λ）

主结果，30 次：

```bash
python stage1_reproduction/run_stage1.py coral --data_dir data_cache
```

## 10. 阶段一：Deep CORAL（规则标定的 λ）

不用目标标签、按“两项损失相等”标定 λ*，再跑 30 次（标定另需 6–12 次）：

```bash
python stage1_reproduction/run_stage1.py rule --data_dir data_cache
```

## 11. 阶段一：共用 λ 对照（可选）

所有任务都用 A→W 标定出的 λ*，需先完成第 10 步，25 次：

```bash
python stage1_reproduction/run_stage1.py shared --data_dir data_cache
```

## 12. 阶段一：差距归因（可选）

每次只把一项设置换回最初版本，跑无适应基线，4 组 × 6 任务 × 3 种子，共 72 次：

```bash
python stage1_reproduction/run_stage1.py attribution --data_dir data_cache --data_dir_pil data_cache_pil
```

## 13. 上传结果

每一步跑完都会自动更新 `stage1_reproduction/results/stage1_summary.md`。只上传日志、表格和图；模型权重（`ckpt/`）和特征（`features/`）太大，不要上传。

```bash
git add -f stage1_reproduction/results/logs stage1_reproduction/results/lambda_star.json stage1_reproduction/results/stage1_summary.md stage1_reproduction/results/stage1_runs.csv stage1_reproduction/results/figures
git commit -m "Stage 1 results"
git pull --rebase
git push
```

如果第 10 步还没跑，`lambda_star.json` 不存在，把它从第一条命令里删掉即可。

## 14. 阶段二：改进 1（目标域信息最大化）

需要第 8、9 步的结果已经在 `stage1_reproduction/results/logs` 里（从仓库拉下来的也可以）。先拉取最新代码：

```bash
git pull
```

主实验，4 个任务 × 3 个种子 × 2 种方法，共 24 次，每次约 7 分钟：

```bash
python stage2_improvement/run_improve.py main --data_dir data_cache
```

第一次运行开始后，看一眼终端输出：每 100 步打印的 `tgt` 应在 0.4–0.7 之间逐步上升，不应出现“NaN / Inf，提前停止”。

β 敏感性（8 次）和多样性项消融（4 次），可选：

```bash
python stage2_improvement/run_improve.py sensitivity --data_dir data_cache
python stage2_improvement/run_improve.py ablation --data_dir data_cache
```

跑完查看 `stage2_improvement/results/improve_summary.md`。上传方法与第 13 步相同：

```bash
git add -f stage2_improvement/results/logs stage2_improvement/results/improve_summary.md stage2_improvement/results/improve_runs.csv stage2_improvement/results/figures
git commit -m "Stage 2 results"
git pull --rebase
git push
```

## 中断与续跑

任何一步中断后，重新运行同一条命令即可，已完成的运行会自动跳过。
