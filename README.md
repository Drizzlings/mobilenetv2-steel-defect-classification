# MobileNetV2 钢材表面缺陷分类

使用 PyTorch 实现 MobileNetV2，在 NEU-CLS 数据集上完成六类钢材表面缺陷分类，并比较普通卷积、SE/CBAM 注意力、GELU/Mish 激活函数和 GroupNorm 的影响。实验同时评估原始图像及低亮度、低对比度、高斯噪声、模糊四种条件下的分类表现。

## 仓库内容

GitHub 提交包含核心代码、实验配置、依赖清单和运行说明。数据集需要下载，模型权重由训练生成。

```text
inclass3/
├── src/
│   ├── models.py              # MobileNetV2 与八组结构变体
│   ├── data.py                # 数据划分、标准化、增强与扰动
│   ├── download_data.py       # 下载 NEU-CLS 压缩包
│   ├── experiment.py          # 训练、评估与效率测量
│   ├── predict.py             # 推理与 Grad-CAM
│   ├── run_suite.py           # 基础四组实验队列（Linux）
│   ├── run_components.py      # 激活与归一化队列（Linux）
│   ├── checks.py              # 数据、结构、梯度及学习能力检查
│   └── check_cbam.py          # CBAM 结构与预测结果检查
├── configs/
│   ├── config.json
│   ├── config_components.json
│   └── config_cbam.json
├── requirements.txt
├── README.md
└── .gitignore
```

本地另保留 `reports/`（Word、Visio、Markdown 报告）、`tools/`（核验与打包工具）、`data/`（数据与划分）、`outputs/`（权重、日志、结果）。这四个目录由 `.gitignore` 排除，不上传 GitHub。

## 安装环境

原实验环境为 Linux、Python 3.10.21、PyTorch 2.1.2、torchvision 0.16.2，使用 CUDA 11.8 版 PyTorch 和 RTX 4090。训练与完整评估需要 CUDA；推理支持 CPU。

克隆仓库后，进入包含本 README 的项目根目录。以下命令均在该目录执行：

```bash
conda create -n mobilenet-defects python=3.10 -y
conda activate mobilenet-defects
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

## 数据准备

[NEU 官方数据集页面](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/)提供 NEU-CLS 图像分类数据集。共 1800 张 200×200 灰度图，六类各 300 张：裂纹（Cr）、夹杂（In）、斑块（Pa）、麻点（PS）、轧入氧化皮（RS）、划痕（Sc）。

```bash
python src/download_data.py
unrar x -o- data/NEU-CLS.rar data/
python src/data.py
```

下载脚本使用作者提供的 Google Drive 链接，需要能够访问该服务。也可以从官方页面手动下载，用 7-Zip 等软件解压。解压后应能找到 `data/NEU-CLS/Cr_1.bmp` 等 1800 张 BMP 图像，再执行 `python src/data.py`。

划分种子为 2026，训练、验证、测试集各 1080、360、360 张。相同像素及旋转、镜像等价图像归入同一子集。脚本生成 `data/manifest.json` 和 `data/dataset_stats.json`，分别保存划分及训练集统计量。

## 实验配置

八组模型均从零训练，训练种子 42，共 80 轮，batch size 为 32。使用 AdamW、标签平滑交叉熵、5 轮预热和余弦学习率调度；结构相同的层使用相同初始参数。输入保持 200×200 分辨率，灰度复制为三通道。

| `--model` | 修改内容 | 配置文件 |
|---|---|---|
| `mobilenet_v2` | 基线 | `configs/config.json` |
| `mobilenet_v2_dense` | 逐通道 3×3 卷积换为普通卷积 | `configs/config.json` |
| `mobilenet_v2_se` | 投影 BN 后加入 SE | `configs/config.json` |
| `mobilenet_v2_dense_se` | 普通卷积与 SE 组合 | `configs/config.json` |
| `mobilenet_v2_cbam` | 投影 BN 后加入 CBAM | `configs/config_cbam.json` |
| `mobilenet_v2_gelu` | ReLU6 替换为 GELU | `configs/config_components.json` |
| `mobilenet_v2_mish` | ReLU6 替换为 Mish | `configs/config_components.json` |
| `mobilenet_v2_gn` | BN 替换为 GN，组数为 gcd(32,C) | `configs/config_components.json` |

## 训练与评估

准备好数据后，运行基线：

```bash
python src/experiment.py train --model mobilenet_v2 --seed 42 --config configs/config.json
python src/experiment.py evaluate --model mobilenet_v2 --seed 42 --config configs/config.json
```

按上表替换模型名和配置文件即可运行其他变体。结果写入 `outputs/runs/<模型名>_seed42/`。按验证准确率选取 `best.pt`，同分时选择验证交叉熵更低的权重；`last.pt` 用于断点续训：

```bash
python src/experiment.py train --model mobilenet_v2 --seed 42 --config configs/config.json --resume
```

完整评估读取 `best.pt`，生成五种图像条件下的准确率、Macro-F1、混淆矩阵、逐样本预测，以及参数量、卷积/线性层 MACs 和推理速度。

Linux 服务器上可依次运行以下命令完成全部实验：

```bash
CUDA_VISIBLE_DEVICES=0 python src/checks.py
python src/run_suite.py --gpu 0
CUDA_VISIBLE_DEVICES=0 python src/run_components.py
CUDA_VISIBLE_DEVICES=0 python src/experiment.py train --model mobilenet_v2_cbam --seed 42 --config configs/config_cbam.json --resume
CUDA_VISIBLE_DEVICES=0 python src/experiment.py evaluate --model mobilenet_v2_cbam --seed 42 --config configs/config_cbam.json
python src/check_cbam.py
```

`checks.py` 检查基线与 torchvision 实现的一致性、数据划分、反向传播和小样本学习能力。`check_cbam.py` 在 CBAM 评估完成后检查模块与预测结果。两个队列脚本使用 Linux 的 `fcntl` 文件锁，每两小时记录一次进度，当前阶段结束后自动执行下一阶段。可在 tmux 中运行，GPU 编号按空闲情况调整。

Windows PowerShell 使用 `$env:CUDA_VISIBLE_DEVICES='0'` 设置显卡，再执行单组训练或评估命令。队列脚本在 Linux 上运行。已完成的训练会被跳过；重新训练同名实验前请另存原有输出。

## 推理与可视化

训练得到权重后，可进行 CPU 单图推理和 Grad-CAM 可视化：

```bash
python src/predict.py --checkpoint outputs/runs/mobilenet_v2_seed42/best.pt --input data/NEU-CLS/Cr_1.bmp --output outputs/example --device cpu --gradcam
```

文件夹批量推理：

```bash
python src/predict.py --checkpoint outputs/runs/mobilenet_v2_seed42/best.pt --input data/NEU-CLS --output outputs/batch --device cuda:0
```

程序输出类别、概率及 CSV/JSON；添加 `--gradcam` 时额外输出响应图。六个类别均为缺陷类型，未设置正常钢材类别。

## 本次实验结果

以下数值来自已完成的单种子实验；各条件使用同一组 360 张测试图像，单位为 %。

| 模型 | 原始 | 低亮度 | 低对比度 | 高斯噪声 | 模糊 |
|---|---:|---:|---:|---:|---:|
| 基线 | 99.72 | 97.22 | 51.94 | 28.06 | 52.78 |
| Dense | 99.17 | 86.39 | 41.94 | 30.56 | 40.83 |
| SE | 99.72 | 89.44 | 46.67 | 26.39 | 43.61 |
| CBAM | 98.89 | 82.78 | 37.50 | 35.00 | 34.17 |
| Dense+SE | 98.89 | 92.22 | 43.89 | 32.50 | 39.44 |
| GELU | 99.72 | 97.78 | 51.94 | 31.11 | 48.61 |
| Mish | 99.72 | 97.78 | 59.17 | 30.28 | 45.28 |
| GN | 98.61 | 89.44 | 96.94 | 42.50 | 61.11 |

扰动设置见配置文件：亮度系数 0.6、对比度系数 0.5、高斯噪声标准差 0.05、高斯模糊标准差 1.5（9×9 核）。每种扰动独立施加，训练配置保持一致。重新运行的耗时和准确率可能随软硬件环境变化。
