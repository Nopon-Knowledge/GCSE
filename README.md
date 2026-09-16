# GCSE 运行说明

## 1. 安装环境

使用 Python 3.10 或更高版本，在项目根目录执行以下命令。完整训练建议使用 NVIDIA GPU，分布式训练需要 CUDA 环境。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell 激活虚拟环境时使用 `.venv\Scripts\Activate.ps1`。

检查模型是否可以运行（无需数据集或权重）：

```bash
python -c "import torch; from gcse import gcse_resnet50; model = gcse_resnet50().eval(); print(model(torch.randn(1, 3, 64, 64)).shape)"
```

输出应为 `torch.Size([1, 1000])`。运行 `python -c "import torch; print(torch.cuda.is_available())"` 可检查 GPU 是否可用。

## 2. ImageNet 分类训练与验证

准备 ImageNet-1K 数据，训练集与验证集都需要按类别放入子目录：

```text
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

训练 GCSE-ResNet-50：

```bash
python train.py \
  --data-root /path/to/imagenet \
  --model gcse_resnet50 \
  --epochs 90 \
  --batch-size 256 \
  --seed 3407 \
  --output runs/imagenet/gcse_resnet50/seed3407 \
  --verify-paper-protocol
```

请将示例路径替换为实际数据路径。若只做小规模试运行，移除 `--verify-paper-protocol`，再调整 `--epochs`、`--batch-size` 和 `--workers`；使用 CPU 时增加 `--device cpu`。可用模型与全部参数见 `python train.py --help`。

训练日志、指标和权重保存在 `--output` 目录。训练完成后验证第 90 轮权重：

```bash
python val_imagenet.py \
  --data-root /path/to/imagenet \
  --checkpoint runs/imagenet/gcse_resnet50/seed3407/checkpoint_090.pth
```

中断后继续训练时，在原训练命令中增加：

```text
--resume runs/imagenet/gcse_resnet50/seed3407/latest.pth
```

## 3. COCO 目标检测训练

准备 COCO 2017 数据：

```text
/path/to/coco/
├── images/
│   ├── train2017/
│   └── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

先完成上面的 ImageNet 分类训练，取得 GCSE 骨干网络权重。单卡运行：

```bash
python train_coco.py \
  --task detection \
  --backbone gcse_resnet50 \
  --no-pretrained-backbone \
  --gcse-backbone-weights runs/imagenet/gcse_resnet50/seed3407/checkpoint_090.pth \
  --coco-root /path/to/coco \
  --batch-size 2 \
  --output-dir runs/coco/gcse_resnet50_single
```

单卡脚本默认训练 36 轮，显存不足时可减小 `--batch-size`。CPU 试运行需增加 `--device cpu --no-amp`。

按论文配置运行 8 卡、24 轮训练：

```bash
torchrun --standalone --nproc_per_node=8 train_coco_ddp.py \
  --task detection \
  --backbone gcse_resnet50 \
  --backbone-weights runs/imagenet/gcse_resnet50/seed3407/checkpoint_090.pth \
  --coco-root /path/to/coco \
  --seed 3407 \
  --output-dir runs/coco/gcse_resnet50 \
  --verify-paper-protocol
```

这条命令要求 8 张 GPU，以及同模型、同 seed、通过协议检查的 ImageNet 第 90 轮权重。单卡和分布式训练请使用不同的输出目录，避免覆盖已有训练结果。

## 4. 检测验证与图片推理

使用单卡检测训练生成的 `latest.pth` 验证（分布式训练请替换为对应输出目录）：

```bash
python val_coco.py \
  --coco-root /path/to/coco \
  --checkpoint runs/coco/gcse_resnet50_single/latest.pth
```

对单张图片或图片目录进行推理：

```bash
python infer_coco.py \
  --checkpoint runs/coco/gcse_resnet50_single/latest.pth \
  --input /path/to/image.jpg \
  --output-dir runs/coco_infer \
  --score-threshold 0.5
```

将 `--input` 改为目录路径即可批量处理图片。可视化图片和预测记录保存到 `--output-dir`。数据集和训练权重需自行准备。

## 5. 其他运行入口

查看完整参数：

```bash
python train_coco.py --help
python train_coco_ddp.py --help
python val_imagenet.py --help
python val_coco.py --help
python infer_coco.py --help
```

运行旧版 CIFAR 脚本前，额外安装其依赖：

```bash
python -m pip install -e '.[legacy]'
python cifar.py --epochs 200 --batch_size 64
```
