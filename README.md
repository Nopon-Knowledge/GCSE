# GCSE.pytorch

PyTorch implementation of Group-Channel-Spatial Excitation (GCSE) attention, with ResNet / Inception examples and training scripts for CIFAR-10, ImageNet, and COCO.

## Features

- GCSE attention module: multi-stat channel squeeze, local channel interaction, spatial guidance, and residual scaling
- GCSE-ResNet and GCSE-Inception v3 implementations
- SENet baseline modules for comparison
- Training / validation scripts for classification and COCO detection or segmentation

## Requirements

- Python >= 3.8
- PyTorch
- torchvision
- homura, for `cifar.py` and `imagenet.py`
- pycocotools, for COCO experiments

```bash
pip install torch torchvision
pip install git+https://github.com/moskomule/homura@v2020.07
pip install pycocotools
```

## Usage

CIFAR-10:

```bash
python cifar.py
python cifar.py --baseline
```

ImageNet:

```bash
python train.py --data-root /path/to/imagenet --model gcse_resnet50 --epochs 90
```

COCO detection / segmentation:

```bash
python train_coco.py --task detection --backbone gcse_resnet50 --output-dir runs/coco_det
python train_coco.py --task segmentation --backbone gcse_resnet50 --output-dir runs/coco_segm
python val_coco.py --checkpoint runs/coco_det/best.pth
python infer_coco.py --checkpoint runs/coco_det/best.pth --input /path/to/image_or_dir
```

## Project Structure

```text
gcse/       GCSE attention, ResNet, Inception, and baseline modules
senet/      SENet comparison modules
coco_exp/   COCO dataloader, model builder, and evaluation utilities
*.py        Training, validation, inference, and dataset helper scripts
```

## Notes

- This repository does not include pretrained GCSE weights.
- Training outputs, datasets, checkpoints, and local IDE files are ignored by `.gitignore`.
- Some helper scripts contain local default paths; pass command-line arguments or update paths for your environment.
