# GCSE.pytorch

GCSE.pytorch is a PyTorch implementation of Group-Channel-Spatial Excitation (GCSE) attention and its integration into ResNet and Inception v3. The repo includes CIFAR-10 and ImageNet training scripts, plus dataset preparation helpers.

**Highlights**
- GCSEAttention: multi-stat squeeze (avg/max/std) + local channel interaction + spatial guidance + residual scaling
- GCSE-ResNet: ImageNet variants (18/34/50/101/152) and CIFAR variants (20/32/56)
- GCSE-Inception v3: GCSEAttention inserted into Mixed blocks and AuxLogits
- Training scripts: `cifar.py`, `imagenet.py`, `train.py`

**Repository Layout**
- `gcse/gcse_module.py`: GCSEAttention implementation
- `gcse/gcse_resnet.py`: GCSE-ResNet and CIFAR variants
- `gcse/gcse_inception.py`: GCSE-Inception v3
- `gcse/baseline.py`: CIFAR baseline ResNet (no attention)
- `senet/se_module.py`: SENet SELayer implementation (for comparison)
- `senet/se_resnet.py`: SENet-ResNet (ImageNet/CIFAR)
- `senet/se_inception.py`: SENet-Inception v3
- `cifar.py`: CIFAR-10 training (homura)
- `imagenet.py`: ImageNet training (homura)
- `train.py`: ImageNet training (torchvision)
- `val_by_csv.py` / `val_rearrange.py`: ImageNet val set reorganization helpers
- `coco_exp/`: COCO detection/segmentation data/model/engine modules
- `train_coco.py`: COCO training (detection + instance segmentation)
- `train_coco_ddp.py`: COCO multi-GPU training (DDP / torchrun)
- `val_coco.py`: COCO validation/evaluation
- `infer_coco.py`: COCO checkpoint inference + visualization

**Requirements**
- Python >= 3.8
- PyTorch >= 1.6.0
- torchvision >= 0.7
- homura (for `cifar.py` and `imagenet.py`)
- pycocotools (for COCO parsing/evaluation)

```bash
pip install torch torchvision
pip install git+https://github.com/moskomule/homura@v2020.07
pip install pycocotools
```

**GCSEAttention Summary**
- Multi-stat squeeze: channel avg/max/std
- Channel interaction: adaptive 1D conv (ECA-style)
- Spatial guidance: depthwise conv + group spatial pooling
- Residual scaling: `y = x * (1 + alpha * scale)` with learnable `alpha`

**Quickstart: CIFAR-10**
- Train GCSE-ResNet20:

```bash
python cifar.py
```

- Train plain ResNet baseline:

```bash
python cifar.py --baseline
```

**ImageNet Training (homura)**
`imagenet.py` uses homura.
- Set ImageNet root (choose one):
  - Environment variable: `IMAGENET_ROOT=/path/to/imagenet`
  - Or edit the script to your path
- Run:

```bash
python imagenet.py
```

**ImageNet Training (torchvision)**
`train.py` is a more general ImageNet script.
- Default directory structure:
  - Train: `<data-root>/train/<wnid>/*.JPEG`
  - Val: `<data-root>/val_by_wnid_csv/<wnid>/*.JPEG`
- You can override with `--val-dir`.

Example:

```bash
python train.py --data-root /path/to/imagenet --model gcse_resnet50 --epochs 90
```

**ImageNet Val Reorganization**
The official ImageNet val set is often flat; it needs ImageFolder structure.
- CSV-based:

```bash
python val_by_csv.py
```

- XML-based:

```bash
python val_rearrange.py
```

Paths are hard-coded inside the scripts; update them to your environment.

**COCO Detection / Instance Segmentation**
The repository now includes a full COCO experiment pipeline (dataloader + train + val + inference).

- Supported tasks:
  - `detection`: Faster R-CNN
  - `segmentation`: Mask R-CNN
- Supported backbones:
  - `gcse_resnet50` (this repo)
  - `resnet50` (torchvision)

COCO directory example:

```text
/storage/home/402005/datasets/coco/
  images/
    train2017/
    val2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
```

By default, `train_coco.py`, `train_coco_ddp.py`, and `val_coco.py` use
`/storage/home/402005/datasets/coco` as `--coco-root`.
If your dataset is elsewhere, override with `--coco-root /path/to/coco`
or pass explicit `--train-images/--train-annotations/--val-images/--val-annotations`.

Train on COCO detection:

```bash
python train_coco.py \
  --task detection \
  --backbone gcse_resnet50 \
  --output-dir runs/coco_det
```

Train on COCO instance segmentation:

```bash
python train_coco.py \
  --task segmentation \
  --backbone gcse_resnet50 \
  --output-dir runs/coco_segm
```

Use torchvision ResNet-50 pretrained backbone:

```bash
python train_coco.py ... --backbone resnet50 --pretrained-backbone
```

Multi-GPU training (example: 2x A800 on one node):

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_coco_ddp.py \
  --task detection \
  --backbone gcse_resnet50 \
  --output-dir runs/coco_det_ddp \
  --batch-size 2 \
  --scale-lr-by-world-size
```

Validate a checkpoint:

```bash
python val_coco.py
```

Default checkpoint path is `runs/coco/best.pth`.

Run inference (single image or directory):

```bash
python infer_coco.py
```

Default inference config:
- checkpoint: `runs/coco/best.pth`
- input: `/storage/home/402005/datasets/coco/images/val2017`
- output-dir: `runs/coco_infer`

Notes:
- `train_coco.py` stores `latest.pth`, periodic epoch checkpoints, and `best.pth`.
- `category_mapping.json` is saved in `--output-dir` and reused by `val_coco.py`/`infer_coco.py`.
- If you have a pretrained GCSE classification checkpoint, pass it with `--gcse-backbone-weights`.
- `train_coco.py` and `train_coco_ddp.py` write detailed logs by default:
  - text log: `<output-dir>/train.log`
  - structured metrics: `<output-dir>/metrics.jsonl`
  - customize via `--log-file` and `--metrics-file`

**GCSE-ResNet / Inception v3 API**
ImageNet:
- `gcse_resnet18(num_classes=1000)`
- `gcse_resnet34(num_classes=1000)`
- `gcse_resnet50(num_classes=1000, pretrained=False)`
- `gcse_resnet101(num_classes=1000)`
- `gcse_resnet152(num_classes=1000)`

CIFAR:
- `gcse_resnet20(num_classes=10)`
- `gcse_resnet32(num_classes=10)`
- `gcse_resnet56(num_classes=10)`
- `gcse_preactresnet20(num_classes=10)`
- `gcse_preactresnet32(num_classes=10)`
- `gcse_preactresnet56(num_classes=10)`

Inception v3:
- `gcse_inception_v3(num_classes=1000, aux_logits=True)`
- Input must be `299x299`

**SENet Baseline (Comparison)**
SENet implementations are preserved under `senet/` and not used by default. Import them directly for comparison:

```python
from senet.se_resnet import se_resnet50
model = se_resnet50(num_classes=1000)
```

**Torch Hub**
`hubconf.py` exposes Torch Hub entry points (functions named `gcse_resnet*`).

```python
import torch.hub
model = torch.hub.load('yourname/gcse.pytorch', 'gcse_resnet20', num_classes=10)
```

**Pretrained**
This repo does not provide GCSE pretrained weights.
`gcse_resnet50(pretrained=True)` raises `NotImplementedError`.

**Notes**
- If you migrate existing experiments, note that module names were changed from `se_*` to `gcse_*`, and the package name is `gcse`.
- `imagenet_dataloader.py` provides a minimal ImageNet DataLoader example.

**License / Reference**
- This repository implements GCSE attention and integrates it into ResNet/Inception examples.
