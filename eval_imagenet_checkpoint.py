from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from gcse.gcse_resnet import gcse_resnet50
from senet.se_resnet import se_resnet50

DEFAULT_CHECKPOINT = "/storage/home/402005/python_project/gcse/best.pth"
DEFAULT_DATA_ROOT = "/storage/home/402005/datasets/imagenet-1k/imagenet-object-localization-challenge(1)/ILSVRC/Data/CLS-LOC"
DEFAULT_VAL_DIR = "/storage/home/402005/datasets/imagenet-1k/imagenet-object-localization-challenge(1)/ILSVRC/Data/CLS-LOC/val_by_wnid_csv"
DEFAULT_METRICS_FILE = "/storage/home/402005/python_project/gcse/runs/imagenet_eval/metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ImageNet checkpoint (Top-1/Top-5)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--val-dir", type=str, default=DEFAULT_VAL_DIR)
    parser.add_argument("--model", type=str, default=None, choices=["gcse_resnet50", "se_resnet50"])
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--metrics-file", type=str, default=DEFAULT_METRICS_FILE)
    return parser.parse_args()


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    return cleaned


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> Tuple[float, float]:
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        values = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            values.append(float(correct_k.mul_(100.0 / target.size(0)).item()))
    return values[0], values[1]


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "se_resnet50":
        return se_resnet50(num_classes=num_classes, pretrained=False)
    if model_name == "gcse_resnet50":
        return gcse_resnet50(num_classes=num_classes, pretrained=False)
    raise ValueError(f"Unsupported model: {model_name}")


def infer_model_and_classes(args: argparse.Namespace, checkpoint_obj) -> Tuple[str, int]:
    model_name = args.model
    num_classes = args.num_classes

    if isinstance(checkpoint_obj, dict):
        ckpt_args = checkpoint_obj.get("args", {})
        if isinstance(ckpt_args, dict):
            if model_name is None:
                model_name = ckpt_args.get("model", None)
            if num_classes is None and ckpt_args.get("num_classes", None) is not None:
                num_classes = int(ckpt_args["num_classes"])

    if model_name is None:
        model_name = "se_resnet50"
    if num_classes is None:
        num_classes = 1000
    return model_name, num_classes


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_obj = torch.load(str(checkpoint_path), map_location="cpu")
    model_name, num_classes = infer_model_and_classes(args, checkpoint_obj)

    model = build_model(model_name, num_classes)
    state_dict = checkpoint_obj.get("state_dict", checkpoint_obj)
    state_dict = clean_state_dict(state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    model.to(device)
    model.eval()

    data_root = Path(args.data_root).expanduser()
    val_dir = Path(args.val_dir).expanduser() if args.val_dir else data_root / "val_by_wnid_csv"
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    val_transform = transforms.Compose(
        [
            transforms.Resize(args.resize_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: {model_name} | num_classes: {num_classes}")
    print(f"Val dir: {val_dir} | images: {len(val_dataset)}")
    print(f"Missing keys: {len(missing_keys)} | Unexpected keys: {len(unexpected_keys)}")
    if missing_keys:
        print("Missing sample:", missing_keys[:8])
    if unexpected_keys:
        print("Unexpected sample:", unexpected_keys[:8])

    criterion = nn.CrossEntropyLoss().to(device)
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_steps = 0
    eval_start = time.time()

    with torch.no_grad():
        for step, (images, target) in enumerate(val_loader, start=1):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, target)
            acc1, acc5 = topk_accuracy(logits, target, topk=(1, 5))

            total_loss += float(loss.item())
            total_acc1 += acc1
            total_acc5 += acc5
            total_steps += 1

            if step % max(1, args.print_freq) == 0:
                print(
                    f"Val Step[{step}/{len(val_loader)}] "
                    f"Loss {total_loss / total_steps:.4f} "
                    f"Acc@1 {total_acc1 / total_steps:.2f} "
                    f"Acc@5 {total_acc5 / total_steps:.2f}"
                )

    elapsed = time.time() - eval_start
    metrics = {
        "loss": total_loss / max(1, total_steps),
        "acc1": total_acc1 / max(1, total_steps),
        "acc5": total_acc5 / max(1, total_steps),
        "num_samples": float(len(val_dataset)),
        "eval_time_sec": elapsed,
        "samples_per_sec": len(val_dataset) / max(1e-8, elapsed),
    }

    print("Validation metrics:")
    for key in sorted(metrics.keys()):
        print(f"  {key}: {metrics[key]:.4f}")

    metrics_path = Path(args.metrics_file).expanduser()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(checkpoint_path),
        "model": model_name,
        "num_classes": num_classes,
        "val_dir": str(val_dir),
        "missing_keys": len(missing_keys),
        "unexpected_keys": len(unexpected_keys),
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
