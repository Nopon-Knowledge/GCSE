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

from gcse.gcse_resnet import (
    gcse_resnet18,
    gcse_resnet34,
    gcse_resnet50,
    gcse_resnet101,
    gcse_resnet152,
)


DEFAULT_CHECKPOINT = "/storage/home/402005/python_project/gcse/best.pth"
DEFAULT_DATA_ROOT = "/storage/home/402005/datasets/imagenet-1k/imagenet-object-localization-challenge(1)/ILSVRC/Data/CLS-LOC"

MODEL_FACTORY = {
    "gcse_resnet18": gcse_resnet18,
    "gcse_resnet34": gcse_resnet34,
    "gcse_resnet50": gcse_resnet50,
    "gcse_resnet101": gcse_resnet101,
    "gcse_resnet152": gcse_resnet152,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImageNet evaluation for GCSE models")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Checkpoint path")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT, help="ImageNet root path")
    parser.add_argument("--val-dir", type=str, default=None, help="Validation dir (defaults to <data-root>/val_by_wnid_csv)")
    parser.add_argument("--model", type=str, default="gcse_resnet50", choices=MODEL_FACTORY.keys(), help="Model architecture")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes")
    parser.add_argument("--batch-size", type=int, default=256, help="Validation batch size")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--img-size", type=int, default=224, help="Center crop size")
    parser.add_argument("--resize-size", type=int, default=256, help="Resize shorter side")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda/cpu")
    parser.add_argument("--print-freq", type=int, default=20, help="Log every N steps")
    parser.add_argument("--metrics-file", type=str, default="runs/imagenet_eval/metrics.json", help="Save metrics json")
    return parser.parse_args()


def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        if "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):
            state_dict = checkpoint_obj["state_dict"]
        elif "model" in checkpoint_obj and isinstance(checkpoint_obj["model"], dict):
            state_dict = checkpoint_obj["model"]
        elif "model_state" in checkpoint_obj and isinstance(checkpoint_obj["model_state"], dict):
            state_dict = checkpoint_obj["model_state"]
        else:
            state_dict = checkpoint_obj
    else:
        raise TypeError("Unsupported checkpoint format.")

    cleaned = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    return cleaned


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> Tuple[float, float]:
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(float(correct_k.mul_(100.0 / target.size(0)).item()))
    return res[0], res[1]


def build_val_loader(args: argparse.Namespace) -> DataLoader:
    data_root = Path(args.data_root).expanduser()
    val_dir = Path(args.val_dir).expanduser() if args.val_dir else data_root / "val_by_wnid_csv"
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose(
        [
            transforms.Resize(args.resize_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    return DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )


def infer_model_config_from_checkpoint(args: argparse.Namespace, checkpoint_obj) -> Tuple[str, int]:
    model_name = args.model
    num_classes = args.num_classes
    if isinstance(checkpoint_obj, dict):
        ckpt_args = checkpoint_obj.get("args", {})
        if isinstance(ckpt_args, dict):
            model_name = ckpt_args.get("model", model_name)
            num_classes = int(ckpt_args.get("num_classes", num_classes))
    return model_name, num_classes


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, print_freq: int = 20) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()

    loss_sum = 0.0
    acc1_sum = 0.0
    acc5_sum = 0.0
    num_batches = len(loader)
    seen_samples = 0
    start = time.time()

    with torch.no_grad():
        for step, (images, target) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            seen_samples += int(target.size(0))

            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            loss_sum += float(loss.item())
            acc1_sum += acc1
            acc5_sum += acc5

            if step % max(1, print_freq) == 0:
                print(
                    f"Val Step[{step}/{num_batches}] "
                    f"Loss {loss_sum / step:.4f} Acc@1 {acc1_sum / step:.2f} Acc@5 {acc5_sum / step:.2f}"
                )

    elapsed = time.time() - start
    return {
        "loss": loss_sum / max(1, num_batches),
        "acc1": acc1_sum / max(1, num_batches),
        "acc5": acc5_sum / max(1, num_batches),
        "num_samples": float(seen_samples),
        "eval_time_sec": elapsed,
        "samples_per_sec": seen_samples / max(1e-8, elapsed),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_obj = torch.load(str(checkpoint_path), map_location="cpu")
    model_name, num_classes = infer_model_config_from_checkpoint(args, checkpoint_obj)
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Unsupported model in checkpoint/args: {model_name}")

    model_fn = MODEL_FACTORY[model_name]
    model = model_fn(num_classes=num_classes, pretrained=False) if model_name == "gcse_resnet50" else model_fn(num_classes=num_classes)

    state_dict = extract_state_dict(checkpoint_obj)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded checkpoint: {checkpoint_path}\n"
        f"Model: {model_name}, num_classes: {num_classes}\n"
        f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
    )

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    model.to(device)

    val_loader = build_val_loader(args)
    metrics = evaluate(model, val_loader, device, print_freq=args.print_freq)

    print("Validation metrics:")
    for key in sorted(metrics.keys()):
        value = metrics[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    metrics_path = Path(args.metrics_file).expanduser()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
