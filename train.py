import argparse  # CLI argument parsing
import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast  # AMP utilities
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# GCSE-ResNet constructors
# SENet comparison (optional): from senet.se_resnet import se_resnet18, se_resnet34, se_resnet50, se_resnet101, se_resnet152
from gcse.gcse_resnet import (
    gcse_resnet18,
    gcse_resnet34,
    gcse_resnet50,
    gcse_resnet101,
    gcse_resnet152,
)


# Map string names to model constructors for CLI selection.
MODEL_FACTORY = {
    "gcse_resnet18": gcse_resnet18,
    "gcse_resnet34": gcse_resnet34,
    "gcse_resnet50": gcse_resnet50,
    "gcse_resnet101": gcse_resnet101,
    "gcse_resnet152": gcse_resnet152,
}


def parse_args() -> argparse.Namespace:
    # Define all tunable training hyperparameters and paths.
    parser = argparse.ArgumentParser(description="ImageNet training script for GCSE-ResNet")
    parser.add_argument("--data-root", type=str, default="/storage/home/402005/datasets/imagenet-1k/imagenet-object-localization-challenge(1)/ILSVRC/Data/CLS-LOC", help="Root that contains train/ and val/ folders")
    parser.add_argument("--train-dir", type=str, default=None, help="Override train directory (defaults to <data-root>/train)")
    parser.add_argument("--val-dir", type=str, default=None, help="Override val directory (defaults to <data-root>/val)")
    parser.add_argument("--output", type=str, default="./runs/imagenet", help="Directory to save checkpoints and logs")

    parser.add_argument("--model", type=str, default="gcse_resnet50", choices=MODEL_FACTORY.keys(), help="Model variant")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes")
    parser.add_argument("--pretrained", action="store_true", help="Pretrained weights for GCSE are not provided (flag reserved)")

    parser.add_argument("--epochs", type=int, default=90, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Global batch size")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--img-size", type=int, default=224, help="Training crop size")
    parser.add_argument("--resize-size", type=int, default=256, help="Resize shorter side for validation")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for cross entropy")
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Use mixup with given alpha (>0 enables)")

    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--nesterov", action="store_true", help="Use Nesterov momentum")
    parser.add_argument("--scheduler", type=str, default="multistep", choices=["multistep", "cosine"], help="LR scheduler type")
    parser.add_argument("--lr-milestones", type=str, default="30,60,80", help="Comma separated milestones for multistep scheduler")
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="Decay factor for multistep scheduler")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Linear warmup epochs")
    parser.add_argument("--warmup-factor", type=float, default=0.1, help="Starting LR factor for warmup")
    parser.add_argument("--clip-grad", type=float, default=0.0, help="Gradient clipping max norm (0 to disable)")

    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use, e.g. cuda or cpu")
    parser.add_argument("--print-freq", type=int, default=1, help="Iterations between logs")
    parser.add_argument("--save-every", type=int, default=1, help="Save epoch checkpoint every N epochs")
    parser.add_argument("--log-file", type=str, default="train.log", help="Log filename under output directory")
    parser.add_argument("--metrics-file", type=str, default="metrics.jsonl", help="Structured epoch metrics filename under output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true", help="Only run validation")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    # Fix random seed for reproducibility.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace) -> nn.Module:
    # Select and instantiate model based on CLI.
    model_fn = MODEL_FACTORY[args.model]
    model = model_fn(num_classes=args.num_classes, pretrained=args.pretrained) if args.model == "gcse_resnet50" else model_fn(num_classes=args.num_classes)
    return model


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    # Build train/val datasets and DataLoaders.
    data_root = Path(args.data_root).expanduser()
    train_dir = Path(args.train_dir).expanduser() if args.train_dir else data_root / "train"
    val_dir = Path(args.val_dir).expanduser() if args.val_dir else data_root / "val_by_wnid_csv"

    # Standard ImageNet normalization.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Training augmentation: random crop + horizontal flip.
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    # Validation is deterministic: resize then center crop.
    val_transforms = transforms.Compose(
        [
            transforms.Resize(args.resize_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Directory layout must follow ImageFolder: subfolders are class names.
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    # Shuffle train set and drop_last to keep batch sizes consistent.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> Iterable[torch.Tensor]:
    # Compute top-k accuracy (default top1 and top5).
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / target.size(0)))
        return res


def save_checkpoint(state: dict, output_dir: Path, epoch: int, is_best: bool, save_epoch: bool = True) -> None:
    # Save latest checkpoint every epoch, optionally save epoch snapshot, and keep best snapshots.
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_dir / "latest.pth")
    if save_epoch:
        checkpoint_path = output_dir / f"checkpoint_{epoch:03d}.pth"
        torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state, output_dir / "best.pth")
        torch.save(state, output_dir / f"best_epoch_{epoch:03d}.pth")


def setup_logger(output_dir: Path, log_filename: str) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("imagenet_train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / log_filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def append_jsonl(path: Path, payload: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def maybe_mixup(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]:
    # Apply mixup when alpha > 0; otherwise return inputs unchanged.
    if alpha <= 0.0:
        return x, (y, y, 1.0)
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b, lam)


def mixup_criterion(criterion, preds, targets):
    # Mixup cross-entropy (two labels weighted by lam).
    y_a, y_b, lam = targets
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)


def adjust_learning_rate(optimizer, base_lr, epoch, it, iters_per_epoch, args):
    # Adjust LR linearly only during warmup; afterward use scheduler.
    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        warmup_progress = (epoch + it / iters_per_epoch) / max(1.0, args.warmup_epochs)
        lr = base_lr * (args.warmup_factor + (1 - args.warmup_factor) * warmup_progress)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def train_one_epoch(model: nn.Module, criterion, optimizer, scaler: GradScaler, loader: DataLoader, device: torch.device, epoch: int, args: argparse.Namespace, logger: logging.Logger) -> Dict[str, float]:
    # Train for one epoch: AMP, mixup, grad clipping, and logging.
    model.train()
    loss_sum = 0.0
    acc1_sum = 0.0
    acc5_sum = 0.0
    num_batches = len(loader)
    start = time.time()
    iter_time_sum = 0.0
    seen_samples = 0
    for i, (images, target) in enumerate(loader):
        iter_start = time.time()
        adjust_learning_rate(optimizer, args.lr, epoch, i, num_batches, args)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        seen_samples += int(target.size(0))

        if args.mixup_alpha > 0.0:
            images, mix_target = maybe_mixup(images, target, args.mixup_alpha)
        else:
            mix_target = None

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            output = model(images)
            loss = mixup_criterion(criterion, output, mix_target) if mix_target else criterion(output, target)

        scaler.scale(loss).backward()
        if args.clip_grad > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        loss_sum += loss.item()
        acc1_sum += prec1.item()
        acc5_sum += prec5.item()
        iter_time = time.time() - iter_start
        iter_time_sum += iter_time

        if (i + 1) % args.print_freq == 0:
            avg_loss = loss_sum / (i + 1)
            avg_acc1 = acc1_sum / (i + 1)
            avg_acc5 = acc5_sum / (i + 1)
            lr = optimizer.param_groups[0]["lr"]
            max_mem_mb = 0.0
            if device.type == "cuda":
                max_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            logger.info(
                "Epoch[%03d] Iter[%04d/%04d] LR %.6f Loss %.4f Acc@1 %.2f Acc@5 %.2f "
                "BatchTime %.3fs MaxMem %.0fMB",
                epoch + 1,
                i + 1,
                num_batches,
                lr,
                avg_loss,
                avg_acc1,
                avg_acc5,
                iter_time,
                max_mem_mb,
            )

    epoch_time = time.time() - start
    avg_iter_time = iter_time_sum / max(1, num_batches)
    samples_per_sec = seen_samples / max(1e-8, epoch_time)
    return {
        "loss": loss_sum / max(1, num_batches),
        "acc1": acc1_sum / max(1, num_batches),
        "acc5": acc5_sum / max(1, num_batches),
        "num_samples": float(seen_samples),
        "epoch_time_sec": epoch_time,
        "avg_iter_time_sec": avg_iter_time,
        "samples_per_sec": samples_per_sec,
    }


def validate(model: nn.Module, criterion, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> Dict[str, float]:
    # Evaluate on validation set; return avg loss and top1/top5.
    model.eval()
    loss_sum = 0.0
    acc1_sum = 0.0
    acc5_sum = 0.0
    num_batches = len(loader)
    start = time.time()
    seen_samples = 0
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            seen_samples += int(target.size(0))
            output = model(images)
            loss = criterion(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            loss_sum += loss.item()
            acc1_sum += prec1.item()
            acc5_sum += prec5.item()
    epoch_time = time.time() - start
    samples_per_sec = seen_samples / max(1e-8, epoch_time)
    return {
        "loss": loss_sum / max(1, num_batches),
        "acc1": acc1_sum / max(1, num_batches),
        "acc5": acc5_sum / max(1, num_batches),
        "num_samples": float(seen_samples),
        "epoch_time_sec": epoch_time,
        "samples_per_sec": samples_per_sec,
    }


def main():
    # Entry point: parse args, build model/data, train/validate, save checkpoints.
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output)
    logger = setup_logger(output_dir, args.log_file)
    metrics_path = output_dir / args.metrics_file
    if metrics_path.exists():
        metrics_path.unlink()

    logger.info("Starting training with args: %s", vars(args))

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    torch.backends.cudnn.benchmark = True
    model = build_model(args)
    model.to(device)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    if args.scheduler == "multistep":
        milestones = [int(m) for m in args.lr_milestones.split(",") if m]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = GradScaler(enabled=args.amp)

    start_epoch = 0
    best_acc1 = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0)
        best_acc1 = checkpoint.get("best_acc1", 0.0)
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    train_loader, val_loader = build_dataloaders(args)
    logger.info(
        "Data loaded. train_batches=%d val_batches=%d batch_size=%d",
        len(train_loader),
        len(val_loader),
        args.batch_size,
    )

    if args.eval_only:
        val_stats = validate(model, criterion, val_loader, device, args)
        logger.info(
            "Eval only -> Loss %.4f Acc@1 %.2f Acc@5 %.2f Time %.2fs Throughput %.2f img/s",
            val_stats["loss"],
            val_stats["acc1"],
            val_stats["acc5"],
            val_stats["epoch_time_sec"],
            val_stats["samples_per_sec"],
        )
        append_jsonl(
            metrics_path,
            {
                "mode": "eval_only",
                "val": val_stats,
                "timestamp": int(time.time()),
            },
        )
        return

    for epoch in range(start_epoch, args.epochs):
        epoch_index = epoch + 1
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %03d/%03d started. LR=%.6f", epoch_index, args.epochs, current_lr)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_stats = train_one_epoch(model, criterion, optimizer, scaler, train_loader, device, epoch, args, logger)
        val_stats = validate(model, criterion, val_loader, device, args)

        scheduler.step()
        next_lr = optimizer.param_groups[0]["lr"]

        is_best = val_stats["acc1"] > best_acc1
        best_acc1 = max(best_acc1, val_stats["acc1"])
        state = {
            "epoch": epoch_index,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc1": best_acc1,
            "train_stats": train_stats,
            "val_stats": val_stats,
            "args": vars(args),
        }
        save_epoch_snapshot = (epoch_index % args.save_every == 0) or (epoch_index == args.epochs)
        save_checkpoint(state, output_dir, epoch_index, is_best, save_epoch=save_epoch_snapshot)

        peak_mem_mb = 0.0
        if device.type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        logger.info(
            "Epoch %03d/%03d done | "
            "Train: loss %.4f acc1 %.2f acc5 %.2f time %.2fs speed %.2f img/s | "
            "Val: loss %.4f acc1 %.2f acc5 %.2f time %.2fs speed %.2f img/s | "
            "LR %.6f -> %.6f | Best Acc@1 %.2f | PeakMem %.0fMB%s",
            epoch_index,
            args.epochs,
            train_stats["loss"],
            train_stats["acc1"],
            train_stats["acc5"],
            train_stats["epoch_time_sec"],
            train_stats["samples_per_sec"],
            val_stats["loss"],
            val_stats["acc1"],
            val_stats["acc5"],
            val_stats["epoch_time_sec"],
            val_stats["samples_per_sec"],
            current_lr,
            next_lr,
            best_acc1,
            peak_mem_mb,
            " | NEW_BEST" if is_best else "",
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch_index,
                "train": train_stats,
                "val": val_stats,
                "lr": current_lr,
                "next_lr": next_lr,
                "best_acc1": best_acc1,
                "is_best": is_best,
                "peak_memory_mb": peak_mem_mb,
                "timestamp": int(time.time()),
            },
        )

    logger.info("Training completed. Logs: %s, metrics: %s", output_dir / args.log_file, metrics_path)


if __name__ == "__main__":
    main()
