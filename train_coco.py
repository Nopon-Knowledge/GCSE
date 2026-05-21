from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from coco_exp.coco_dataloader import build_coco_dataloader, category_mapping_payload
from coco_exp.coco_engine import evaluate_coco, train_one_epoch
from coco_exp.coco_models import build_coco_model, parse_mapping_payload, trainable_parameters
from coco_exp.coco_paths import DEFAULT_COCO_ROOT, ensure_coco_paths_exist, resolve_coco_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COCO detection/segmentation training")

    parser.add_argument("--task", choices=["detection", "segmentation"], default="detection")
    parser.add_argument("--backbone", choices=["gcse_resnet50", "resnet50"], default="resnet50")
    parser.add_argument(
        "--pretrained-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet pretrained torchvision resnet50 backbone",
    )
    parser.add_argument(
        "--resnet50-backbone-weights",
        type=str,
        default=None,
        help="Local path to resnet50 ImageNet weights (for offline/no-network environments).",
    )
    parser.add_argument("--gcse-backbone-weights", type=str, default=None, help="Optional GCSE classification weights for backbone init")

    parser.add_argument(
        "--coco-root",
        type=str,
        default=str(DEFAULT_COCO_ROOT),
        help="COCO root directory. Defaults to /storage/home/402005/datasets/coco",
    )
    parser.add_argument("--train-images", type=str, default=None, help="COCO train image directory, e.g. /path/coco/images/train2017")
    parser.add_argument("--train-annotations", type=str, default=None, help="COCO train annotation json, e.g. .../instances_train2017.json")
    parser.add_argument("--val-images", type=str, default=None, help="COCO val image directory, e.g. /path/coco/images/val2017")
    parser.add_argument("--val-annotations", type=str, default=None, help="COCO val annotation json, e.g. .../instances_val2017.json")

    parser.add_argument("--output-dir", type=str, default="runs/coco_res50_pretrain_v1")
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--horizontal-flip-prob", type=float, default=0.5)
    parser.add_argument("--color-jitter-strength", type=float, default=0.1)
    parser.add_argument("--grayscale-prob", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=0.0075)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["multistep", "cosine"], default="cosine")
    parser.add_argument("--lr-milestones", type=str, default="16,22")
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--accumulate-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--warmup-factor", type=float, default=0.001)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.9998)

    parser.add_argument("--trainable-backbone-layers", type=int, default=5)
    parser.add_argument("--train-min-sizes", type=str, default="640,672,704,736,768,800")
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)

    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--log-file", type=str, default="train.log", help="Log filename under output directory")
    parser.add_argument("--metrics-file", type=str, default="metrics.jsonl", help="Structured metrics filename under output directory")

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_train_min_sizes(train_min_sizes: str, fallback_min_size: int) -> tuple:
    parsed = []
    for token in str(train_min_sizes).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value > 0:
            parsed.append(value)
    if not parsed:
        parsed = [int(fallback_min_size)]

    seen = set()
    ordered = []
    for value in parsed:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return tuple(ordered)


def resolve_resnet50_backbone_weights(args: argparse.Namespace) -> str | None:
    if args.backbone != "resnet50" or not args.pretrained_backbone:
        return None

    if args.resnet50_backbone_weights:
        candidate = Path(args.resnet50_backbone_weights).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"resnet50 backbone weights not found: {candidate}")
        return str(candidate)

    candidates = [
        Path("resnet50-19c8e357.pth"),
        Path(__file__).resolve().parent / "resnet50-19c8e357.pth",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def build_scheduler(optimizer, args):
    if args.scheduler == "multistep":
        milestones = [int(item) for item in args.lr_milestones.split(",") if item.strip()]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))


def save_checkpoint(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def setup_logger(output_dir: Path, log_filename: str) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("coco_train")
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


def maybe_load_checkpoint(args, model, optimizer, scheduler, scaler, ema_model, logger: logging.Logger):
    start_epoch = 0
    best_metric = -1.0
    mapping_payload = None

    if not args.resume:
        return start_epoch, best_metric, mapping_payload

    checkpoint = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if ema_model is not None and checkpoint.get("ema_model", None) is not None:
        ema_model.load_state_dict(checkpoint["ema_model"])

    if not args.eval_only:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint and checkpoint["scaler"] is not None:
            scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_metric = float(checkpoint.get("best_metric", -1.0))
    mapping_payload = checkpoint.get("category_mapping", None)

    logger.info("Resumed from %s at epoch %d.", args.resume, start_epoch)
    return start_epoch, best_metric, mapping_payload


def main() -> None:
    args = parse_args()
    coco_paths = resolve_coco_paths(
        coco_root=args.coco_root,
        train_images=args.train_images,
        train_annotations=args.train_annotations,
        val_images=args.val_images,
        val_annotations=args.val_annotations,
    )
    ensure_coco_paths_exist(coco_paths, require_train=True, require_val=True)
    args.train_images = str(coco_paths["train_images"])
    args.train_annotations = str(coco_paths["train_annotations"])
    args.val_images = str(coco_paths["val_images"])
    args.val_annotations = str(coco_paths["val_annotations"])

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    logger = setup_logger(output_dir, args.log_file)
    metrics_path = output_dir / args.metrics_file

    logger.info("Starting COCO training with args: %s", vars(args))
    logger.info("Environment: torch=%s cuda_available=%s num_gpus=%d", torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
    append_jsonl(
        metrics_path,
        {
            "event": "run_start",
            "timestamp": int(time.time()),
            "args": vars(args),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "num_gpus": int(torch.cuda.device_count()),
        },
    )

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)
    logger.info("Using device: %s", device)

    resnet50_backbone_weights = resolve_resnet50_backbone_weights(args)
    if args.backbone == "resnet50" and args.pretrained_backbone:
        if resnet50_backbone_weights:
            logger.info("Using local resnet50 backbone weights: %s", resnet50_backbone_weights)
        else:
            logger.info("No local resnet50 backbone weights found; will try torchvision pretrained download.")

    train_dataset, train_loader = build_coco_dataloader(
        images_dir=args.train_images,
        annotation_file=args.train_annotations,
        task=args.task,
        batch_size=args.batch_size,
        workers=args.workers,
        is_train=True,
        horizontal_flip_prob=args.horizontal_flip_prob,
        color_jitter_strength=args.color_jitter_strength,
        grayscale_prob=args.grayscale_prob,
    )

    val_dataset, val_loader = build_coco_dataloader(
        images_dir=args.val_images,
        annotation_file=args.val_annotations,
        task=args.task,
        batch_size=args.batch_size,
        workers=args.workers,
        is_train=False,
        category_id_to_contiguous=train_dataset.category_id_to_contiguous,
        remove_crowd=False,
        horizontal_flip_prob=0.0,
        color_jitter_strength=0.0,
        grayscale_prob=0.0,
    )
    logger.info(
        "Data loaded. train_images=%d val_images=%d train_batches=%d val_batches=%d batch_size=%d",
        len(train_dataset),
        len(val_dataset),
        len(train_loader),
        len(val_loader),
        args.batch_size,
    )

    train_min_sizes = parse_train_min_sizes(args.train_min_sizes, args.min_size)
    model_min_size = train_min_sizes if len(train_min_sizes) > 1 else train_min_sizes[0]
    logger.info(
        "Model resize config: train_min_sizes=%s eval_min_size=%d max_size=%d",
        train_min_sizes,
        args.min_size,
        args.max_size,
    )

    num_classes = len(train_dataset.category_id_to_contiguous) + 1
    model = build_coco_model(
        task=args.task,
        num_classes=num_classes,
        backbone_name=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
        gcse_backbone_weights=args.gcse_backbone_weights,
        resnet50_backbone_weights=resnet50_backbone_weights,
        trainable_backbone_layers=args.trainable_backbone_layers,
        min_size=model_min_size,
        max_size=args.max_size,
    )
    model.to(device)

    ema_model = None
    if args.use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.SGD(
        trainable_parameters(model),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)

    start_epoch, best_metric, ckpt_mapping_payload = maybe_load_checkpoint(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ema_model=ema_model,
        logger=logger,
    )

    mapping_payload = category_mapping_payload(train_dataset)
    if ckpt_mapping_payload is not None:
        loaded_cat_to_contiguous, _, _ = parse_mapping_payload(ckpt_mapping_payload)
        if loaded_cat_to_contiguous:
            mapping_payload = ckpt_mapping_payload

    (output_dir / "category_mapping.json").write_text(
        json.dumps(mapping_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _, contiguous_to_category_id, _ = parse_mapping_payload(mapping_payload)
    if not contiguous_to_category_id:
        contiguous_to_category_id = train_dataset.contiguous_to_category_id

    if args.eval_only:
        eval_model = ema_model if ema_model is not None else model
        metrics = evaluate_coco(
            model=eval_model,
            data_loader=val_loader,
            device=device,
            contiguous_to_category_id=contiguous_to_category_id,
            task=args.task,
            print_freq=args.print_freq,
            mask_threshold=args.mask_threshold,
            logger=logger,
        )
        logger.info("Evaluation metrics:")
        for key in sorted(metrics.keys()):
            logger.info("  %s: %.4f", key, metrics[key])
        append_jsonl(
            metrics_path,
            {
                "event": "eval_only",
                "timestamp": int(time.time()),
                "metrics": metrics,
            },
        )
        return

    primary_metric_key = "segm_AP" if args.task == "segmentation" else "bbox_AP"

    for epoch in range(start_epoch, args.epochs):
        epoch_index = epoch + 1
        current_lr = optimizer.param_groups[0]["lr"]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        logger.info("Epoch %03d/%03d started. lr=%.6f", epoch_index, args.epochs, current_lr)
        train_stats = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            print_freq=args.print_freq,
            amp=args.amp,
            scaler=scaler,
            max_grad_norm=args.max_grad_norm,
            accumulate_steps=args.accumulate_steps,
            warmup_steps=args.warmup_steps,
            warmup_factor=args.warmup_factor,
            ema_model=ema_model,
            ema_decay=args.ema_decay,
            logger=logger,
        )

        scheduler.step()
        next_lr = optimizer.param_groups[0]["lr"]

        metrics = {}
        should_eval = args.eval_every > 0 and ((epoch_index % args.eval_every == 0) or (epoch_index == args.epochs))
        if should_eval:
            eval_model = ema_model if ema_model is not None else model
            metrics = evaluate_coco(
                model=eval_model,
                data_loader=val_loader,
                device=device,
                contiguous_to_category_id=contiguous_to_category_id,
                task=args.task,
                print_freq=args.print_freq,
                mask_threshold=args.mask_threshold,
                logger=logger,
            )
        else:
            logger.info("Epoch %03d/%03d skipped validation because eval-every=%d.", epoch_index, args.epochs, args.eval_every)

        primary_metric = metrics.get(primary_metric_key, -1.0)
        is_best = primary_metric > best_metric
        if primary_metric >= 0:
            best_metric = max(best_metric, primary_metric)

        checkpoint_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict() if ema_model is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if args.amp else None,
            "best_metric": best_metric,
            "task": args.task,
            "backbone": args.backbone,
            "num_classes": num_classes,
            "category_mapping": mapping_payload,
            "train_stats": train_stats,
            "val_metrics": metrics,
            "args": vars(args),
        }

        save_checkpoint(output_dir / "latest.pth", checkpoint_payload)
        if epoch_index % args.save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch_index:03d}.pth", checkpoint_payload)
        if is_best:
            save_checkpoint(output_dir / "best.pth", checkpoint_payload)

        logger.info(
            "Epoch %03d/%03d done | TrainLoss %.4f | TrainTime %.2fs | TrainSpeed %.2f img/s | "
            "LR %.6f -> %.6f | Best %s %.4f%s",
            epoch_index,
            args.epochs,
            train_stats["loss_total"],
            train_stats.get("epoch_time_sec", -1.0),
            train_stats.get("samples_per_sec", -1.0),
            current_lr,
            next_lr,
            primary_metric_key,
            best_metric,
            " | NEW_BEST" if is_best else "",
        )
        if metrics:
            for key in sorted(metrics.keys()):
                logger.info("  %s: %.4f", key, metrics[key])

        append_jsonl(
            metrics_path,
            {
                "event": "epoch_end",
                "timestamp": int(time.time()),
                "epoch": epoch_index,
                "lr": current_lr,
                "next_lr": next_lr,
                "train_stats": train_stats,
                "val_metrics": metrics,
                "best_metric": best_metric,
                "is_best": is_best,
                "checkpoint_latest": str(output_dir / "latest.pth"),
                "checkpoint_best": str(output_dir / "best.pth") if is_best else None,
            },
        )

    logger.info("Training completed. Detailed logs: %s / %s", output_dir / args.log_file, metrics_path)
    append_jsonl(
        metrics_path,
        {
            "event": "run_end",
            "timestamp": int(time.time()),
            "best_metric": best_metric,
            "primary_metric_key": primary_metric_key,
        },
    )


if __name__ == "__main__":
    main()
