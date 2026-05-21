from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from coco_exp.coco_dataloader import (
    CocoDetectionDataset,
    category_mapping_payload,
    collate_fn,
    get_coco_transforms,
)
from coco_exp.coco_engine import evaluate_coco, train_one_epoch
from coco_exp.coco_models import build_coco_model, parse_mapping_payload, trainable_parameters
from coco_exp.coco_paths import DEFAULT_COCO_ROOT, ensure_coco_paths_exist, resolve_coco_paths



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COCO detection/segmentation distributed training")

    parser.add_argument("--task", choices=["detection", "segmentation"], default="detection")
    parser.add_argument("--backbone", choices=["gcse_resnet50", "resnet50"], default="gcse_resnet50")
    parser.add_argument("--pretrained-backbone", action="store_true", help="Use ImageNet pretrained torchvision resnet50 backbone")
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

    parser.add_argument("--output-dir", type=str, default="runs/coco_ddp")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU/process")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--horizontal-flip-prob", type=float, default=0.5)

    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--scale-lr-by-world-size", action="store_true", help="Scale lr by number of GPUs")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["multistep", "cosine"], default="multistep")
    parser.add_argument("--lr-milestones", type=str, default="16,22")
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)

    parser.add_argument("--trainable-backbone-layers", type=int, default=5)
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)

    parser.add_argument("--amp", action="store_true")
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
    parser.add_argument("--find-unused-parameters", action="store_true", help="Enable DDP find_unused_parameters")

    parser.add_argument("--dist-url", type=str, default="env://")
    parser.add_argument("--local_rank", type=int, default=-1, help="For torch.distributed.launch compatibility")

    return parser.parse_args()



def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()



def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()



def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()



def is_main_process() -> bool:
    return get_rank() == 0



def setup_for_distributed(is_master: bool) -> None:
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print_fn(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print_fn



def init_distributed_mode(args: argparse.Namespace) -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    else:
        args.rank = 0
        args.world_size = 1
        args.distributed = False
        setup_for_distributed(is_master=True)
        return

    args.distributed = True

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Distributed training requested on CUDA, but CUDA is not available.")

    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    dist.init_process_group(
        backend=backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    setup_for_distributed(is_master=args.rank == 0)



def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()



def set_seed(seed: int, rank: int = 0) -> None:
    mixed_seed = seed + rank
    random.seed(mixed_seed)
    np.random.seed(mixed_seed)
    torch.manual_seed(mixed_seed)
    torch.cuda.manual_seed_all(mixed_seed)


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



def reduce_stats(input_dict: Dict[str, float], device: torch.device, average: bool = True) -> Dict[str, float]:
    if not is_dist_avail_and_initialized():
        return input_dict

    keys = sorted(input_dict.keys())
    values = torch.tensor([float(input_dict[key]) for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values)
    if average:
        values /= get_world_size()

    reduced = {key: float(values[idx].item()) for idx, key in enumerate(keys)}
    return reduced



def save_checkpoint(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def setup_logger(output_dir: Path, log_filename: str, enabled: bool) -> logging.Logger:
    logger = logging.getLogger("coco_train_ddp")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    if not enabled:
        logger.addHandler(logging.NullHandler())
        return logger

    output_dir.mkdir(parents=True, exist_ok=True)
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



def maybe_load_checkpoint(args, model_without_ddp, optimizer, scheduler, scaler, logger: logging.Logger):
    start_epoch = 0
    best_metric = -1.0
    mapping_payload = None

    if not args.resume:
        return start_epoch, best_metric, mapping_payload

    checkpoint = torch.load(args.resume, map_location="cpu")
    model_without_ddp.load_state_dict(checkpoint["model"])

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



def build_train_loader(args):
    train_dataset = CocoDetectionDataset(
        images_dir=args.train_images,
        annotation_file=args.train_annotations,
        task=args.task,
        transforms=get_coco_transforms(is_train=True, horizontal_flip_prob=args.horizontal_flip_prob),
        remove_crowd=True,
        category_id_to_contiguous=None,
    )

    if args.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            drop_last=True,
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return train_dataset, train_loader, train_sampler



def build_val_loader_if_needed(args, category_id_to_contiguous: Dict[int, int]):
    if args.distributed and not is_main_process():
        return None, None

    val_dataset = CocoDetectionDataset(
        images_dir=args.val_images,
        annotation_file=args.val_annotations,
        task=args.task,
        transforms=get_coco_transforms(is_train=False, horizontal_flip_prob=0.0),
        remove_crowd=False,
        category_id_to_contiguous=category_id_to_contiguous,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return val_dataset, val_loader



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

    init_distributed_mode(args)

    rank = get_rank()
    world_size = get_world_size()
    set_seed(args.seed, rank=rank)

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.", force=True)
        requested_device = "cpu"

    if requested_device.startswith("cuda") and args.distributed:
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device(requested_device)

    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    if args.distributed:
        dist.barrier()
    logger = setup_logger(output_dir, args.log_file, enabled=is_main_process())
    metrics_path = output_dir / args.metrics_file

    if is_main_process():
        logger.info("Starting distributed COCO training with args: %s", vars(args))
        logger.info(
            "Environment: torch=%s cuda_available=%s world_size=%d rank=%d local_rank=%d",
            torch.__version__,
            torch.cuda.is_available(),
            world_size,
            rank,
            args.local_rank,
        )
        append_jsonl(
            metrics_path,
            {
                "event": "run_start",
                "timestamp": int(time.time()),
                "args": vars(args),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "world_size": world_size,
                "rank": rank,
                "local_rank": args.local_rank,
            },
        )

    train_dataset, train_loader, train_sampler = build_train_loader(args)

    mapping_payload = category_mapping_payload(train_dataset)
    category_id_to_contiguous, contiguous_to_category_id, _ = parse_mapping_payload(mapping_payload)
    if not category_id_to_contiguous:
        category_id_to_contiguous = train_dataset.category_id_to_contiguous
    if not contiguous_to_category_id:
        contiguous_to_category_id = train_dataset.contiguous_to_category_id

    num_classes = len(train_dataset.category_id_to_contiguous) + 1
    resnet50_backbone_weights = resolve_resnet50_backbone_weights(args)
    if is_main_process() and args.backbone == "resnet50" and args.pretrained_backbone:
        if resnet50_backbone_weights:
            logger.info("Using local resnet50 backbone weights: %s", resnet50_backbone_weights)
        else:
            logger.info("No local resnet50 backbone weights found; will try torchvision pretrained download.")

    model = build_coco_model(
        task=args.task,
        num_classes=num_classes,
        backbone_name=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
        gcse_backbone_weights=args.gcse_backbone_weights,
        resnet50_backbone_weights=resnet50_backbone_weights,
        trainable_backbone_layers=args.trainable_backbone_layers,
        min_size=args.min_size,
        max_size=args.max_size,
    )
    model.to(device)

    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank] if device.type == "cuda" else None,
            output_device=args.local_rank if device.type == "cuda" else None,
            find_unused_parameters=args.find_unused_parameters,
        )

    model_without_ddp = model.module if args.distributed else model

    lr = args.lr * world_size if args.scale_lr_by_world_size else args.lr
    if is_main_process():
        global_batch = args.batch_size * world_size
        logger.info("World size: %d, per-GPU batch: %d, global batch: %d", world_size, args.batch_size, global_batch)
        logger.info("Learning rate: %.6f (base %.6f, scaled=%s)", lr, args.lr, args.scale_lr_by_world_size)

    optimizer = torch.optim.SGD(
        trainable_parameters(model),
        lr=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch, best_metric, ckpt_mapping_payload = maybe_load_checkpoint(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        logger=logger,
    )

    if ckpt_mapping_payload is not None:
        loaded_cat_to_contiguous, loaded_contiguous_to_cat, _ = parse_mapping_payload(ckpt_mapping_payload)
        if loaded_cat_to_contiguous:
            mapping_payload = ckpt_mapping_payload
            category_id_to_contiguous = loaded_cat_to_contiguous
        if loaded_contiguous_to_cat:
            contiguous_to_category_id = loaded_contiguous_to_cat

    _, val_loader = build_val_loader_if_needed(args, category_id_to_contiguous)

    if is_main_process():
        logger.info(
            "Data loaded. train_images=%d train_batches=%d batch_size_per_gpu=%d",
            len(train_dataset),
            len(train_loader),
            args.batch_size,
        )
        if val_loader is not None:
            logger.info("Validation loaded. val_batches=%d", len(val_loader))
        (output_dir / "category_mapping.json").write_text(
            json.dumps(mapping_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.eval_only:
        if is_main_process():
            if val_loader is None:
                raise RuntimeError("val_loader is not initialized on the main process.")
            metrics = evaluate_coco(
                model=model_without_ddp,
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

        if args.distributed:
            dist.barrier()
            cleanup_distributed()
        return

    primary_metric_key = "segm_AP" if args.task == "segmentation" else "bbox_AP"

    for epoch in range(start_epoch, args.epochs):
        epoch_index = epoch + 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if is_main_process():
            logger.info("Epoch %03d/%03d started. lr=%.6f", epoch_index, args.epochs, current_lr)

        train_stats = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            print_freq=args.print_freq if is_main_process() else 10**9,
            amp=args.amp,
            scaler=scaler,
            max_grad_norm=args.max_grad_norm,
            logger=logger if is_main_process() else None,
        )
        train_stats = reduce_stats(train_stats, device=device, average=True)

        scheduler.step()
        next_lr = optimizer.param_groups[0]["lr"]

        metrics: Dict[str, float] = {}
        should_eval = args.eval_every > 0 and ((epoch_index % args.eval_every == 0) or (epoch_index == args.epochs))

        if should_eval and is_main_process():
            if val_loader is None:
                raise RuntimeError("val_loader is not initialized on the main process.")
            metrics = evaluate_coco(
                model=model_without_ddp,
                data_loader=val_loader,
                device=device,
                contiguous_to_category_id=contiguous_to_category_id,
                task=args.task,
                print_freq=args.print_freq,
                mask_threshold=args.mask_threshold,
                logger=logger,
            )
        elif is_main_process():
            logger.info("Epoch %03d/%03d skipped validation because eval-every=%d.", epoch_index, args.epochs, args.eval_every)

        if args.distributed:
            dist.barrier()

        if is_main_process():
            primary_metric = metrics.get(primary_metric_key, -1.0)
            is_best = primary_metric > best_metric
            if primary_metric >= 0:
                best_metric = max(best_metric, primary_metric)

            checkpoint_payload = {
                "epoch": epoch,
                "model": model_without_ddp.state_dict(),
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

    if is_main_process():
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

    if args.distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
