from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    from pycocotools import mask as coco_mask
    from pycocotools.cocoeval import COCOeval
except ImportError:  # pragma: no cover - runtime dependency
    coco_mask = None
    COCOeval = None


def _move_targets_to_device(targets: List[Dict[str, torch.Tensor]], device: torch.device) -> List[Dict]:
    moved_targets = []
    for target in targets:
        moved = {}
        for key, value in target.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        moved_targets.append(moved)
    return moved_targets


@torch.no_grad()
def _update_ema(model, ema_model, decay: float) -> None:
    model_state = model.state_dict()
    ema_state = ema_model.state_dict()
    for key, ema_value in ema_state.items():
        model_value = model_state[key]
        if not torch.is_floating_point(ema_value):
            ema_value.copy_(model_value)
            continue
        ema_value.mul_(decay).add_(model_value.detach(), alpha=1.0 - decay)


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device: torch.device,
    epoch: int,
    print_freq: int = 20,
    amp: bool = False,
    scaler=None,
    max_grad_norm: float = 0.0,
    accumulate_steps: int = 1,
    warmup_steps: int = 1000,
    warmup_factor: float = 0.001,
    ema_model=None,
    ema_decay: float = 0.9998,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    model.train()
    running = {"loss_total": 0.0}
    num_steps = 0
    num_samples = 0
    iter_time_sum = 0.0
    epoch_start = time.time()
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    accumulate_steps = max(1, int(accumulate_steps))

    optimizer.zero_grad(set_to_none=True)

    for step, (images, targets) in enumerate(data_loader, start=1):
        iter_start = time.time()
        num_steps += 1
        num_samples += len(images)
        images = [image.to(device) for image in images]
        targets = _move_targets_to_device(list(targets), device)

        if epoch == 0 and warmup_steps > 0 and step <= warmup_steps:
            alpha = float(step) / float(max(1, warmup_steps))
            factor = warmup_factor + (1.0 - warmup_factor) * alpha
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * factor

        with torch.amp.autocast(device_type=device.type, enabled=amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_for_backward = losses / accumulate_steps

        loss_value = float(losses.detach().item())
        if not math.isfinite(loss_value):
            breakdown = ", ".join(
                f"{name}={float(loss.detach().item()):.4f}" for name, loss in loss_dict.items()
            )
            raise RuntimeError(f"Loss is {loss_value}, stopping training. Breakdown: {breakdown}")

        should_step = (step % accumulate_steps == 0) or (step == len(data_loader))

        if amp:
            if scaler is None:
                raise ValueError("amp=True requires a valid GradScaler instance.")
            scaler.scale(loss_for_backward).backward()
            if should_step:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss_for_backward.backward()
            if should_step:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if should_step and ema_model is not None:
            _update_ema(model, ema_model, decay=ema_decay)

        running["loss_total"] += loss_value
        for loss_name, loss_tensor in loss_dict.items():
            key = f"loss_{loss_name}"
            running[key] = running.get(key, 0.0) + float(loss_tensor.detach().item())
        iter_time_sum += time.time() - iter_start

        if step % max(1, print_freq) == 0:
            avg_total = running["loss_total"] / num_steps
            elapsed = time.time() - epoch_start
            message = (
                f"Epoch[{epoch + 1}] Step[{step}/{len(data_loader)}] "
                f"Loss {avg_total:.4f} LR {optimizer.param_groups[0]['lr']:.6f} "
                f"Time {elapsed / step:.3f}s/iter"
            )
            if logger is not None:
                logger.info(message)
            else:
                print(message)

    for key in list(running.keys()):
        if key.startswith("loss_"):
            running[key] /= max(1, num_steps)

    epoch_time_sec = time.time() - epoch_start
    running["epoch_time_sec"] = epoch_time_sec
    running["avg_iter_time_sec"] = iter_time_sum / max(1, num_steps)
    running["num_steps"] = float(num_steps)
    running["num_samples"] = float(num_samples)
    running["samples_per_sec"] = num_samples / max(1e-8, epoch_time_sec)
    if device.type == "cuda":
        running["max_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    else:
        running["max_memory_mb"] = 0.0

    if epoch == 0 and warmup_steps > 0:
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
    return running


def _evaluate_single_iou(coco_gt, results: List[Dict], iou_type: str) -> Dict[str, float]:
    metric_names = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]

    if COCOeval is None:
        raise ImportError("COCO evaluation requires pycocotools. Install with: pip install pycocotools")

    if len(results) == 0:
        return {f"{iou_type}_{name}": 0.0 for name in metric_names}

    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = {}
    for idx, name in enumerate(metric_names):
        metrics[f"{iou_type}_{name}"] = float(evaluator.stats[idx])
    return metrics


def _to_image_id(image_id_value) -> int:
    if torch.is_tensor(image_id_value):
        if image_id_value.ndim == 0:
            return int(image_id_value.item())
        return int(image_id_value[0].item())
    if isinstance(image_id_value, (list, tuple)):
        return int(image_id_value[0])
    return int(image_id_value)


@torch.no_grad()
def evaluate_coco(
    model,
    data_loader,
    device: torch.device,
    contiguous_to_category_id: Dict[int, int],
    task: str,
    print_freq: int = 50,
    mask_threshold: float = 0.5,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    if task not in {"detection", "segmentation"}:
        raise ValueError("task must be 'detection' or 'segmentation'.")
    if task == "segmentation" and coco_mask is None:
        raise ImportError("Segmentation evaluation requires pycocotools. Install with: pip install pycocotools")

    model.eval()
    coco_gt = data_loader.dataset.coco

    bbox_results: List[Dict] = []
    segm_results: List[Dict] = []
    num_images = 0
    eval_start = time.time()

    for step, (images, targets) in enumerate(data_loader, start=1):
        num_images += len(images)
        images = [image.to(device) for image in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = _to_image_id(target["image_id"])

            boxes = output.get("boxes", torch.zeros((0, 4)))
            labels = output.get("labels", torch.zeros((0,), dtype=torch.int64))
            scores = output.get("scores", torch.zeros((0,), dtype=torch.float32))

            boxes = boxes.detach().cpu()
            labels = labels.detach().cpu()
            scores = scores.detach().cpu()

            for box, label, score in zip(boxes, labels, scores):
                contiguous_label = int(label.item())
                category_id = contiguous_to_category_id.get(contiguous_label, None)
                if category_id is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                bbox_results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(category_id),
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score.item()),
                    }
                )

            if task == "segmentation" and "masks" in output:
                masks = output["masks"].detach().cpu()
                for mask, label, score in zip(masks, labels, scores):
                    contiguous_label = int(label.item())
                    category_id = contiguous_to_category_id.get(contiguous_label, None)
                    if category_id is None:
                        continue

                    binary_mask = (mask[0] > mask_threshold).numpy().astype(np.uint8)
                    if binary_mask.max() == 0:
                        continue

                    encoded = coco_mask.encode(np.asfortranarray(binary_mask))
                    if isinstance(encoded, list):
                        encoded = encoded[0]
                    if isinstance(encoded.get("counts"), bytes):
                        encoded["counts"] = encoded["counts"].decode("utf-8")

                    segm_results.append(
                        {
                            "image_id": image_id,
                            "category_id": int(category_id),
                            "segmentation": encoded,
                            "score": float(score.item()),
                        }
                    )

        if step % max(1, print_freq) == 0:
            message = f"Eval Step[{step}/{len(data_loader)}]"
            if logger is not None:
                logger.info(message)
            else:
                print(message)

    metrics = _evaluate_single_iou(coco_gt, bbox_results, iou_type="bbox")
    if task == "segmentation":
        metrics.update(_evaluate_single_iou(coco_gt, segm_results, iou_type="segm"))
    eval_time_sec = time.time() - eval_start
    metrics["meta_eval_time_sec"] = eval_time_sec
    metrics["meta_num_images"] = float(num_images)
    metrics["meta_images_per_sec"] = num_images / max(1e-8, eval_time_sec)
    metrics["meta_bbox_predictions"] = float(len(bbox_results))
    if task == "segmentation":
        metrics["meta_segm_predictions"] = float(len(segm_results))
    if logger is not None:
        logger.info(
            "Eval summary: images=%d time=%.2fs speed=%.2f img/s bbox_preds=%d%s",
            num_images,
            eval_time_sec,
            num_images / max(1e-8, eval_time_sec),
            len(bbox_results),
            f" segm_preds={len(segm_results)}" if task == "segmentation" else "",
        )
    return metrics
