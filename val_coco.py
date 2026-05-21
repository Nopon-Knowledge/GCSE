import argparse
import json
from pathlib import Path

import torch

from coco_exp.coco_dataloader import build_coco_dataloader
from coco_exp.coco_engine import evaluate_coco
from coco_exp.coco_models import build_coco_model, parse_mapping_payload
from coco_exp.coco_paths import DEFAULT_COCO_ROOT, ensure_coco_paths_exist, resolve_coco_paths

DEFAULT_VAL_CHECKPOINT = "/storage/home/402005/python_project/gcse/runs/coco_res50_pretrain_v1/best.pth"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COCO validation/evaluation")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_VAL_CHECKPOINT,
        help="Path to train_coco.py checkpoint.",
    )
    parser.add_argument(
        "--coco-root",
        type=str,
        default=str(DEFAULT_COCO_ROOT),
        help="COCO root directory. Defaults to /storage/home/402005/datasets/coco",
    )
    parser.add_argument("--val-images", type=str, default=None, help="COCO val image directory")
    parser.add_argument("--val-annotations", type=str, default=None, help="COCO val annotation json")

    parser.add_argument("--task", choices=["detection", "segmentation"], default=None)
    parser.add_argument("--backbone", choices=["gcse_resnet50", "resnet50"], default=None)
    parser.add_argument("--category-mapping", type=str, default=None, help="Optional mapping json override")

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--mask-threshold", type=float, default=0.5)

    parser.add_argument("--trainable-backbone-layers", type=int, default=5)
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--gcse-backbone-weights", type=str, default=None)

    return parser.parse_args()



def load_mapping(args, checkpoint):
    if args.category_mapping:
        mapping_payload = json.loads(Path(args.category_mapping).read_text(encoding="utf-8"))
        return mapping_payload

    mapping_payload = checkpoint.get("category_mapping", None)
    if mapping_payload is not None:
        return mapping_payload

    return None



def main() -> None:
    args = parse_args()
    coco_paths = resolve_coco_paths(
        coco_root=args.coco_root,
        val_images=args.val_images,
        val_annotations=args.val_annotations,
    )
    ensure_coco_paths_exist(coco_paths, require_train=False, require_val=True)
    args.val_images = str(coco_paths["val_images"])
    args.val_annotations = str(coco_paths["val_annotations"])

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please pass --checkpoint or place model at default path: {DEFAULT_VAL_CHECKPOINT}"
        )

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    task = args.task or checkpoint.get("task", "detection")
    backbone = args.backbone or checkpoint.get("backbone", "gcse_resnet50")

    mapping_payload = load_mapping(args, checkpoint)
    cat_to_contiguous = None
    contiguous_to_cat = None

    if mapping_payload is not None:
        cat_to_contiguous, contiguous_to_cat, _ = parse_mapping_payload(mapping_payload)

    val_dataset, val_loader = build_coco_dataloader(
        images_dir=args.val_images,
        annotation_file=args.val_annotations,
        task=task,
        batch_size=args.batch_size,
        workers=args.workers,
        is_train=False,
        category_id_to_contiguous=cat_to_contiguous,
        remove_crowd=False,
        horizontal_flip_prob=0.0,
    )

    if contiguous_to_cat is None or len(contiguous_to_cat) == 0:
        contiguous_to_cat = val_dataset.contiguous_to_category_id

    num_classes = len(contiguous_to_cat) + 1

    model = build_coco_model(
        task=task,
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained_backbone=args.pretrained_backbone,
        gcse_backbone_weights=args.gcse_backbone_weights,
        trainable_backbone_layers=args.trainable_backbone_layers,
        min_size=args.min_size,
        max_size=args.max_size,
    )
    model.load_state_dict(checkpoint["model"])

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    model.to(device)
    metrics = evaluate_coco(
        model=model,
        data_loader=val_loader,
        device=device,
        contiguous_to_category_id=contiguous_to_cat,
        task=task,
        print_freq=args.print_freq,
        mask_threshold=args.mask_threshold,
    )

    print("Validation metrics:")
    for key in sorted(metrics.keys()):
        print(f"  {key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
