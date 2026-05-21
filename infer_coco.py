import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F

from coco_exp.coco_models import build_coco_model, parse_mapping_payload
from coco_exp.coco_paths import DEFAULT_COCO_ROOT

DEFAULT_INFER_CHECKPOINT = "/storage/home/402005/python_project/gcse/runs/coco_res50_pretrain_v1/best.pth"
DEFAULT_INFER_INPUT = str(DEFAULT_COCO_ROOT / "images" / "val2017")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COCO inference for detection/segmentation")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_INFER_CHECKPOINT,
        help="Path to train_coco.py checkpoint.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INFER_INPUT,
        help="Input image path or directory. Default: /storage/home/402005/datasets/coco/images/val2017",
    )
    parser.add_argument("--output-dir", type=str, default="runs/coco_infer")

    parser.add_argument("--task", choices=["detection", "segmentation"], default=None)
    parser.add_argument("--backbone", choices=["gcse_resnet50", "resnet50"], default=None)
    parser.add_argument("--category-mapping", type=str, default=None, help="Optional mapping json override")

    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--trainable-backbone-layers", type=int, default=5)
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--gcse-backbone-weights", type=str, default=None)

    return parser.parse_args()



def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}



def collect_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if not is_image_file(input_path):
            raise ValueError(f"Input file is not an image: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    images = [path for path in sorted(input_path.rglob("*")) if path.is_file() and is_image_file(path)]
    if not images:
        raise ValueError(f"No images found under directory: {input_path}")
    return images



def load_mapping_payload(args, checkpoint):
    if args.category_mapping:
        return json.loads(Path(args.category_mapping).read_text(encoding="utf-8"))
    return checkpoint.get("category_mapping", None)



def color_for_label(label: int):
    # Deterministic vivid-ish color per label id.
    base = (label * 37) % 255
    return ((base + 50) % 255, (base * 2 + 80) % 255, (base * 3 + 110) % 255)



def draw_predictions(
    image: Image.Image,
    output,
    score_threshold: float,
    mask_threshold: float,
    contiguous_to_cat: Dict[int, int],
    cat_to_name: Dict[int, str],
    task: str,
):
    vis = image.convert("RGBA")
    draw = ImageDraw.Draw(vis)

    boxes = output.get("boxes", torch.zeros((0, 4))).detach().cpu()
    labels = output.get("labels", torch.zeros((0,), dtype=torch.int64)).detach().cpu()
    scores = output.get("scores", torch.zeros((0,), dtype=torch.float32)).detach().cpu()
    masks = output.get("masks", None)
    if masks is not None:
        masks = masks.detach().cpu()

    records = []
    for index, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        score_value = float(score.item())
        if score_value < score_threshold:
            continue

        contiguous_label = int(label.item())
        category_id = int(contiguous_to_cat.get(contiguous_label, contiguous_label))
        category_name = cat_to_name.get(category_id, str(category_id))

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        color = color_for_label(contiguous_label)

        if task == "segmentation" and masks is not None and index < len(masks):
            binary_mask = (masks[index, 0] > mask_threshold).numpy().astype(np.uint8)
            if binary_mask.max() > 0:
                overlay = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 4), dtype=np.uint8)
                overlay[binary_mask > 0] = (*color, 90)
                vis = Image.alpha_composite(vis, Image.fromarray(overlay, mode="RGBA"))
                draw = ImageDraw.Draw(vis)

        draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=2)

        caption = f"{category_name} {score_value:.2f}"
        text_y = max(0.0, y1 - 14)
        text_w = max(40, 7 * len(caption))
        draw.rectangle([x1, text_y, x1 + text_w, text_y + 14], fill=color + (180,))
        draw.text((x1 + 2, text_y + 1), caption, fill=(255, 255, 255, 255))

        record = {
            "bbox": [x1, y1, x2, y2],
            "score": score_value,
            "label": contiguous_label,
            "category_id": category_id,
            "category_name": category_name,
        }
        if task == "segmentation" and masks is not None and index < len(masks):
            mask_area = int((masks[index, 0] > mask_threshold).sum().item())
            record["mask_area"] = mask_area
        records.append(record)

    return vis.convert("RGB"), records



def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please pass --checkpoint or place model at default path: {DEFAULT_INFER_CHECKPOINT}"
        )

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

    task = args.task or checkpoint.get("task", "detection")
    backbone = args.backbone or checkpoint.get("backbone", "gcse_resnet50")
    mapping_payload = load_mapping_payload(args, checkpoint)

    cat_to_contiguous = {}
    contiguous_to_cat = {}
    cat_to_name = {}
    if mapping_payload is not None:
        cat_to_contiguous, contiguous_to_cat, cat_to_name = parse_mapping_payload(mapping_payload)

    num_classes = int(checkpoint.get("num_classes", 0))
    if num_classes <= 0:
        if contiguous_to_cat:
            num_classes = len(contiguous_to_cat) + 1
        else:
            raise ValueError("Unable to infer num_classes. Provide a checkpoint produced by train_coco.py.")

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
    model.eval()

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)
    model.to(device)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(input_path)

    all_predictions = {}

    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            tensor = F.to_tensor(image).to(device)
            output = model([tensor])[0]

            vis_image, records = draw_predictions(
                image=image,
                output=output,
                score_threshold=args.score_threshold,
                mask_threshold=args.mask_threshold,
                contiguous_to_cat=contiguous_to_cat,
                cat_to_name=cat_to_name,
                task=task,
            )

            output_image_name = f"{image_path.stem}_pred{image_path.suffix}"
            output_image_path = output_dir / output_image_name
            vis_image.save(output_image_path)

            all_predictions[str(image_path)] = records
            print(f"Saved prediction image: {output_image_path}")

    predictions_json_path = output_dir / "predictions.json"
    predictions_json_path.write_text(
        json.dumps(all_predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved prediction records: {predictions_json_path}")


if __name__ == "__main__":
    main()
