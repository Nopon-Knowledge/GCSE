from __future__ import annotations

from pathlib import Path
from typing import Dict

DEFAULT_COCO_ROOT = Path("/storage/home/402005/datasets/coco")


def _resolve_default_image_dir(root: Path, split: str) -> Path:
    images_layout = root / "images" / split
    flat_layout = root / split

    if images_layout.exists():
        return images_layout
    if flat_layout.exists():
        return flat_layout

    # Prefer the COCO root/images/* layout used on the target platform.
    if (root / "images").exists():
        return images_layout
    return flat_layout


def resolve_coco_paths(
    coco_root: str | Path = DEFAULT_COCO_ROOT,
    train_images: str | Path | None = None,
    train_annotations: str | Path | None = None,
    val_images: str | Path | None = None,
    val_annotations: str | Path | None = None,
) -> Dict[str, Path]:
    root = Path(coco_root).expanduser()
    return {
        "coco_root": root,
        "train_images": Path(train_images).expanduser() if train_images else _resolve_default_image_dir(root, "train2017"),
        "train_annotations": (
            Path(train_annotations).expanduser()
            if train_annotations
            else root / "annotations" / "instances_train2017.json"
        ),
        "val_images": Path(val_images).expanduser() if val_images else _resolve_default_image_dir(root, "val2017"),
        "val_annotations": (
            Path(val_annotations).expanduser()
            if val_annotations
            else root / "annotations" / "instances_val2017.json"
        ),
    }


def ensure_coco_paths_exist(paths: Dict[str, Path], require_train: bool, require_val: bool) -> None:
    required_keys = []
    if require_train:
        required_keys.extend(["train_images", "train_annotations"])
    if require_val:
        required_keys.extend(["val_images", "val_annotations"])

    errors = []
    for key in required_keys:
        path = paths[key]
        if key.endswith("images"):
            if not path.is_dir():
                errors.append(f"{key}: expected existing directory -> {path}")
        else:
            if not path.is_file():
                errors.append(f"{key}: expected existing file -> {path}")

    if errors:
        message = (
            "COCO path validation failed:\n"
            + "\n".join(f"  - {err}" for err in errors)
            + f"\nResolved coco_root: {paths['coco_root']}"
        )
        raise FileNotFoundError(message)
