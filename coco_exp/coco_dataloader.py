from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoDetection
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F

try:
    from pycocotools import mask as coco_mask
except ImportError:  # pragma: no cover - runtime dependency
    coco_mask = None


class Compose:
    """Compose transforms that operate on (image, target)."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, image: Image.Image | torch.Tensor, target: Dict[str, torch.Tensor]):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class ToTensor:
    """Convert PIL image to float tensor in [0, 1]."""

    def __call__(self, image: Image.Image | torch.Tensor, target: Dict[str, torch.Tensor]):
        if isinstance(image, torch.Tensor):
            return image, target
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    """Randomly horizontally flip image, boxes and masks."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image | torch.Tensor, target: Dict[str, torch.Tensor]):
        if torch.rand(1).item() >= self.p:
            return image, target

        if isinstance(image, torch.Tensor):
            _, _, width = image.shape
            image = image.flip(-1)
        else:
            width, _ = image.size
            image = F.hflip(image)

        boxes = target["boxes"]
        if boxes.numel() > 0:
            boxes = boxes.clone()
            x_min = width - boxes[:, 2]
            x_max = width - boxes[:, 0]
            boxes[:, 0] = x_min
            boxes[:, 2] = x_max
            target["boxes"] = boxes

        if "masks" in target:
            target["masks"] = target["masks"].flip(-1)

        return image, target


class RandomColorJitter:
    """Apply color jitter for stronger appearance augmentation."""

    def __init__(self, p: float = 0.8, strength: float = 0.2):
        self.p = float(max(0.0, min(1.0, p)))
        strength = max(0.0, float(strength))
        self.transform = ColorJitter(
            brightness=strength,
            contrast=strength,
            saturation=strength,
            hue=min(0.5, strength * 0.5),
        )

    def __call__(self, image: Image.Image | torch.Tensor, target: Dict[str, torch.Tensor]):
        if torch.rand(1).item() < self.p:
            image = self.transform(image)
        return image, target


class RandomGrayscale:
    """Occasionally drop color to improve robustness to illumination/style shifts."""

    def __init__(self, p: float = 0.05):
        self.p = float(max(0.0, min(1.0, p)))

    def __call__(self, image: Image.Image | torch.Tensor, target: Dict[str, torch.Tensor]):
        if torch.rand(1).item() >= self.p:
            return image, target
        if isinstance(image, torch.Tensor):
            # Keep 3 channels to preserve model input contract.
            gray = image.mean(dim=0, keepdim=True)
            image = gray.repeat(3, 1, 1)
        else:
            image = F.to_grayscale(image, num_output_channels=3)
        return image, target


def _convert_coco_segmentation_to_mask(segmentation, height: int, width: int) -> torch.Tensor:
    if coco_mask is None:
        raise ImportError(
            "pycocotools is required for COCO segmentation parsing. "
            "Install with: pip install pycocotools"
        )

    if isinstance(segmentation, list):
        if len(segmentation) == 0:
            return torch.zeros((height, width), dtype=torch.uint8)
        rles = coco_mask.frPyObjects(segmentation, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(segmentation, dict):
        rle = segmentation
    else:
        return torch.zeros((height, width), dtype=torch.uint8)

    decoded = coco_mask.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return torch.as_tensor(decoded, dtype=torch.uint8)


class CocoDetectionDataset(Dataset):
    """COCO dataset wrapper producing detection/segmentation targets for torchvision models."""

    def __init__(
        self,
        images_dir: str | Path,
        annotation_file: str | Path,
        task: str = "detection",
        transforms: Optional[Callable] = None,
        remove_crowd: bool = True,
        category_id_to_contiguous: Optional[Dict[int, int]] = None,
    ) -> None:
        if task not in {"detection", "segmentation"}:
            raise ValueError(f"Unsupported task: {task}. Expected 'detection' or 'segmentation'.")

        self.images_dir = Path(images_dir)
        self.annotation_file = Path(annotation_file)
        self.task = task
        self.transforms = transforms if transforms is not None else Compose([ToTensor()])
        self.remove_crowd = remove_crowd

        self.dataset = CocoDetection(root=str(self.images_dir), annFile=str(self.annotation_file))
        self.ids = self.dataset.ids
        self.coco = self.dataset.coco

        if category_id_to_contiguous is None:
            category_ids = sorted(self.coco.getCatIds())
            self.category_id_to_contiguous = {
                int(category_id): index + 1 for index, category_id in enumerate(category_ids)
            }
        else:
            self.category_id_to_contiguous = {
                int(category_id): int(contiguous)
                for category_id, contiguous in category_id_to_contiguous.items()
            }

        self.contiguous_to_category_id = {
            contiguous: category_id for category_id, contiguous in self.category_id_to_contiguous.items()
        }

        self.category_id_to_name: Dict[int, str] = {}
        cats = self.coco.loadCats(list(self.category_id_to_contiguous.keys()))
        for cat in cats:
            self.category_id_to_name[int(cat["id"])] = str(cat["name"])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, annotations = self.dataset[index]
        image_id = int(self.ids[index])
        width, height = image.size

        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []
        masks: List[torch.Tensor] = []

        for ann in annotations:
            crowd_flag = int(ann.get("iscrowd", 0))
            if self.remove_crowd and crowd_flag == 1:
                continue

            category_id = int(ann.get("category_id", -1))
            if category_id not in self.category_id_to_contiguous:
                continue

            bbox = ann.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                continue

            x, y, w, h = [float(v) for v in bbox]
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(width), x + max(w, 0.0))
            y2 = min(float(height), y + max(h, 0.0))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.category_id_to_contiguous[category_id])
            areas.append(float(ann.get("area", (x2 - x1) * (y2 - y1))))
            iscrowd.append(crowd_flag)

            if self.task == "segmentation":
                segmentation = ann.get("segmentation", [])
                mask = _convert_coco_segmentation_to_mask(segmentation, height, width)
                masks.append(mask)

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            area_tensor = torch.tensor(areas, dtype=torch.float32)
            iscrowd_tensor = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor,
        }

        if self.task == "segmentation":
            if masks:
                target["masks"] = torch.stack(masks, dim=0)
            else:
                target["masks"] = torch.zeros((0, height, width), dtype=torch.uint8)

        image, target = self.transforms(image, target)
        return image, target



def get_coco_transforms(
    is_train: bool,
    horizontal_flip_prob: float = 0.5,
    color_jitter_strength: float = 0.2,
    grayscale_prob: float = 0.05,
) -> Compose:
    transforms: List[Callable] = []
    if is_train:
        if color_jitter_strength > 0:
            transforms.append(RandomColorJitter(p=0.8, strength=color_jitter_strength))
        if grayscale_prob > 0:
            transforms.append(RandomGrayscale(p=grayscale_prob))
        if horizontal_flip_prob > 0:
            transforms.append(RandomHorizontalFlip(horizontal_flip_prob))
    transforms.append(ToTensor())
    return Compose(transforms)



def collate_fn(batch):
    return tuple(zip(*batch))



def build_coco_dataloader(
    images_dir: str | Path,
    annotation_file: str | Path,
    task: str,
    batch_size: int,
    workers: int,
    is_train: bool,
    category_id_to_contiguous: Optional[Dict[int, int]] = None,
    remove_crowd: bool = True,
    horizontal_flip_prob: float = 0.5,
    color_jitter_strength: float = 0.2,
    grayscale_prob: float = 0.05,
) -> Tuple[CocoDetectionDataset, DataLoader]:
    dataset = CocoDetectionDataset(
        images_dir=images_dir,
        annotation_file=annotation_file,
        task=task,
        transforms=get_coco_transforms(
            is_train,
            horizontal_flip_prob=horizontal_flip_prob,
            color_jitter_strength=color_jitter_strength,
            grayscale_prob=grayscale_prob,
        ),
        remove_crowd=remove_crowd,
        category_id_to_contiguous=category_id_to_contiguous,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=workers,
        pin_memory=True,
        drop_last=is_train,
        collate_fn=collate_fn,
    )
    return dataset, loader



def category_mapping_payload(dataset: CocoDetectionDataset) -> Dict[str, Dict[str, int | str]]:
    """Serialize category mapping for checkpoints and inference."""
    cat_to_contiguous = {
        str(category_id): int(contiguous)
        for category_id, contiguous in dataset.category_id_to_contiguous.items()
    }
    contiguous_to_cat = {
        str(contiguous): int(category_id)
        for contiguous, category_id in dataset.contiguous_to_category_id.items()
    }
    cat_to_name = {
        str(category_id): name for category_id, name in dataset.category_id_to_name.items()
    }
    return {
        "cat_id_to_contiguous": cat_to_contiguous,
        "contiguous_to_cat_id": contiguous_to_cat,
        "cat_id_to_name": cat_to_name,
    }
