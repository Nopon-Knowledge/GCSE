from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torchvision
from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from gcse.gcse_resnet import gcse_resnet50

SUPPORTED_TASKS = {"detection", "segmentation"}
SUPPORTED_BACKBONES = {"gcse_resnet50", "resnet50"}



def _extract_state_dict(checkpoint: Dict) -> Dict[str, torch.Tensor]:
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    elif "model_state" in checkpoint and isinstance(checkpoint["model_state"], dict):
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("module."):
            clean_state_dict[key[len("module.") :]] = value
        else:
            clean_state_dict[key] = value
    return clean_state_dict



def _load_local_resnet50_weights(backbone, weights_path: str | Path) -> bool:
    weights_path = Path(weights_path).expanduser()
    if not weights_path.is_file():
        return False

    checkpoint = torch.load(str(weights_path), map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded local ResNet50 backbone weights from {weights_path}. "
        f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
    )
    return True


def _build_torchvision_resnet50(pretrained: bool, local_weights_path: str | None = None):
    # In many training containers network access is disabled. If pretrained
    # weights cannot be downloaded, gracefully fall back to random init.
    try:
        from torchvision.models import ResNet50_Weights

        if not pretrained:
            return torchvision.models.resnet50(weights=None)

        if local_weights_path:
            model = torchvision.models.resnet50(weights=None)
            if _load_local_resnet50_weights(model, local_weights_path):
                return model
            print(
                f"[WARN] local resnet50 weights not found at {local_weights_path}. "
                "Trying torchvision weights download."
            )

        try:
            return torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except Exception as exc:
            print(
                "[WARN] Failed to load torchvision pretrained ResNet50 weights "
                f"({exc}). Falling back to random initialization."
            )
            return torchvision.models.resnet50(weights=None)
    except Exception:
        if not pretrained:
            return torchvision.models.resnet50(pretrained=False)
        if local_weights_path:
            model = torchvision.models.resnet50(pretrained=False)
            if _load_local_resnet50_weights(model, local_weights_path):
                return model
            print(
                f"[WARN] local resnet50 weights not found at {local_weights_path}. "
                "Trying torchvision weights download."
            )
        try:
            return torchvision.models.resnet50(pretrained=True)
        except Exception as exc:
            print(
                "[WARN] Failed to load torchvision pretrained ResNet50 weights "
                f"({exc}). Falling back to random initialization."
            )
            return torchvision.models.resnet50(pretrained=False)



def _freeze_backbone_layers(backbone, trainable_layers: int) -> None:
    if not 0 <= trainable_layers <= 5:
        raise ValueError("trainable_backbone_layers must be in [0, 5].")

    if trainable_layers == 5:
        return

    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    for name, parameter in backbone.named_parameters():
        if not any(name.startswith(layer_name) for layer_name in layers_to_train):
            parameter.requires_grad_(False)



def build_resnet_fpn_backbone(
    backbone_name: str,
    pretrained_backbone: bool = False,
    gcse_backbone_weights: str | None = None,
    resnet50_backbone_weights: str | None = None,
    trainable_backbone_layers: int = 5,
    out_channels: int = 256,
):
    if backbone_name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone {backbone_name}. "
            f"Expected one of: {sorted(SUPPORTED_BACKBONES)}"
        )

    if backbone_name == "gcse_resnet50":
        backbone = gcse_resnet50(num_classes=1000)
    else:
        backbone = _build_torchvision_resnet50(
            pretrained=pretrained_backbone,
            local_weights_path=resnet50_backbone_weights,
        )

    if gcse_backbone_weights:
        checkpoint = torch.load(Path(gcse_backbone_weights), map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded backbone weights from {gcse_backbone_weights}. "
            f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
        )

    _freeze_backbone_layers(backbone, trainable_backbone_layers)

    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    in_channels_list = [256, 512, 1024, 2048]

    fpn_backbone = BackboneWithFPN(
        backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool(),
    )
    return fpn_backbone



def build_coco_model(
    task: str,
    num_classes: int,
    backbone_name: str,
    pretrained_backbone: bool = False,
    gcse_backbone_weights: str | None = None,
    resnet50_backbone_weights: str | None = None,
    trainable_backbone_layers: int = 5,
    min_size: int = 800,
    max_size: int = 1333,
):
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task {task}. Expected one of: {sorted(SUPPORTED_TASKS)}")

    backbone = build_resnet_fpn_backbone(
        backbone_name=backbone_name,
        pretrained_backbone=pretrained_backbone,
        gcse_backbone_weights=gcse_backbone_weights,
        resnet50_backbone_weights=resnet50_backbone_weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    if task == "detection":
        model = FasterRCNN(backbone, num_classes=num_classes, min_size=min_size, max_size=max_size)
    else:
        model = MaskRCNN(backbone, num_classes=num_classes, min_size=min_size, max_size=max_size)
    return model



def trainable_parameters(model) -> list:
    return [param for param in model.parameters() if param.requires_grad]



def parse_mapping_payload(payload: Dict[str, Dict[str, int | str]]) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, str]]:
    cat_to_contiguous = {
        int(category_id): int(contiguous)
        for category_id, contiguous in payload.get("cat_id_to_contiguous", {}).items()
    }
    contiguous_to_cat = {
        int(contiguous): int(category_id)
        for contiguous, category_id in payload.get("contiguous_to_cat_id", {}).items()
    }
    cat_to_name = {
        int(category_id): str(name)
        for category_id, name in payload.get("cat_id_to_name", {}).items()
    }
    return cat_to_contiguous, contiguous_to_cat, cat_to_name
