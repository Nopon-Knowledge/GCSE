from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def build_imagenet_loaders(data_root="~/.torch/data/imagenet", batch_size=256, workers=8, img_size=224, resize_size=256):
    data_root = Path(data_root).expanduser()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(data_root / "val", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True)
    return train_loader, val_loader

# Example usage
train_loader, val_loader = build_imagenet_loaders("D:/data/imagenet", batch_size=128, workers=8)
for images, labels in train_loader:
    images = images.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    # Forward/backward ...
