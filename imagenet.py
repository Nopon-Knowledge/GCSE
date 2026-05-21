import torch
from torch.nn import functional as F

# homura provides distributed init, optimizers/schedulers, callbacks, and logging.
from homura import callbacks, init_distributed, lr_scheduler, optim, reporters, is_distributed
from homura.trainers import SupervisedTrainer
# DATASET_REGISTRY returns dataset factories by name (ImageNet root can be overridden via IMAGENET_ROOT).
from homura.vision import DATASET_REGISTRY
from gcse.gcse_resnet import gcse_resnet50
# SENet comparison (optional): from senet.se_resnet import se_resnet50

import os
os.environ["IMAGENET_ROOT"] = r"D:\data\imagenet"


def main():
    # If launched in distributed mode (e.g., torch.distributed.launch), init process group.
    if is_distributed():
        init_distributed()

    # Build GCSE-ResNet50 with 1000 classes by default.
    model = gcse_resnet50(num_classes=1000)
    state = torch.load("gcse_resnet50.pth", map_location="cpu")
    model.load_state_dict(state)
    
    # Linear scaling for initial LR: 0.6/1024 * batch_size
    optimizer = optim.SGD(lr=0.6 / 1024 * args.batch_size, momentum=0.9, weight_decay=1e-4)
    # Multi-step decay at epochs 50 and 70.
    scheduler = lr_scheduler.MultiStepLR([50, 70])
    # Get ImageNet train/val sets (root from IMAGENET_ROOT or ~/.torch/data/imagenet).
    train_loader, test_loader = DATASET_REGISTRY("imagenet")(args.batch_size)

    # Callbacks/logs: Top1/Top5, loss, weight save, TensorBoard, progress bar.
    c = [
        callbacks.AccuracyCallback(),
        callbacks.AccuracyCallback(k=5),
        callbacks.LossCallback(),
        callbacks.WeightSave("."),
        reporters.TensorboardReporter("."),
        reporters.TQDMReporter(range(args.epochs)),
    ]

    # Use SupervisedTrainer to wrap train/val loops.
    with SupervisedTrainer(model, optimizer, F.cross_entropy, callbacks=c, scheduler=scheduler,) as trainer:
        for _ in c[-1]:
            trainer.train(train_loader)
            trainer.test(test_loader)


if __name__ == "__main__":
    import argparse
    import warnings

    # Suppress some EXIF warnings.
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    # Basic run args: epochs, batch size, local rank (for distributed).
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--local_rank", type=int, default=-1)
    args = p.parse_args()

    main()
