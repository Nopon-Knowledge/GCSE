import torch.nn as nn
from torchvision.models import ResNet

from gcse.gcse_module import GCSEAttention
# from senet.se_module import SELayer  # SENet module (manual switch for comparison)

# 3x3 conv with padding; bias=False since BN follows.
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class GCSEBasicBlock(nn.Module):
    """
    Basic ResNet block with GCSE attention, for 18/34 variants.
    Structure: Conv-BN-ReLU -> Conv-BN -> GCSE -> residual add -> ReLU
    """
    expansion = 1

    # inplanes: input channels
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.se = SELayer(planes, reduction)  # SENet channel attention (optional compare)
        self.gcse = GCSEAttention(planes)

        self.downsample = downsample  # 1x1 conv when size/channel mismatch
        self.stride = stride

    def forward(self, x):
        residual = x  # Cache residual
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.se(out)
        out = self.gcse(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GCSEBottleneck(nn.Module):
    """
    Bottleneck block with GCSE attention, for 50/101/152 variants.
    Structure: 1x1 reduce -> 3x3 -> 1x1 expand -> GCSE -> residual add -> ReLU
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        # self.se = SELayer(planes * 4, reduction)
        self.gcse = GCSEAttention(planes * 4)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out = self.se(out)
        out = self.gcse(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def gcse_resnet18(num_classes=1_000):
    """Build a GCSE-ResNet-18 model."""
    model = ResNet(GCSEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)  # Adaptive global avg pool for variable input size
    return model


def gcse_resnet34(num_classes=1_000):
    """Build a GCSE-ResNet-34 model."""
    model = ResNet(GCSEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def gcse_resnet50(num_classes=1_000, pretrained=False):
    """Build a GCSE-ResNet-50 model."""
    model = ResNet(GCSEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrained:
        raise NotImplementedError("Pretrained weights for GCSE-ResNet50 are not provided.")
    return model


def gcse_resnet101(num_classes=1_000):
    """Build a GCSE-ResNet-101 model."""
    model = ResNet(GCSEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def gcse_resnet152(num_classes=1_000):
    """Build a GCSE-ResNet-152 model."""
    model = ResNet(GCSEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


class CifarGCSEBasicBlock(nn.Module):
    """GCSE basic block for CIFAR inputs."""
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.se = SELayer(planes, reduction)
        self.gcse = GCSEAttention(planes)
        
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out = self.se(out)
        out = self.gcse(out)

        out += residual
        out = self.relu(out)

        return out


class CifarGCSEResNet(nn.Module):
    """
    GCSE-ResNet for CIFAR (32x32).
    block: basic block type (CifarGCSEBasicBlock)
    n_size: blocks per stage; e.g., 3->20 layers, 5->32, 9->56
    """
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super().__init__()
        self.inplane = 16
        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 16, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(
            block, 32, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(
            block, 64, blocks=n_size, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        # Kaiming init for conv, BN gamma=1 and beta=0.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        # Build a stage of residual blocks; first block may change stride.
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes  # Next block input channels = previous output channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CifarGCSEPreActResNet(CifarGCSEResNet):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super().__init__(block, n_size, num_classes, reduction)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def gcse_resnet20(**kwargs):
    """Build a GCSE-ResNet-20 model for CIFAR."""
    model = CifarGCSEResNet(CifarGCSEBasicBlock, 3, **kwargs)
    return model


def gcse_resnet32(**kwargs):
    """Build a GCSE-ResNet-32 model for CIFAR."""
    model = CifarGCSEResNet(CifarGCSEBasicBlock, 5, **kwargs)
    return model


def gcse_resnet56(**kwargs):
    """Build a GCSE-ResNet-56 model for CIFAR."""
    model = CifarGCSEResNet(CifarGCSEBasicBlock, 9, **kwargs)
    return model


def gcse_preactresnet20(**kwargs):
    """Build a GCSE-PreAct-ResNet-20 model for CIFAR."""
    model = CifarGCSEPreActResNet(CifarGCSEBasicBlock, 3, **kwargs)
    return model


def gcse_preactresnet32(**kwargs):
    """Build a GCSE-PreAct-ResNet-32 model for CIFAR."""
    model = CifarGCSEPreActResNet(CifarGCSEBasicBlock, 5, **kwargs)
    return model


def gcse_preactresnet56(**kwargs):
    """Build a GCSE-PreAct-ResNet-56 model for CIFAR."""
    model = CifarGCSEPreActResNet(CifarGCSEBasicBlock, 9, **kwargs)
    return model
