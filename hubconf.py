dependencies = ["torch", "math"]


def gcse_resnet20(**kwargs):
    from gcse.gcse_resnet import gcse_resnet20 as _gcse_resnet20

    return _gcse_resnet20(**kwargs)


def gcse_resnet56(**kwargs):
    from gcse.gcse_resnet import gcse_resnet56 as _gcse_resnet56

    return _gcse_resnet56(**kwargs)


def gcse_resnet50(**kwargs):
    from gcse.gcse_resnet import gcse_resnet50 as _gcse_resnet50

    return _gcse_resnet50(**kwargs)


def gcse_resnet101(**kwargs):
    from gcse.gcse_resnet import gcse_resnet101 as _gcse_resnet101

    return _gcse_resnet101(**kwargs)
