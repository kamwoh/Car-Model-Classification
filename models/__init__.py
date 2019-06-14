import torchvision

from models.network_v1 import NetworkV1
from models.network_v2 import NetworkV2
from models.network_v3 import NetworkV3


def construct_model(config, num_classes, num_makes, num_types):
    if config['arch'] == 'resnext50':
        base = torchvision.models.resnext50_32x4d(pretrained=True)
    elif config['arch'] == 'resnet34':
        base = torchvision.models.resnet34(pretrained=True)
    else:  # mobilenetv2
        base = torchvision.models.mobilenet_v2(pretrained=True)

    if config['version'] == 1:
        model = NetworkV1(base, num_classes)
    elif config['version'] == 2:
        model = NetworkV2(base, num_classes, num_makes, num_types)
    else:
        model = NetworkV3(base, num_classes, num_makes, num_types)

    return model
