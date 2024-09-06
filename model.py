# This file contains the model definition.
# Creat Model based on config file.
import torch

from torch import nn

from dataloader import config_reader

CONFIG_PATH = "yolov3.cfg"


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    def forward(self, x):
        return x


def add_convolutional(info, module, index, pre_filters_num, output_filters):
    activation = info['activation']
    try:
        batch_normalize = int(info['batch_normalize'])
        bias = False
    except:
        batch_normalize = 0
        bias = True

    filters_num = int(info['filters'])
    padding = int(info['pad'])
    kernel_size = int(info['size'])
    stride = int(info['stride'])
    pad = (kernel_size - 1) // 2 if padding else 0

    conv = nn.Conv2d(pre_filters_num, filters_num, kernel_size, stride, pad, bias=bias)
    module.add_module('conv_{0}'.format(index), conv)

    if batch_normalize:
        bn = nn.BatchNorm2d(filters_num)
        module.add_module('batch_norm_{0}'.format(index), bn)

    if activation == 'leaky':
        activn = nn.LeakyReLU(0.1, inplace=True)
        module.add_module('leaky_{0}'.format(index), activn)

    return module, filters_num, output_filters


def add_upsample(info, module, index, pre_filters_num, output_filters):
    stride = int(info['stride'])
    upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
    module.add_module('upsample_{0}'.format(index), upsample)

    return module, pre_filters_num, output_filters


def add_shortcut(info, module, index, pre_filters_num, output_filters):
    shortcut = EmptyLayer()
    module.add_module('shortcut_{0}'.format(index), shortcut)

    return module, pre_filters_num, output_filters


def add_yolo(info, module, index, pre_filters_num, output_filters):
    mask = info['mask'].split(',')
    mask = [int(x) for x in mask]

    anchors = info["anchors"].split(',')
    anchors = [int(a) for a in anchors]
    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
    anchors = [anchors[i] for i in mask]

    detection = DetectionLayer(anchors)
    module.add_module('Detection_{0}'.format(index), detection)

    return module, pre_filters_num, output_filters


def add_route(info, module, index, pre_filters_num, output_filters):
    layers = info['layers'].split(',')
    start = int(layers[0])

    try:
        end = int(layers[1])
    except:
        end = 0

    if start > 0:
        start = start - index
    if end > 0:
        end = end - index
    route = EmptyLayer()
    module.add_module('route_{0}'.format(index), route)
    if end < 0:
        filters_num = output_filters[index + start] + output_filters[index + end]
    else:
        filters_num = output_filters[index + start]

    return module, filters_num, output_filters


def add_layers(index, block_info, pre_filters, output_filters):
    """
    Add a block of layers to the model.
    :param index: The index of the block in Network.
    :param block_info: Information of this block.
    :param pre_filters: Number of filters in layer.
    :param output_filters: A list of output filters num.
    :return: All these parameters are updated.
    """
    layer_type = block_info["type"]
    module = nn.Sequential()
    layer_dict = {
        "convolutional": lambda: add_convolutional(block_info, module, index, pre_filters, output_filters),
        "upsample": lambda: add_upsample(block_info, module, index, pre_filters, output_filters),
        "route": lambda: add_route(block_info, module, index, pre_filters, output_filters),
        "shortcut": lambda: add_shortcut(block_info, module, index, pre_filters, output_filters),
        "yolo": lambda: add_yolo(block_info, module, index, pre_filters, output_filters),
    }
    module, filters, output_filters = layer_dict[layer_type]()

    return filters, output_filters, module


def create_model(config_list, module_list=nn.ModuleList()):
    """
    Create a model from config.
    :param config_list: List of all blocks config.
    :param module_list: ModuleList to add blocks in.
    :return: Net info and Model. module_list is model.
    """
    output_filters = []
    prev_filters = 3
    net_info = config_list[0]  # Captures the information about the input and pre-processing

    for index, info in enumerate(config_list[1:]):
        # check the type of block
        # create a new module for the block
        # append to module_list
        filters, output_filters, module = add_layers(index, info, prev_filters, output_filters)
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list


def compile_model():
    """
    Compile the model.
    :return: module.
    """
    _config = config_reader(CONFIG_PATH)
    _netinfo, compiled_model = create_model(_config)

    return compiled_model


if __name__ == '__main__':
    config = config_reader(CONFIG_PATH)
    netinfo, model = create_model(config)
    print(netinfo, model)
