# This file contains the model definition.
# Creat Model based on config file.
from torch import nn

from dataloader import config_reader, CONFIG_PATH


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        pass


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def add_convolutional(info, module, index, pre_filters_num, output_filters):
    """
    Add a convolutional block to module.
    :param output_filters:
    :param info: Config of block
    :param pre_filters_num: Number of filters in previous layer
    :param module: Module to add block to
    :param index: Index of block
    :return: Same as input but exclude index.
    """
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

    return module, index, filters_num, output_filters


def add_upsample(info, module, index, pre_filters_num, output_filters):
    """
    Add a block with specific info to module list.
    :param info: Block info.
    :param module: Module to add block to.
    :param index: Index of block in model list,
    :param pre_filters_num: Filters size of previous block.
    :param output_filters: Filters size of current block.
    :return: Same as input but exclude index.
    """
    stride = int(info['stride'])
    upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
    module.add_module('upsample_{0}'.format(index), upsample)

    return module, index, pre_filters_num, output_filters


def add_route(info, module, index, pre_filters_num, output_filters):
    """
    Add a block with specific info to module list.
    :param info: Block info.
    :param module: Module to add block to.
    :param index: Index of block in model list,
    :param pre_filters_num: Filters size of previous block.
    :param output_filters: Filters size of current block.
    :return: Same as input but exclude index.
    """

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

    return module, index, filters_num, output_filters


def add_shortcut(info, module, index, pre_filters_num, output_filters):
    """
    Add a block with specific info to module list.
    :param info: Block info.
    :param module: Module to add block to.
    :param index: Index of block in model list,
    :param pre_filters_num: Filters size of previous block.
    :param output_filters: Filters size of current block.
    :return: Same as input but exclude index.
    """
    shortcut = EmptyLayer()
    module.add_module('shortcut_{0}'.format(index), shortcut)

    return module, index, pre_filters_num, output_filters


def add_yolo(info, module, index, pre_filters_num, output_filters):
    """
        Add a block with specific info to module list.
        :param info: Block info.
        :param module: Module to add block to.
        :param index: Index of block in model list,
        :param pre_filters_num: Filters size of previous block.
        :param output_filters: Filters size of current block.
        :return: Same as input but exclude index.
        """
    mask = info['mask'].split(',')
    mask = [int(x) for x in mask]

    anchors = info["anchors"].split(',')
    anchors = [int(a) for a in anchors]
    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
    anchors = [anchors[i] for i in mask]

    detection = DetectionLayer(anchors)
    module.add_module('Detection_{0}'.format(index), detection)

    return module, index, pre_filters_num, output_filters


def add_layers(info, module, index, pre_filters_num, output_filters):
    """
    Add a block with specific info to module list.
    :param info: Block info.
    :param module: Module to add block to.
    :param index: Index of block in model list,
    :param pre_filters_num: Filters size of previous block.
    :param output_filters: Filters size of current block.
    :return: Same as input but exclude index.
    """
    layer_type = info["type"]
    layer_dict = {
        "convolutional": lambda: add_convolutional(info, module, index, pre_filters_num, output_filters),
        "upsample": lambda: add_upsample(info, module, index, pre_filters_num, output_filters),
        "route": lambda: add_route(info, module, index, pre_filters_num, output_filters),
        "shortcut": lambda: add_shortcut(info, module, index, pre_filters_num, output_filters),
        "yolo": lambda: add_yolo(info, module, index, pre_filters_num, output_filters),
    }

    # 使用字典的get方法获取对应的函数，并执行它
    module, index, filters_num, output_filters = layer_dict.get(layer_type, lambda: None)()

    return module, index, filters_num, output_filters


def create_model(config_list, module_lists=nn.ModuleList(), init_filters=3):
    """
    Create a model from config.
    :param config_list: List of block config.
    :param module_lists: ModuleList to add blocks to.
    :param init_filters: Init filters size.
    :return: Net info and Model.
    """
    output_filters = []
    for index, info in enumerate(config_list[1:]):
        module = nn.Sequential()
        module, index, filters_num, output_filters = add_layers(info, module, index, init_filters, output_filters)

        module_lists.append(module)
        output_filters.append(filters_num)
        init_filters = filters_num

    return config_list[0], module_lists


if __name__ == '__main__':
    config = config_reader(CONFIG_PATH)
    netinfo, module_list = create_model(config)
    print(netinfo, module_list)
