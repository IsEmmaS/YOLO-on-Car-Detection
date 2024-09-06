import cv2
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from dataloader import config_reader
from model import CONFIG_PATH, create_model

config = config_reader(CONFIG_PATH)
info, module = create_model(config)


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """
    获取检测特征图并将其转换为2-D张量，其中张量的每一行对应于边界框的属性。
    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride

    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Sigmoid the  centre_X, centre_Y. and object confidence
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))
    prediction[:, :, :4] *= stride

    return prediction


class Darknet(nn.Module):
    def __init__(self, cfg: str):
        """
        初始化Darknet模型。
        参数:
            cfg (str): 配置文件路径。
        """
        super(Darknet, self).__init__()
        self.blocks_info = config_reader(cfg)
        self.net_info, self.module_list = create_model(self.blocks_info)

    def forward(self, x):
        """
        模型的前向传播。
        参数:
            x (torch.Tensor): 输入张量。
        返回:
            torch.Tensor: 输出张量。
        """
        outputs = {}
        write = 0
        index_of_detection = 0
        x = x.squeeze(1)  # 去掉channel维度
        for index, model_info in enumerate(self.blocks_info[1:]):
            model_name = model_info['type']

            # print(index, model_name, end=':\t')
            if model_name == 'convolutional' or model_name == 'upsample':
                x = self.module_list[index](x)

            elif model_name == 'route':
                layers = model_info['layers'].split(',')
                layers = [int(i) for i in layers]

                if layers[0] > 0:
                    layers[0] -= index

                if len(layers) == 1:
                    x = outputs[index + (layers[0])]

                else:
                    if layers[1] > 0:
                        layers[1] -= index

                    map1 = outputs[index + layers[0]]
                    map2 = outputs[index + layers[1]]

                    x = torch.cat(tensors=(map1, map2), dim=1)

            elif model_name == 'shortcut':
                from_ = int(model_info['from'])
                x = outputs[index - 1] + outputs[index + from_]

            elif model_name == 'yolo':
                index_of_detection += 1
                anchors = self.module_list[index][0].anchors
                inp_dim = int(self.net_info['height'])

                class_num = int(model_info['classes'])

                x = x.data
                x = predict_transform(x, inp_dim=inp_dim, anchors=anchors, num_classes=class_num,
                                      CUDA=torch.cuda.is_available())

                if not write:
                    detections = x
                    write = 1

                else:
                    # Concatenate the YOLO layer outputs
                    detections = torch.cat(tensors=(detections, x), dim=1)
                # print(f"The {index_of_detection}th Output Shape: ", x.shape)

            outputs[index] = x
            # print(x.shape)

        return detections


if __name__ == '__main__':
    # Take One Image as Input have a test
    from dataloader import OpenDataset, DATA_PATH, CSV_PATH, DEVICE

    data = OpenDataset(dataframe=DATA_PATH + CSV_PATH, image_path=DATA_PATH + 'training_images/')
    module = Darknet(CONFIG_PATH).to(DEVICE)
    inp = data[2][0].to(DEVICE)
    pred = module(inp)
    print(f'Shape of Net Output: ', pred.shape)
