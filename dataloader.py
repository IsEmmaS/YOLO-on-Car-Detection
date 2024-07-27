import glob
import cv2
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

# 数据路径定义
DATA_PATH = "data/"
CSV_PATH = "train_solution_bounding_boxes.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 读取CSV文件
df = pd.read_csv(DATA_PATH + CSV_PATH)


class OpenDataset(torch.utils.data.Dataset):
    """
    自定义数据集类，用于加载和处理图像及其边界框数据。
    """

    def __init__(self, dataframe, image_path):
        """
        初始化数据集。

        :param dataframe: 包含图像边界框数据的pandas DataFrame，
                         或指向包含这些数据的CSV文件的路径。
                         应包含'image', 'xmin', 'ymin', 'xmax', 'ymax'列。
        :param image_path: 图像文件的路径前缀。
        """
        super().__init__()
        self.w, self.h = 608, 608  # 固定图像尺寸
        self.data = pd.read_csv(dataframe) if isinstance(dataframe, str) else dataframe
        self.image_path = image_path
        self.files = glob.glob(self.image_path + '*.jpg')  # 获取所有图像文件路径

    def __getitem__(self, index):
        """
        获取数据集中的一个样本。

        :param index: 样本的索引。
        :return: 包含图像和边界框的变量。
        """
        # 获取图像名称和路径
        image_name = self.data.iloc[index]['image']
        image_path = self.image_path + image_name

        # 读取和预处理图像
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        img = img[..., ::-1].transpose((2, 0, 1)) / 255.0  # 从BGR转换到RGB并归一化
        img = torch.from_numpy(img[np.newaxis, :, :, :]).float()

        # 获取边界框数据并归一化
        data = self.data[self.data['image'] == image_name]
        box = data[['xmin', 'ymin', 'xmax', 'ymax']].values
        origin_size = img.shape[1:3]
        box = box * np.array([self.w / origin_size[0], self.h / origin_size[1]] * 2)

        return Variable(img).to(DEVICE), box

    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return len(self.files)


def config_reader(config_file):
    """
    读取配置文件。

    :param config_file: 配置文件的路径。
    :return: 配置块列表。
    """
    file = open(config_file, 'r')
    lines = [line.strip() for line in file.readlines() if line.strip() and line.strip()[0] != '#']
    blocks_info = []
    block = {}
    for line in lines:
        if line.startswith('['):
            if block:
                blocks_info.append(block)
            block = {'type': line[1:-1].strip()}
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
    blocks_info.append(block)
    return blocks_info

# 确保文件路径正确，然后可以实例化数据集并使用
# dataset = OpenDataset(df, DATA_PATH)
