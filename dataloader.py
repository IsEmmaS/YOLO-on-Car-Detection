import glob

import numpy as np
import pandas as pd
import torch
from PIL import Image

DATA_PATH = "data/"
CSV_PATH = "train_solution_bounding_boxes.csv"
CONFIG_PATH = "yolov3.cfg"

df = pd.read_csv(DATA_PATH + CSV_PATH)


class OpenDataset(torch.utils.data.Dataset):
    """
    Open dataset class.
    """
    def __init__(self, dataframe, data_path):
        """
        :param dataframe: dataframe is a pandas dataframe with images' box data
        :param data_path: path to the images
        """
        super().__init__()
        self.w, self.h = 224, 224

        self.data = dataframe
        self.datapath = data_path
        self.files = glob.glob(self.datapath + '*.jpg')

    def __getitem__(self, index):
        image_name = self.data.iloc[index]['image']
        image_path = self.datapath + image_name

        print(image_name, image_path)

        img = Image.open(image_path).convert('RGB').resize(size=(self.w, self.h), resample=Image.Resampling.BICUBIC)
        img = np.asarray(img) / 255

        data = self.data[self.data['image'] == image_name]
        box = data[['xmin', 'ymin', 'xmax', 'ymax']].values

        return img, box

    def __len__(self):
        return len(self.files)


def config_reader(config_file):
    """
    Read config file.
    :param config_file: path to config file
    :return: blocks_info: list of blocks_info
    """

    # Delete any comments and empty lines from the config file
    # Split the config file into blocks_info
    # Each block is a dictionary with keys and values
    file = open(config_file, 'r')
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if len(line) > 0 and line[0] != '#']

    blocks_info = []
    block = {}
    for idx, line in enumerate(lines):
        if line[0] == '[':
            if not len(block) == 0:
                blocks_info.append(block)
                block = {}
            block['type'] = line[1:-1].strip()
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
    blocks_info.append(block)

    return blocks_info
