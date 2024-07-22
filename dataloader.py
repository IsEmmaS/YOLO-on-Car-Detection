import glob

import numpy as np
import pandas as pd
import torch
from PIL import Image

DATA_PATH = "data/"
CSV_PATH = "train_solution_bounding_boxes.csv"

df = pd.read_csv(DATA_PATH + CSV_PATH)
print(df.head())


class OpenDataset(torch.utils.data.Dataset):

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
