import re
import torch
import numpy as np
import cv2
import csv
from pathlib import Path
from torch._six import container_abcs, string_classes, int_classes

from src.data.datasets import BaseDataset
from src.data.transforms import Compose
import torchvision.transforms as torchTransform


np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def bgr2rgb(img_bgr):
    return np.concatenate((img_bgr[:, :, 2:], img_bgr[:, :, 1:2], img_bgr[:, :, 0:1]), axis=2) 


class TestDataset(BaseDataset):
    """The dataset of the Automated Cardiac Diagnosis Challenge (ACDC) in MICCAI 2017
    for the segmentation task.
    Ref: 
        https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html
    Args:
        data_split_file_path (str): The data split file path.
        transforms (BoxList): The preprocessing techniques applied to the data.
        augments (BoxList): The augmentation techniques applied to the training data (default: None).
    """

    def __init__(self, data_dir, test_data_csv, resize=False, augments=None, **kwargs):
        super().__init__(**kwargs)
        self.data_paths = []
        self.data_dir = data_dir
        with open(test_data_csv, "r") as f:
            rows = csv.reader(f)
            for i, row in enumerate(rows):
                if i == 0:
                    continue
                file_name, _ = row
            
                self.data_paths.append(file_name)

        self.transform = torchTransform.Compose([
            # transforms.RandomRotation(15),
            torchTransform.ToTensor(),
            torchTransform.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])

        self.resize = resize

    def __getitem__(self, index):
        file_name = self.data_paths[index]
        img_path = str(Path(self.data_dir) / Path(file_name))
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise Exception(f'The path of file is invalid! [{img_path}]')
        if self.resize:
            img_bgr = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_CUBIC)
        img = bgr2rgb(img_bgr)

        img = self.transform(img)

        metadata = {'input': img, 'file_name': file_name}

        # np.save('/home/tony/Desktop/image_normalize.npy', img.numpy())
        # np.save('/home/tony/Desktop/gt.npy', gt.numpy())

        return metadata

    def __len__(self):
        return len(self.data_paths)
