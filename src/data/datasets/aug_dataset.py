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


class AugDataset(BaseDataset):
    """The dataset of the Automated Cardiac Diagnosis Challenge (ACDC) in MICCAI 2017
    for the segmentation task.
    Ref: 
        https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html
    Args:
        data_split_file_path (str): The data split file path.
        transforms (BoxList): The preprocessing techniques applied to the data.
        augments (BoxList): The augmentation techniques applied to the training data (default: None).
    """

    def __init__(self, data_dir, train_data_csv, valid_data_csv, transforms=None, resize=None, augments=None, **kwargs):
        super().__init__(**kwargs)
        self.data_paths = []
        class_type = {}
        class_type['A'] = 0
        class_type['B'] = 1
        class_type['C'] = 2
        if self.type == 'train':
            csv_path = train_data_csv
        elif self.type == 'valid':
            csv_path = valid_data_csv
        else:
            raise Exception('The type of dataset is undefined!')
    
        with open(csv_path, "r") as f:
            rows = csv.reader(f)
            for i, row in enumerate(rows):
                if i == 0:
                    continue
                num_img, label = row
                img_path = str(Path(data_dir) / Path(num_img))
            
                self.data_paths.append((img_path, class_type[label]))

        # self.transforms = Compose.compose(transforms)
        # self.augments = Compose.compose(augments)
        # self.to_tensor = ToTensor()

        self.transform_train = torchTransform.Compose([
            # transforms.RandomRotation(15),
            torchTransform.ToTensor(),
            torchTransform.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
        self.transform_valid = torchTransform.Compose([
            # transforms.RandomRotation(15),
            torchTransform.ToTensor(),
            torchTransform.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
        
        self.augments = Compose.compose(augments)

    def __getitem__(self, index):
        img_path, gt = self.data_paths[index]
        img_bgr = cv2.imread(img_path)
        img = bgr2rgb(img_bgr)
        gt = torch.as_tensor([gt]).long()

        if self.type == 'train':
            # transforms_kwargs = {}
            # img = self.transforms(img, **transforms_kwargs)
            # img = self.augments(img)
            # img = self.to_tensor(img)
            img = self.augments(img)[0]
            img = self.transform_train(img)
        else:
            # img = self.transforms(img, **transforms_kwargs)
            # img = self.to_tensor(img)
            img = self.transform_valid(img)

        # img = img.permute(2, 0, 1).contiguous()
        metadata = {'inputs': img, 'targets': gt}

        # np.save('/home/tony/Desktop/image_normalize.npy', img.numpy())
        # np.save('/home/tony/Desktop/gt.npy', gt.numpy())

        return metadata

    def __len__(self):
        return len(self.data_paths)
