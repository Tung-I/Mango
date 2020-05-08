import re
import numpy as np
import csv
import torch
from pathlib import Path

from src.data.datasets import BaseDataset
# from src.data.transforms import Compose, ToTensor
import torchvision.transforms as torchTransform


class CIFARDataset(BaseDataset):
    def __init__(self, data_dir, csv_name, transforms, augments=None, **kwargs):
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.csv_name = csv_name

        if self.type != 'Testing':
            self.data_dir = self.data_dir / Path('train')
        else:
            self.data_dir = self.data_dir / Path('test')
            c
        self.class_folder_path = sorted([_dir for _dir in self.data_dir.iterdir() if _dir.is_dir()])
        self.data_paths = []

        for label, folder_path in enumerate(self.class_folder_path):
            if self.type != 'Testing':
                csv_path = str(folder_path / csv_name)
                with open(csv_path, 'r', newline='') as csvfile:
                    rows = csv.reader(csvfile)
                    for _path, _type in rows:
                        if self.type == 'train' and _type == 'Training':
                            self.data_paths.append((_path, label))
                        if self.type == 'valid' and _type == 'Validation':
                            self.data_paths.append((_path, label))
            else:
                file_paths = list(folder_path.glob('*.npy'))
                for _path in file_paths:
                    self.data_paths.append((_path, label))

        self.transform_train = torchTransform.Compose([
            # transforms.RandomRotation(15),
            torchTransform.ToTensor(),
            torchTransform.Normalize(mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                         std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
        self.transform_valid = torchTransform.Compose([
            # transforms.RandomRotation(15),
            torchTransform.ToTensor(),
            torchTransform.Normalize(mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                         std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])

    def __getitem__(self, index):
        data_path, gt = self.data_paths[index]
        img = np.load(data_path).astype(np.float32)
        gt = torch.as_tensor([gt]).long()

        if self.type == 'train':
            # transforms_kwargs = {}
            # img = self.transforms(img, **transforms_kwargs)
            # img = self.augments(img)
            # img = self.to_tensor(img)
            img = self.transform_train(img)
        else:
            # img = self.transforms(img, **transforms_kwargs)
            # img = self.to_tensor(img)
            img = self.transform_valid(img)

        # img = img.permute(2, 0, 1).contiguous()
        metadata = {'inputs': img, 'targets': gt}

        return metadata

    def __len__(self):
        return len(self.data_paths)
