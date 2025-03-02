"""SpaceNet1 dataset."""

import kornia as K
import torch
from torchgeo.samplers.utils import _to_tuple
import logging
import torch
from torch import Tensor
from .spacenet import SpaceNet1
from .base_dataset import BaseDataset


class SegDataAugmentation(torch.nn.Module):
    def __init__(self, mean, std, size, split="valid"):
        super().__init__()

        self.norm = K.augmentation.Normalize(mean=mean, std=std)

        if split == "train":
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.RandomResizedCrop(_to_tuple(size), scale=(0.8, 1.0)),
                K.augmentation.RandomHorizontalFlip(p=0.5),
                K.augmentation.RandomVerticalFlip(p=0.5),
                data_keys=["input", "mask"],
            )
        else:
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.Resize(size=size, align_corners=True),
                data_keys=["input", "mask"],
            )

    @torch.no_grad()
    def forward(self, x, y):
        x = self.norm(x)
        x_out, y_out = self.transform(x, y)
        return x_out, y_out


class SegGeoBenchTransform(object):
    MEAN = torch.tensor([270.4807120500449, 324.15669565737215, 507.8092169819377, 537.8422401853196, 537.7207406977241, 1209.7003277637814, 1866.045165626201, 2004.8784342115375])
    STD = torch.tensor([410.9821842645062, 466.7364072457086, 534.92315090301, 618.9212492576934, 647.7404624683102, 819.0946643361298, 1225.529915856995, 1294.781212164937])


    def __init__(self, split, size):
        self.transform = SegDataAugmentation(mean=self.MEAN, std=self.STD, size=size, split=split)

    def __call__(self, sample):
        array, mask = self.transform(sample['image'].unsqueeze(0), sample['mask'].unsqueeze(0).unsqueeze(0))
        array, mask = array.squeeze(0), mask.squeeze(0).squeeze(0)


        return array, mask
    

class SpaceNet1Dataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.return_rgb = config.get('return_rgb', False)
        self.num_channels = config.get('num_channels', 4)
        if self.num_channels == 3:
            self.image = '3band'
        elif self.num_channels == 8:
            self.image = '8band'

    
    def create_dataset(self):
        train_transform = SegGeoBenchTransform(split="train", size=self.img_size)
        eval_transform = SegGeoBenchTransform(split="test", size=self.img_size)

        dataset_train = SpaceNet1(
            root=self.root_dir, split="train", transforms=train_transform, image=self.image)
        dataset_val = SpaceNet1(
            root=self.root_dir, split="val", transforms=eval_transform, image=self.image)
        dataset_test = SpaceNet1(
            root=self.root_dir, split="test", transforms=eval_transform, image=self.image)

        return dataset_train, dataset_val, dataset_test