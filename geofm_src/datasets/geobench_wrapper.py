import geobench
import kornia as K
import torch
from torchgeo.samplers.utils import _to_tuple
import logging

logger = logging.getLogger()

class ClsDataAugmentation(torch.nn.Module):
    def __init__(self, mean, std, size, split="valid"):
        super().__init__()

        if split == "train":
            self.transform = torch.nn.Sequential(
                K.augmentation.Normalize(mean=mean, std=std),
                K.augmentation.RandomResizedCrop(
                    size=size, scale=(0.8, 1.0), align_corners=True
                ),  # croma sentinel 2
                K.augmentation.RandomHorizontalFlip(p=0.5),
                K.augmentation.RandomVerticalFlip(p=0.5),
            )
        else:
            self.transform = torch.nn.Sequential(
                K.augmentation.Normalize(mean=mean, std=std),
                K.augmentation.Resize(
                    size=size, align_corners=True
                ),  # croma sentinel 2
                
            )

    @torch.no_grad()
    def forward(self, x):
        x_out = self.transform(x)
        return x_out


class ClsGeoBenchTransform(object):
    def __init__(self, task, split, size, band_names=None, num_channels=None):
        self.band_names = band_names
        MEAN, STD = task.get_dataset(band_names=band_names).normalization_stats()
        self.transform = ClsDataAugmentation(mean=MEAN, std=STD, split=split, size=size)
        self.num_channels = num_channels

    def __call__(self, sample):
        array, band_names = sample.pack_to_3d(
            band_names=self.band_names,
            resample=False,
            fill_value=None,
            resample_order=3,
        )  # h,w,c
        array = torch.from_numpy(array.astype("float32")).permute(2, 0, 1)
        array = self.transform(array).squeeze(0)

                # HACKY: Will only work for models that dont need chn_ids
        # Pad with zeros if num_channels is greater than the actual channels
        if hasattr(self, 'num_channels') and self.num_channels > array.shape[0]: 
            padding_channels = self.num_channels - array.shape[0]
            padding = torch.zeros((padding_channels, *array.shape[1:]), device=array.device, dtype=array.dtype)
            array = torch.cat([array, padding], dim=0)

        return array, torch.tensor(sample.label)  # , band_names


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
                K.augmentation.CenterCrop(size=size, align_corners=True),
                data_keys=["input", "mask"],
            )

    @torch.no_grad()
    def forward(self, x, y):
        x = self.norm(x)
        x_out, y_out = self.transform(x, y)
        return x_out, y_out


class SegGeoBenchTransform(object):
    def __init__(self, task, split, size, band_names=None, num_channels=None):
        self.band_names = band_names
        MEAN, STD = task.get_dataset(band_names=band_names).normalization_stats()
        if task.patch_size[0] < size[0]:
            size = task.patch_size

        self.transform = SegDataAugmentation(mean=MEAN, std=STD, size=size, split=split)
        self.num_channels = num_channels

    def __call__(self, sample):
        array, band_names = sample.pack_to_3d(
            band_names=self.band_names, resample=True, fill_value=None, resample_order=3
        )  # h,w,c
        array = torch.from_numpy(array.astype("float32")).permute(2, 0, 1)
        mask = torch.from_numpy(sample.label.data.astype("float32")).squeeze(-1)
        array, mask = self.transform(array.unsqueeze(0), mask.unsqueeze(0).unsqueeze(0))
        array, mask = array.squeeze(0), mask.squeeze(0).squeeze(0)
        # HACKY: Will only work for models that dont need chn_ids
        #pad with zeros if num_channels is greater than the actual channels
        if hasattr(self, 'num_channels') and self.num_channels > array.shape[0]: 
            padding_channels = self.num_channels - array.shape[0]
            padding = torch.zeros((padding_channels, *array.shape[1:]), device=array.device, dtype=array.dtype)
            array = torch.cat([array, padding], dim=0)

        return array, mask


class GeoBenchDataset:
    def __init__(self, config):
        self.dataset_config = config
        task_iter = geobench.task_iterator(benchmark_name=config.benchmark_name)
        self.tasks = {task.dataset_name: task for task in task_iter}
        self.img_size = (config.image_resolution, config.image_resolution)
        self.num_channels = config.num_channels

    def create_dataset(self):
        task = self.tasks.get(self.dataset_config.dataset_name)
        assert task is not None, (
            f"{self.dataset_config.dataset_name} doesn't exist in geobench"
        )
        GeoBenchTransform = (
            ClsGeoBenchTransform
            if self.dataset_config.task == "classification"
            else SegGeoBenchTransform
        )

        train_transform = GeoBenchTransform(
            task,
            split="train",
            size=self.img_size,
            band_names=self.dataset_config.band_names,
            num_channels=self.num_channels,  # Added num_channels here
        )
        val_transform = GeoBenchTransform(
            task,
            split="valid",
            size=self.img_size,
            band_names=self.dataset_config.band_names,
            num_channels=self.num_channels,  # Added num_channels here
        )

        dataset_train = task.get_dataset(
            split="train",
            transform=train_transform,
            band_names=self.dataset_config.band_names,
        )
        dataset_val = task.get_dataset(
            split="valid",
            transform=val_transform,
            band_names=self.dataset_config.band_names,
        )
        dataset_test = task.get_dataset(
            split="test",
            transform=val_transform,
            band_names=self.dataset_config.band_names,
        )

        return dataset_train, dataset_val, dataset_test
