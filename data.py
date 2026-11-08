"""Loading the CIFAR-10 arrays and building the training augmentations.

All images stay in memory as one uint8 tensor of shape (N, 3, 32, 32).
Per-image augmentation runs inside the DataLoader workers, while mixup and
CutMix work on whole batches on the GPU and live in train.py.
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


AUG_CHOICES = {"cutout", "ta", "aa", "ra"}
NUM_CLASSES = 10


def load_arrays(data_dir="data", val_size=5000, seed=0):
    # Returns the training split, the validation split (None when
    # val_size is 0), the test images with their ids, and the per-channel
    # mean and std used for normalization.
    train = np.load(os.path.join(data_dir, "train.npz"))
    test = np.load(os.path.join(data_dir, "test.npz"))

    images = torch.from_numpy(train["images"]).permute(0, 3, 1, 2).contiguous()
    labels = torch.from_numpy(train["labels"])
    stats = (tuple(train["mean"].tolist()), tuple(train["std"].tolist()))

    if val_size > 0:
        # The same number of images is held out from every class. The split
        # only depends on the seed, so runs with the same seed share it.
        rng = np.random.default_rng(seed)
        per_class = val_size // NUM_CLASSES
        val_index = np.concatenate([
            rng.choice(
                np.flatnonzero(train["labels"] == label),
                per_class,
                replace=False,
            )
            for label in range(NUM_CLASSES)
        ])

        keep = np.ones(len(labels), dtype=bool)
        keep[val_index] = False
        train_index = np.flatnonzero(keep)

        train_split = (images[train_index], labels[train_index])
        val_split = (images[val_index], labels[val_index])
    else:
        train_split = (images, labels)
        val_split = None

    test_images = torch.from_numpy(test["images"]).permute(0, 3, 1, 2).contiguous()
    test_ids = torch.from_numpy(test["ids"])

    return train_split, val_split, (test_images, test_ids), stats


class Cutout:
    # Cutout (DeVries & Taylor, 2017) sets one size x size square at a random
    # centre to zero. The square can hang over the border, so the erased area
    # varies. It runs after normalization, where 0 is the dataset mean.

    def __init__(self, size=16):
        self.size = size

    def __call__(self, image):
        _, height, width = image.shape
        centre_y = int(torch.randint(height, (1,)))
        centre_x = int(torch.randint(width, (1,)))
        half = self.size // 2

        top, bottom = max(centre_y - half, 0), min(centre_y + half, height)
        left, right = max(centre_x - half, 0), min(centre_x + half, width)

        image = image.clone()
        image[:, top:bottom, left:right] = 0.0
        return image


def parse_aug(aug):
    if not aug:
        return set()

    # Crop and flip are always used, so "basic" adds nothing to the set.
    parts = {part for part in aug.split("+") if part and part != "basic"}
    unknown = parts - AUG_CHOICES

    if unknown:
        raise ValueError(
            f"unknown augmentation(s) {unknown}; choose from {AUG_CHOICES}"
        )

    return parts


def train_transform(aug, mean, std, cutout_size=16):
    # aug is a "+"-separated list such as "basic", "cutout" or "aa+cutout":
    #   cutout  Cutout with a 16 x 16 square
    #   ta      TrivialAugmentWide (Mueller & Hutter, 2021)
    #   aa      AutoAugment with the CIFAR-10 policy (Cubuk et al., 2019)
    #   ra      RandAugment (Cubuk et al., 2020)
    # A random crop with 4-pixel reflect padding and a horizontal flip are
    # always applied first.
    parts = parse_aug(aug)
    steps = [
        v2.RandomCrop(32, padding=4, padding_mode="reflect"),
        v2.RandomHorizontalFlip(),
    ]

    if "ta" in parts:
        steps.append(v2.TrivialAugmentWide())

    if "aa" in parts:
        steps.append(v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10))

    if "ra" in parts:
        steps.append(v2.RandAugment())

    steps.append(v2.ToDtype(torch.float32, scale=True))
    steps.append(v2.Normalize(mean, std))

    if "cutout" in parts:
        steps.append(Cutout(cutout_size))

    return v2.Compose(steps)


class ImageTensorDataset(Dataset):
    def __init__(self, images, targets, transform):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.targets[index]


def make_loader(images, targets, transform, batch_size, train, workers):
    dataset = ImageTensorDataset(images, targets, transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
