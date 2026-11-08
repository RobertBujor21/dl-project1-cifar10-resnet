"""Decode the competition PNG files once and save them as numpy arrays.

Decoding 50,000 small PNG files in every epoch would make data loading the
bottleneck, so it is done a single time and saved as:

    data/train.npz  images (50000, 32, 32, 3) uint8, labels (50000,) int64,
                    mean and std (3,) float32, per-channel values in [0, 1]
    data/test.npz   images (2000, 32, 32, 3) uint8, ids (2000,) int64

Usage:
    python prepare_data.py --src ../data --out data
"""
import argparse
import csv
import os

import numpy as np
from PIL import Image


def read_split(csv_path, image_dir, has_labels):
    with open(csv_path, newline="") as file:
        rows = list(csv.DictReader(file))

    images = np.empty((len(rows), 32, 32, 3), dtype=np.uint8)

    for index, row in enumerate(rows):
        with Image.open(os.path.join(image_dir, row["filename"])) as image:
            images[index] = np.asarray(image.convert("RGB"))

    ids = np.array([int(row["id"]) for row in rows], dtype=np.int64)

    if has_labels:
        labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    else:
        labels = None

    return images, ids, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", default="../data", help="folder with the Kaggle files"
    )
    parser.add_argument(
        "--out", default="data", help="where to write the .npz files"
    )
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_images, _, train_labels = read_split(
        os.path.join(args.src, "train_labels.csv"),
        os.path.join(args.src, "train_images"),
        True,
    )

    # Per-channel statistics of the training images, used for normalization.
    pixels = train_images.reshape(-1, 3).astype(np.float64) / 255.0
    mean, std = pixels.mean(axis=0), pixels.std(axis=0)

    np.savez(
        os.path.join(args.out, "train.npz"),
        images=train_images,
        labels=train_labels,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )
    print(f"train: {train_images.shape}, class counts {np.bincount(train_labels).tolist()}")
    print(f"mean {mean.round(4).tolist()}  std {std.round(4).tolist()}")

    test_images, test_ids, _ = read_split(
        os.path.join(args.src, "test.csv"),
        os.path.join(args.src, "test_images"),
        False,
    )
    np.savez(os.path.join(args.out, "test.npz"), images=test_images, ids=test_ids)
    print(f"test:  {test_images.shape}")


if __name__ == "__main__":
    main()
