"""Write a Kaggle submission from a trained checkpoint, without training.

    python predict.py --checkpoint checkpoints/final_kd_wrn_aa_500.pt --out submission.csv

Prints the number of trainable parameters and writes one predicted label for
every test image, with flip test-time augmentation like the submitted file.
"""
import argparse

import numpy as np

from data import load_arrays
from models import PARAM_LIMIT, count_params
from utils import (
    device_name,
    get_device,
    load_trained_model,
    predict_logprobs,
    write_submission,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/final_kd_wrn_aa_500.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument(
        "--no-tta", action="store_true", help="disable horizontal-flip TTA"
    )
    return parser.parse_args()


def main():
    args = get_args()
    device = get_device()

    _, _, (test_images, test_ids), (mean, std) = load_arrays(
        args.data_dir, val_size=0
    )
    model, checkpoint = load_trained_model(args.checkpoint, device)
    n_params = count_params(model)
    print(
        f"{checkpoint['model']}: {n_params:,} trainable parameters "
        f"(limit {PARAM_LIMIT:,}) | device {device_name(device)}"
    )

    logprobs = predict_logprobs(
        model, test_images, mean, std, device, tta=not args.no_tta
    )
    predictions = logprobs.argmax(1)
    write_submission(args.out, test_ids, predictions)

    class_counts = np.bincount(predictions.numpy(), minlength=10).tolist()
    print(
        f"wrote {args.out}: {len(predictions)} predictions, "
        f"class counts {class_counts}"
    )


if __name__ == "__main__":
    main()
