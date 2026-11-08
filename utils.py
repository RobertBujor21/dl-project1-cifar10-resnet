"""Helpers shared by train.py, predict.py and the notebooks."""
import contextlib
import csv

import torch
import torch.nn.functional as F

from models import build_model


def get_device():
    # An NVIDIA GPU if there is one, then the Apple-silicon GPU (MPS),
    # otherwise the CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name()

    if device.type == "mps":
        return "Apple GPU (MPS)"

    return "cpu"


def memory_format(device):
    # channels_last speeds up convolutions on NVIDIA tensor cores. Other
    # devices keep the default layout.
    if device.type == "cuda":
        return torch.channels_last

    return torch.contiguous_format


def amp_dtype(device, enabled=True):
    # Mixed precision is only used on NVIDIA GPUs: bfloat16 on Ampere or
    # newer (A100, L4, RTX 30/40), float16 on older cards (V100, T4).
    # None means full float32.
    if not enabled or device.type != "cuda":
        return None

    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def autocast(device, dtype):
    if dtype is None:
        return contextlib.nullcontext()

    return torch.autocast(device.type, dtype=dtype)


@torch.no_grad()
def predict_logprobs(
    model,
    images,
    mean,
    std,
    device,
    tta=True,
    batch_size=1000,
    dtype=None,
):
    # images is a uint8 tensor of shape (N, 3, 32, 32). With tta=True the
    # class probabilities of each image and of its mirror image are averaged
    # (horizontal-flip test-time augmentation).
    model.eval()
    mean_tensor = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, 3, 1, 1)
    chunks = []

    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device).float().div_(255)
        batch = ((batch - mean_tensor) / std_tensor).contiguous(
            memory_format=memory_format(device)
        )

        with autocast(device, dtype):
            probabilities = F.softmax(model(batch).float(), dim=1)

            if tta:
                flipped = torch.flip(batch, dims=[3])
                flipped_probabilities = F.softmax(model(flipped).float(), dim=1)
                probabilities = (probabilities + flipped_probabilities) / 2

        chunks.append(probabilities.clamp_min(1e-12).log().cpu())

    return torch.cat(chunks)


def accuracy(logprobs, labels):
    return (logprobs.argmax(1) == labels).float().mean().item()


def write_submission(path, ids, predictions):
    # Kaggle format: an "id,label" header and one row per test image,
    # sorted by id.
    rows = sorted(zip(ids.tolist(), predictions.tolist()))

    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "label"])
        writer.writerows(rows)


def load_trained_model(path, device, which=None):
    # Rebuilds a model saved by train.py. which is "ema", "raw" or None,
    # where None loads the weights train.py selected for the submission.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(
        checkpoint["model"],
        **checkpoint.get("model_overrides", {}),
    )

    which = which or checkpoint.get("selected", "raw")

    if which == "ema":
        model.load_state_dict(checkpoint["ema_state_dict"])
    else:
        model.load_state_dict(checkpoint["state_dict"])

    model.to(device).to(memory_format=memory_format(device)).eval()
    return model, checkpoint
