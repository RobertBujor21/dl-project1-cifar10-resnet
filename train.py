"""Train a ResNet with at most 5M parameters and write a Kaggle submission.

Everything a run produces goes to runs/<name>/:
    config.json      all arguments
    log.csv          learning rate, losses and accuracies for every epoch
    model.pt         final weights and EMA weights (utils.load_trained_model)
    metrics.json     final numbers
    submission.csv   test predictions in the Kaggle format
and one summary row is added to runs/summary.csv.

A resumable checkpoint (last.pt) is written every --save-every epochs. If a
run stops early, run the same command again with --resume added.

Examples:
    python train.py --name baseline --model resnet18_slim --epochs 30
    python train.py --name final --model wrn28_3.5 --epochs 300 --aug aa+cutout --val-size 0
    python train.py --name final --model wrn28_3.5 --epochs 300 --aug aa+cutout --val-size 0 --resume
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import time

# On Apple silicon, the few operations without an MPS kernel fall back to
# the CPU instead of failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import load_arrays, make_loader, parse_aug, train_transform
from models import CONFIGS, PARAM_LIMIT, build_model, count_params
from utils import (
    accuracy,
    amp_dtype,
    autocast,
    device_name,
    get_device,
    load_trained_model,
    memory_format,
    predict_logprobs,
    write_submission,
)


LOG_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "ema_val_acc",
    "epoch_time",
]


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a CIFAR ResNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Run bookkeeping
    parser.add_argument(
        "--name", required=True, help="run name; outputs go to <out-dir>/<name>"
    )
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader workers (default: $SLURM_CPUS_PER_TASK or 4)",
    )

    # Model
    parser.add_argument("--model", default="resnet18_slim", choices=sorted(CONFIGS))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--allow-over-limit",
        action="store_true",
        help="allow > 5M parameters (only for distillation teachers)",
    )

    # Data
    parser.add_argument(
        "--val-size",
        type=int,
        default=5000,
        help="images held out for validation; 0 = train on all 50k",
    )
    parser.add_argument(
        "--aug", default="cutout", help="'+'-separated: basic, cutout, ta, aa, ra"
    )
    parser.add_argument(
        "--mix", default="none", choices=["none", "mixup", "cutmix", "both"]
    )
    parser.add_argument("--mix-alpha", type=float, default=1.0)
    parser.add_argument(
        "--mix-prob",
        type=float,
        default=0.5,
        help="fraction of batches that get mixed",
    )

    # Optimization
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1, help="peak learning rate")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--wd",
        type=float,
        default=5e-4,
        help="weight decay (not applied to BN/bias)",
    )
    parser.add_argument("--warmup-epochs", type=float, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--ema",
        type=float,
        default=0.999,
        help="EMA decay per step; 0 disables EMA",
    )

    # Knowledge distillation
    parser.add_argument(
        "--teacher",
        default=None,
        help="teacher checkpoint (runs/<teacher>/model.pt)",
    )
    parser.add_argument(
        "--kd-alpha",
        type=float,
        default=0.5,
        help="weight of the distillation loss",
    )
    parser.add_argument(
        "--kd-temp", type=float, default=4.0, help="softmax temperature"
    )

    # Speed and debugging
    parser.add_argument(
        "--no-amp", action="store_true", help="disable mixed precision"
    )
    parser.add_argument("--compile", action="store_true", help="use torch.compile")
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
        help="debug: batches per epoch",
    )

    # Checkpointing
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="save a resumable checkpoint (last.pt) every N epochs; 0 disables",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run from <out-dir>/<name>/last.pt",
    )

    return parser.parse_args(argv)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EMA:
    # Exponential moving average of the weights and the BatchNorm
    # statistics. The averaged model is usually a little more accurate than
    # the raw one.

    def __init__(self, model, decay):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()

        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1

        # A short warm-up, so the random starting weights are forgotten
        # quickly.
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        current = model.state_dict()

        for key, average in self.module.state_dict().items():
            if average.dtype.is_floating_point:
                average.lerp_(current[key].detach(), 1 - decay)
            else:
                average.copy_(current[key])


def mix_batch(images, labels, mode, alpha):
    # Mixup (Zhang et al., 2018) or CutMix (Yun et al., 2019) on one batch.
    # Returns the mixed images, both sets of labels and the weight of the
    # first set.
    if mode == "both":
        mode = "mixup" if random.random() < 0.5 else "cutmix"

    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.size(0), device=images.device)

    if mode == "mixup":
        images = lam * images + (1 - lam) * images[permutation]
    else:
        height, width = images.shape[-2:]
        ratio = math.sqrt(1 - lam)
        cut_height, cut_width = int(height * ratio), int(width * ratio)
        centre_y, centre_x = random.randrange(height), random.randrange(width)

        top = max(centre_y - cut_height // 2, 0)
        bottom = min(centre_y + cut_height // 2, height)
        left = max(centre_x - cut_width // 2, 0)
        right = min(centre_x + cut_width // 2, width)

        images = images.clone()
        images[:, :, top:bottom, left:right] = (
            images[permutation, :, top:bottom, left:right]
        )

        # The patch is clipped at the border, so the label weight follows
        # the area that was actually pasted.
        lam = 1 - (bottom - top) * (right - left) / (height * width)

    return images, labels, labels[permutation], lam


def distillation_loss(student_logits, teacher_logits, temperature):
    # KL divergence between the softened teacher and student distributions
    # (Hinton et al., 2015). Multiplying by T^2 keeps the gradients about as
    # large as those of the cross-entropy term.
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=1)

    divergence = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction="batchmean",
        log_target=True,
    )
    return divergence * temperature * temperature


def make_optimizer(model, lr, momentum, weight_decay):
    # Weight decay is only applied to convolution and linear weights.
    # BatchNorm weights and biases are 1-D, so they are left out.
    decay = []
    no_decay = []

    for parameter in model.parameters():
        if parameter.ndim > 1:
            decay.append(parameter)
        else:
            no_decay.append(parameter)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)


def make_scheduler(optimizer, total_steps, warmup_steps):
    # Linear warm-up, then cosine decay to zero, updated after every step.
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def save_checkpoint(path, epoch, model, ema, optimizer, scheduler, scaler, minutes):
    # Everything needed to continue exactly where the run stopped.
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "ema": ema.module.state_dict() if ema is not None else None,
        "ema_updates": ema.updates if ema is not None else 0,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "minutes": minutes,
    }

    # The checkpoint is written to a temporary file first, so stopping the
    # run in the middle of a save never leaves a broken last.pt.
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_checkpoint(path, device, model, ema, optimizer, scheduler, scaler, log_path):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    if ema is not None and state["ema"] is not None:
        ema.module.load_state_dict(state["ema"])
        ema.updates = state["ema_updates"]

    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["rng_python"])
    np.random.set_state(state["rng_numpy"])
    torch.set_rng_state(state["rng_torch"].cpu())

    # Log rows written after the checkpoint are dropped, because those
    # epochs are trained again.
    with open(log_path, newline="") as file:
        kept_rows = [
            row
            for row in csv.DictReader(file)
            if int(row["epoch"]) <= state["epoch"]
        ]

    with open(log_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(kept_rows)

    return state["epoch"], state["minutes"]


def train_one_epoch(
    args,
    train_model,
    model,
    ema,
    teacher,
    loader,
    optimizer,
    scheduler,
    scaler,
    criterion,
    device,
    dtype,
):
    train_model.train()
    loss_sum = torch.zeros((), device=device)
    correct_sum = torch.zeros((), device=device)
    seen = 0

    for step, (images, labels) in enumerate(loader):
        if args.limit_batches and step >= args.limit_batches:
            break

        images = images.to(device, non_blocking=True).contiguous(
            memory_format=memory_format(device)
        )
        labels = labels.to(device, non_blocking=True)

        # Only a fraction of the batches (--mix-prob) is mixed.
        if args.mix != "none" and random.random() < args.mix_prob:
            images, labels_a, labels_b, lam = mix_batch(
                images, labels, args.mix, args.mix_alpha
            )
        else:
            labels_a, labels_b, lam = labels, labels, 1.0

        with autocast(device, dtype):
            logits = train_model(images)
            loss = (
                lam * criterion(logits, labels_a)
                + (1 - lam) * criterion(logits, labels_b)
            )

            if teacher is not None:
                # The teacher sees the same augmented (and mixed) batch.
                with torch.no_grad():
                    teacher_logits = teacher(images)

                kd_loss = distillation_loss(
                    logits.float(), teacher_logits.float(), args.kd_temp
                )
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if ema is not None:
            ema.update(model)

        predictions = logits.argmax(1)
        loss_sum += loss.detach() * len(labels)
        correct_sum += (
            lam * (predictions == labels_a).sum()
            + (1 - lam) * (predictions == labels_b).sum()
        )
        seen += len(labels)

    return loss_sum / seen, correct_sum / seen


def append_summary(path, row):
    new_file = not os.path.exists(path)

    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))

        if new_file:
            writer.writeheader()

        writer.writerow(row)


def main(args):
    seed_everything(args.seed)
    device = get_device()
    dtype = amp_dtype(device, not args.no_amp)
    torch.backends.cudnn.benchmark = True

    run_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)

    if args.workers is not None:
        workers = args.workers
    else:
        # On a Slurm cluster, use as many workers as the job has CPUs.
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

    # Checking the augmentation name first means a typo fails right away
    # instead of after the data is loaded.
    parse_aug(args.aug)

    # Data
    train_split, val_split, (test_images, test_ids), (mean, std) = load_arrays(
        args.data_dir, args.val_size, args.seed
    )
    train_images, train_labels = train_split
    train_loader = make_loader(
        train_images,
        train_labels,
        train_transform(args.aug, mean, std),
        args.batch_size,
        train=True,
        workers=workers,
    )

    # Model
    overrides = {"dropout": args.dropout} if args.dropout else {}
    model = build_model(args.model, **overrides)
    model = model.to(device).to(memory_format=memory_format(device))
    n_params = count_params(model)

    if n_params > PARAM_LIMIT and not args.allow_over_limit:
        raise SystemExit(
            f"{args.model} has {n_params:,} parameters (> {PARAM_LIMIT:,})."
        )

    train_model = torch.compile(model) if args.compile else model
    ema = EMA(model, args.ema) if args.ema > 0 else None

    teacher = None

    if args.teacher:
        teacher, _ = load_trained_model(args.teacher, device)
        print(f"teacher: {args.teacher} ({count_params(teacher):,} params)")

    gpu = device_name(device)
    val_count = 0 if val_split is None else len(val_split[0])
    print(
        f"model {args.model}: {n_params:,} trainable parameters | device {gpu} | "
        f"amp {dtype} | train {len(train_images)} val {val_count} | "
        f"workers {workers}",
        flush=True,
    )

    with open(os.path.join(run_dir, "config.json"), "w") as file:
        json.dump({**vars(args), "params": n_params, "gpu": gpu}, file, indent=2)

    # Optimization
    steps_per_epoch = args.limit_batches or len(train_loader)
    optimizer = make_optimizer(model, args.lr, args.momentum, args.wd)
    scheduler = make_scheduler(
        optimizer,
        args.epochs * steps_per_epoch,
        int(args.warmup_epochs * steps_per_epoch),
    )
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    log_path = os.path.join(run_dir, "log.csv")
    last_path = os.path.join(run_dir, "last.pt")
    first_epoch = 1
    minutes_before = 0.0

    if args.resume and os.path.exists(last_path):
        saved_epoch, minutes_before = load_checkpoint(
            last_path,
            device,
            model,
            ema,
            optimizer,
            scheduler,
            scaler,
            log_path,
        )
        first_epoch = saved_epoch + 1
        print(f"resuming from epoch {saved_epoch} ({last_path})", flush=True)
    else:
        if args.resume:
            print(
                f"--resume given but {last_path} does not exist: "
                "starting from scratch"
            )

        with open(log_path, "w", newline="") as file:
            csv.writer(file).writerow(LOG_FIELDS)

    start = time.time()

    for epoch in range(first_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            args,
            train_model,
            model,
            ema,
            teacher,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            criterion,
            device,
            dtype,
        )

        row = {
            "epoch": epoch,
            "lr": f"{optimizer.param_groups[0]['lr']:.5f}",
            "train_loss": f"{train_loss.item():.4f}",
            "train_acc": f"{train_acc.item():.4f}",
            "val_loss": "",
            "val_acc": "",
            "ema_val_acc": "",
            "epoch_time": f"{time.time() - epoch_start:.1f}",
        }

        if val_split is not None:
            val_images, val_labels = val_split
            logprobs = predict_logprobs(
                model, val_images, mean, std, device, tta=False, dtype=dtype
            )
            row["val_loss"] = f"{F.nll_loss(logprobs, val_labels).item():.4f}"
            row["val_acc"] = f"{accuracy(logprobs, val_labels):.4f}"

            if ema is not None:
                logprobs = predict_logprobs(
                    ema.module, val_images, mean, std, device, tta=False, dtype=dtype
                )
                row["ema_val_acc"] = f"{accuracy(logprobs, val_labels):.4f}"

        with open(log_path, "a", newline="") as file:
            csv.DictWriter(file, fieldnames=LOG_FIELDS).writerow(row)

        print(
            " | ".join(f"{key} {value}" for key, value in row.items() if value != ""),
            flush=True,
        )

        if args.save_every and epoch % args.save_every == 0 and epoch < args.epochs:
            save_checkpoint(
                last_path,
                epoch,
                model,
                ema,
                optimizer,
                scheduler,
                scaler,
                minutes_before + (time.time() - start) / 60,
            )

    minutes = minutes_before + (time.time() - start) / 60

    # Final evaluation: the raw and EMA weights are compared on the
    # validation images, with and without flip TTA.
    candidates = {"raw": model}

    if ema is not None:
        candidates["ema"] = ema.module

    metrics = {"params": n_params, "minutes": round(minutes, 1)}

    if val_split is not None:
        val_images, val_labels = val_split

        for tag, candidate in candidates.items():
            for tta in (False, True):
                logprobs = predict_logprobs(
                    candidate, val_images, mean, std, device, tta=tta, dtype=dtype
                )
                key = f"{tag}_val_acc_tta" if tta else f"{tag}_val_acc"
                metrics[key] = round(accuracy(logprobs, val_labels), 4)

        selected = max(candidates, key=lambda tag: metrics[f"{tag}_val_acc_tta"])
    else:
        # Without validation images there is nothing to compare, so the EMA
        # weights are used whenever they exist.
        selected = "ema" if ema is not None else "raw"

    metrics["selected"] = selected

    # Test predictions and the Kaggle submission
    test_logprobs = predict_logprobs(
        candidates[selected], test_images, mean, std, device, tta=True, dtype=dtype
    )
    np.save(os.path.join(run_dir, "test_logprobs.npy"), test_logprobs.numpy())
    write_submission(
        os.path.join(run_dir, "submission.csv"),
        test_ids,
        test_logprobs.argmax(1),
    )

    torch.save(
        {
            "model": args.model,
            "model_overrides": overrides,
            "args": vars(args),
            "state_dict": model.state_dict(),
            "ema_state_dict": ema.module.state_dict() if ema is not None else None,
            "selected": selected,
            "metrics": metrics,
        },
        os.path.join(run_dir, "model.pt"),
    )

    with open(os.path.join(run_dir, "metrics.json"), "w") as file:
        json.dump(metrics, file, indent=2)

    # The run is finished and model.pt holds the final weights, so the
    # resumable checkpoint is no longer needed.
    if os.path.exists(last_path):
        os.remove(last_path)

    append_summary(
        os.path.join(args.out_dir, "summary.csv"),
        {
            "name": args.name,
            "model": args.model,
            "params": n_params,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "wd": args.wd,
            "aug": args.aug,
            "mix": args.mix,
            "label_smoothing": args.label_smoothing,
            "ema": args.ema,
            "teacher": args.teacher or "",
            "val_size": args.val_size,
            "val_acc_tta": metrics.get(f"{selected}_val_acc_tta", ""),
            "selected": selected,
            "minutes": metrics["minutes"],
        },
    )

    print(f"\nNumber of trainable parameters: {n_params:,}")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(
        f"submission written to {os.path.join(run_dir, 'submission.csv')} "
        f"(predictions from the {selected} weights, flip TTA)"
    )


if __name__ == "__main__":
    main(get_args())
