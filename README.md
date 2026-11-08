# Deep Learning Project 1: a 4.5M-parameter ResNet for CIFAR-10

Robert-Constantin Bujor (rcb9438) · NYU Tandon · Deep Learning, Fall 2026

Kaggle competition: train a residual network from scratch with at most 5,000,000 trainable
parameters and classify 2,000 hidden test images of a CIFAR-10 variant.

## Result

| | |
|---|---|
| Model | WRN-28-3.5 (pre-activation Wide ResNet: 3 stages × 4 residual blocks, 56/112/224 channels) |
| Trainable parameters | **4,480,474** |
| Training | 500 epochs on all 50,000 training images, AutoAugment + Cutout, label smoothing, EMA of weights, knowledge distillation from a WRN-28-10 teacher (36.5M parameters, used only during training) |
| Kaggle test accuracy | **0.896 public / 0.868 private** |
| Validation accuracy (same recipe, 200 epochs, 5,000 held-out images) | 96.92% |

All runs, plots and the final submission are in **[`results.ipynb`](results.ipynb)** (executed),
which prints the parameter count and the final test accuracy.

| Step (same 4.48M-parameter WRN-28-3.5) | Kaggle public |
|---|---|
| 60 epochs, Cutout | 0.858 |
| 200 epochs, AutoAugment + Cutout | 0.870 |
| 300 epochs, all 50k images | 0.888 |
| + distillation from WRN-28-10 | 0.895 |
| + 500 epochs (final) | **0.896** |

## Files

| File | Purpose |
|---|---|
| `results.ipynb` | Executed notebook: parameter counts, all results, training curves, re-checked accuracies, final predictions |
| `prepare_data.py` | Decodes the Kaggle PNG files once into `data/train.npz` and `data/test.npz` |
| `models.py` | ResNet / Wide ResNet definitions (`python models.py` prints parameter counts) |
| `data.py` | Validation split, augmentation (crop, flip, Cutout, AutoAugment, TrivialAugment, RandAugment), data loaders |
| `train.py` | Training: SGD + cosine schedule, label smoothing, mixup/CutMix, EMA, knowledge distillation, resumable checkpoints, submission writer |
| `predict.py` | Writes a Kaggle submission from a checkpoint without training |
| `utils.py` | Device selection (NVIDIA / Apple GPU / CPU), flip test-time augmentation, checkpoint loading |
| `run_queue.py`, `plans/` | Runs a list of experiments one after another |
| `checkpoints/` | Weights of the final model and of the distillation run with validation (about 18 MB each) |
| `runs/` | Configuration, per-epoch log, metrics and submission of each run trained locally |
| `figures/` | Figures used in the report |
| `kaggle_notebook.ipynb` | The same code packaged as a Kaggle notebook (used for the 60- and 200-epoch comparisons on 2× T4) |

## Reproduce

Python 3.9 or newer:

```bash
pip install -r requirements.txt
# Kaggle data (train_labels.csv, train_images/, test.csv, test_images/) in ../data
python prepare_data.py --src ../data --out data
```

**Submission from the saved final model (no training):**

```bash
python predict.py --checkpoint checkpoints/final_kd_wrn_aa_500.pt --out submission.csv
```

**Re-run the results notebook:**

```bash
python -m nbconvert --to notebook --execute results.ipynb --inplace
```

**Train the final model from scratch** (teacher first, then the student):

```bash
python train.py --name teacher_wrn28_10 --model wrn28_10 --epochs 150 --aug aa+cutout --allow-over-limit
python train.py --name final_kd_wrn_aa_500 --model wrn28_3.5 --epochs 500 --aug aa+cutout \
    --teacher runs/teacher_wrn28_10/model.pt --val-size 0
```

The teacher keeps the default 5,000-image validation split; `--val-size 0` trains the student on all
50,000 images. An interrupted run continues with the same command plus `--resume`.

**Other runs in the report:**

| Run | Command (`python train.py --name <run> ...`) |
|---|---|
| `arch_resnet18_slim` / `arch_resnet14_wide` / `arch_wrn28_3.5` | `--model <model> --epochs 60 --aug cutout` |
| `wrn_aa`, `wrn_ta`, `wrn_ta_mix` | `--model wrn28_3.5 --epochs 60 --aug aa+cutout` (or `ta+cutout`, `ta+cutout --mix both`) |
| `wrn_cutout_200` | `--model wrn28_3.5 --epochs 200 --aug cutout` |
| `wrn_aa_200` | `--model wrn28_3.5 --epochs 200 --aug aa+cutout` |
| `kd_wrn_aa_200` | `--model wrn28_3.5 --epochs 200 --aug aa+cutout --teacher runs/teacher_wrn28_10/model.pt` |
| `final_wrn_aa_full` | `--model wrn28_3.5 --epochs 300 --aug aa+cutout --val-size 0` |
| `final_kd_wrn_aa_full` | `--model wrn28_3.5 --epochs 300 --aug aa+cutout --teacher runs/teacher_wrn28_10/model.pt --val-size 0` |

Common settings (defaults in `train.py`): SGD with Nesterov momentum 0.9, peak learning rate 0.1,
2-epoch warm-up then cosine decay, weight decay 5e-4 (not on BatchNorm/bias), batch 128, label
smoothing 0.1, EMA decay 0.999, distillation α = 0.5 and T = 4, horizontal-flip TTA at test time.

## Hardware

NYU HPC GPU nodes could not be allocated during the project week, so everything was trained on:

| Machine | Used for | Speed (WRN-28-3.5) |
|---|---|---|
| Kaggle notebook, 2× NVIDIA T4 | 60- and 200-epoch comparisons | ~21–30 s/epoch |
| MacBook Pro, Apple M5 Pro (PyTorch MPS) | 200- and 300-epoch runs without distillation | ~40–50 s/epoch |
| Laptop, NVIDIA RTX 4060 Laptop GPU (8 GB) | teacher, all distillation runs | ~33–37 s/epoch with teacher; teacher ~85 s/epoch |

`run_queue.py` runs a plan file from `plans/` on any of these machines and skips runs that already
finished, so a queue can be restarted after a reboot or sleep:

```bash
python run_queue.py plans/ablations.txt --workers 6
```

## References

- K. He, X. Zhang, S. Ren, J. Sun. Deep residual learning for image recognition. CVPR 2016; Identity mappings in deep residual networks. ECCV 2016.
- S. Zagoruyko, N. Komodakis. Wide residual networks. BMVC 2016.
- T. DeVries, G. Taylor. Improved regularization of convolutional neural networks with Cutout. arXiv:1708.04552, 2017.
- E. D. Cubuk et al. AutoAugment: learning augmentation policies from data. CVPR 2019.
- S. G. Müller, F. Hutter. TrivialAugment: tuning-free yet state-of-the-art data augmentation. ICCV 2021.
- G. Hinton, O. Vinyals, J. Dean. Distilling the knowledge in a neural network. arXiv:1503.02531, 2015.
- C. Szegedy et al. Rethinking the Inception architecture for computer vision (label smoothing). CVPR 2016.
- I. Loshchilov, F. Hutter. SGDR: stochastic gradient descent with warm restarts (cosine schedule). ICLR 2017.
- H. Zhang et al. mixup. ICLR 2018; S. Yun et al. CutMix. ICCV 2019.
- B. Recht et al. Do CIFAR-10 classifiers generalize to CIFAR-10? arXiv:1806.00451, 2018.

The code was written for this project (no code was copied from other repositories).
