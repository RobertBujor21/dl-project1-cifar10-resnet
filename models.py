"""Residual networks for 32 x 32 CIFAR-10 images.

Two residual blocks are defined:

- BasicBlock is the block from the project handout,
  out = ReLU(S(x) + F(x)) with F = conv -> BN -> ReLU -> conv -> BN.
- PreActBlock is the pre-activation block of He et al. (2016), used by Wide
  ResNets: out = S(x) + conv(ReLU(BN(conv(ReLU(BN(x)))))).

A network is a 3 x 3 stem convolution, a few stages of residual blocks
(every stage after the first halves the resolution), global average pooling
and a linear classifier. Each architecture from the experiments is one entry
in CONFIGS, so an experiment only needs a different --model name.

Run `python models.py` to print the parameter count of each config.
"""
import torch.nn as nn
import torch.nn.functional as F


PARAM_LIMIT = 5_000_000


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        kernel=3,
        skip_kernel=1,
        dropout=0.0,
    ):
        super().__init__()
        padding = kernel // 2

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel, stride, padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel, 1, padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # When the block changes the number of channels or the resolution,
        # the shortcut needs a projection so it can be added to F(x).
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    skip_kernel,
                    stride,
                    skip_kernel // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(self.dropout(out)))
        return F.relu(out + self.skip(x))


class PreActBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        kernel=3,
        skip_kernel=1,
        dropout=0.0,
    ):
        super().__init__()
        padding = kernel // 2

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel, stride, padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel, 1, padding, bias=False
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.projection = stride != 1 or in_channels != out_channels

        if self.projection:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                skip_kernel,
                stride,
                skip_kernel // 2,
                bias=False,
            )

    def forward(self, x):
        pre = F.relu(self.bn1(x))

        # The projection shortcut also starts from the normalized input,
        # while the identity shortcut keeps x unchanged.
        shortcut = self.skip(pre) if self.projection else x

        out = self.conv1(pre)
        out = self.conv2(self.dropout(F.relu(self.bn2(out))))
        return out + shortcut


BLOCKS = {"basic": BasicBlock, "preact": PreActBlock}


class ResNet(nn.Module):
    def __init__(
        self,
        widths,
        depths,
        block="basic",
        stem_width=None,
        kernel=3,
        skip_kernel=1,
        dropout=0.0,
        num_classes=10,
        zero_init_residual=True,
    ):
        super().__init__()

        if len(widths) != len(depths):
            raise ValueError("widths and depths need one entry per stage.")

        block_class = BLOCKS[block]
        self.preact = block == "preact"
        stem_width = stem_width or widths[0]

        self.stem = nn.Conv2d(3, stem_width, 3, 1, 1, bias=False)

        # A post-activation network normalizes right after the stem, while a
        # pre-activation network does it at the start of its first block.
        if self.preact:
            self.stem_bn = nn.Identity()
        else:
            self.stem_bn = nn.BatchNorm2d(stem_width)

        stages = []
        in_channels = stem_width

        for stage_index, (width, depth) in enumerate(zip(widths, depths)):
            blocks = []

            for block_index in range(depth):
                # The first block of every stage after the first one halves
                # the resolution.
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(
                    block_class(
                        in_channels, width, stride, kernel, skip_kernel, dropout
                    )
                )
                in_channels = width

            stages.append(nn.Sequential(*blocks))

        self.stages = nn.Sequential(*stages)

        if self.preact:
            self.final_bn = nn.BatchNorm2d(in_channels)
        else:
            self.final_bn = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
        self.initialize_weights(zero_init_residual)

    def initialize_weights(self, zero_init_residual):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        if zero_init_residual:
            # The last BN of every basic block starts at zero, so each block
            # begins as the identity (Goyal et al., 2017), which makes the
            # first epochs more stable. A pre-activation block ends in a
            # convolution instead of a BN, so it is left as it is.
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.zeros_(module.bn2.weight)

    def forward(self, x):
        x = self.stem(x)

        if not self.preact:
            x = F.relu(self.stem_bn(x))

        x = self.stages(x)

        if self.preact:
            x = F.relu(self.final_bn(x))

        x = self.pool(x).flatten(1)
        return self.fc(x)


CONFIGS = {
    # ResNet-18 layout (4 stages x 2 blocks) with a slimmer last stage.
    "resnet18_slim": {
        "block": "basic",
        "widths": (64, 128, 256, 224),
        "depths": (2, 2, 2, 2),
    },
    # Shallower but wider: 3 stages x 2 blocks.
    "resnet14_wide": {
        "block": "basic",
        "widths": (96, 192, 320),
        "depths": (2, 2, 2),
    },
    # Wide ResNet with depth 28 and widening factor 3.5
    # (Zagoruyko & Komodakis, 2016). This is the submitted model.
    "wrn28_3.5": {
        "block": "preact",
        "widths": (56, 112, 224),
        "depths": (4, 4, 4),
        "stem_width": 16,
    },
    # Teacher for distillation only. It is far above the 5M limit and was
    # never submitted.
    "wrn28_10": {
        "block": "preact",
        "widths": (160, 320, 640),
        "depths": (4, 4, 4),
        "stem_width": 16,
    },
    # The original ResNet-18 widths (11.2M parameters), for reference only.
    "resnet18": {
        "block": "basic",
        "widths": (64, 128, 256, 512),
        "depths": (2, 2, 2, 2),
    },
}


def build_model(name, **overrides):
    if name not in CONFIGS:
        raise ValueError(f"unknown model '{name}', choose from {sorted(CONFIGS)}")

    settings = dict(CONFIGS[name])
    settings.update(overrides)
    return ResNet(**settings)


def count_params(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    for name in CONFIGS:
        n_params = count_params(build_model(name))
        status = "ok" if n_params <= PARAM_LIMIT else "OVER LIMIT"
        print(f"{name:15s} {n_params:>11,d}  {status}")
