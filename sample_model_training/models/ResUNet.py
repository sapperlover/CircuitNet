import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from models.FCN import build_output_activation, generation_init_weights, load_state_dict


def build_norm(norm_type, channels):
    norm_type = norm_type.lower()
    if norm_type in ("instance", "in"):
        return nn.InstanceNorm2d(channels, affine=True)
    if norm_type in ("batch", "bn"):
        return nn.BatchNorm2d(channels)
    if norm_type in ("group", "gn"):
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm_type in ("none", "identity"):
        return nn.Identity()
    raise ValueError("Unsupported norm_type: {}".format(norm_type))


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type="instance", negative_slope=0.2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            build_norm(norm_type, out_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
            build_norm(norm_type, out_channels),
        )
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True),
                build_norm(norm_type, out_channels),
            )
        self.activation = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x):
        return self.activation(self.main(x) + self.shortcut(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type="instance", negative_slope=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            build_norm(norm_type, out_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    def forward(self, x, target):
        x = F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=2,
        base_channels=32,
        out_activation="sigmoid",
        norm_type="instance",
        negative_slope=0.2,
        **kwargs
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.e1 = ResConvBlock(in_channels, c1, norm_type, negative_slope)
        self.down1 = nn.MaxPool2d(2)
        self.e2 = ResConvBlock(c1, c2, norm_type, negative_slope)
        self.down2 = nn.MaxPool2d(2)
        self.e3 = ResConvBlock(c2, c3, norm_type, negative_slope)
        self.down3 = nn.MaxPool2d(2)
        self.bottleneck = ResConvBlock(c3, c4, norm_type, negative_slope)

        self.up3 = UpBlock(c4, c3, norm_type, negative_slope)
        self.d3 = ResConvBlock(c3 + c3, c3, norm_type, negative_slope)
        self.up2 = UpBlock(c3, c2, norm_type, negative_slope)
        self.d2 = ResConvBlock(c2 + c2, c2, norm_type, negative_slope)
        self.up1 = UpBlock(c2, c1, norm_type, negative_slope)
        self.d1 = ResConvBlock(c1 + c1, c1, norm_type, negative_slope)

        self.head = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, 3, 1, 1, bias=True),
            build_norm(norm_type, c1 // 2),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(c1 // 2, out_channels, 1, 1, 0, bias=True),
            build_output_activation(out_activation),
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.down1(e1))
        e3 = self.e3(self.down2(e2))
        x = self.bottleneck(self.down3(e3))

        x = self.up3(x, e3)
        x = self.d3(torch.cat([x, e3], dim=1))
        x = self.up2(x, e2)
        x = self.d2(torch.cat([x, e2], dim=1))
        x = self.up1(x, e1)
        x = self.d1(torch.cat([x, e1], dim=1))
        return self.head(x)

    def init_weights(self, pretrained=None, pretrained_transfer=None, strict=False, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location="cpu")["state_dict"]
            for key in weight.keys():
                new_dict[key] = weight[key]
            load_state_dict(self, new_dict, strict=strict, logger=None)
            print("Load state dict form {}".format(pretrained))
        elif pretrained is None:
            generation_init_weights(self)
        else:
            raise TypeError(
                "'pretrained' must be a str or None. "
                "But received {}".format(type(pretrained))
            )

