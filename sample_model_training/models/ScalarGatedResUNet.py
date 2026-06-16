import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from models.FCN import build_output_activation, generation_init_weights, load_state_dict
from models.ResUNet import build_norm


def as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def parse_name_set(value):
    if value is None:
        return set()
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in {"", "0", "false", "none", "no", "off"}:
            return set()
        return {item.strip().lower() for item in value.split(",") if item.strip()}
    return {str(item).strip().lower() for item in value if str(item).strip()}


def parse_int_tuple(value):
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(int(item) for item in value)


class ScalarEncoder(nn.Module):
    def __init__(self, scalar_channels=4, hidden_channels=32, embedding_channels=64, negative_slope=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(scalar_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Linear(hidden_channels, embedding_channels),
            nn.LayerNorm(embedding_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    def forward(self, scalar):
        return self.net(scalar)


class MultiScaleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type="instance", negative_slope=0.2):
        super().__init__()
        branch_channels = max(out_channels // 2, 8)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 3, 1, 1, bias=True),
            build_norm(norm_type, branch_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, 1, 1, bias=True),
            build_norm(norm_type, branch_channels),
        )
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 3, 1, 1, bias=True),
            build_norm(norm_type, branch_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 7, 1, 3, groups=branch_channels, bias=True),
            build_norm(norm_type, branch_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 1, 1, 0, bias=True),
            build_norm(norm_type, branch_channels),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * 2, out_channels, 1, 1, 0, bias=True),
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
        multi_scale = torch.cat([self.branch3(x), self.branch7(x)], dim=1)
        return self.activation(self.fuse(multi_scale) + self.shortcut(x))


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


class DualInputStem(nn.Module):
    def __init__(self, relative_channels, physics_channels, out_channels, norm_type="instance", negative_slope=0.2):
        super().__init__()
        relative_out_channels = out_channels // 2
        physics_out_channels = out_channels - relative_out_channels
        self.relative_channels = int(relative_channels)
        self.physics_channels = int(physics_channels)
        self.relative_stem = nn.Sequential(
            nn.Conv2d(self.relative_channels, relative_out_channels, 3, 1, 1, bias=True),
            build_norm(norm_type, relative_out_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
        )
        self.physics_stem = nn.Sequential(
            nn.Conv2d(self.physics_channels, physics_out_channels, 3, 1, 1, bias=True),
            build_norm(norm_type, physics_out_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    def forward(self, x):
        expected_channels = self.relative_channels + self.physics_channels
        if x.size(1) != expected_channels:
            raise ValueError(
                "DualInputStem expects {} input channels, got {}".format(expected_channels, x.size(1))
            )
        relative = x[:, :self.relative_channels]
        physics = x[:, self.relative_channels:expected_channels]
        return torch.cat([self.relative_stem(relative), self.physics_stem(physics)], dim=1)


class LightASPP(nn.Module):
    def __init__(
        self,
        channels,
        branch_channels=None,
        dilations=(1, 2, 4, 8),
        residual_scale=0.2,
        norm_type="instance",
        negative_slope=0.2,
    ):
        super().__init__()
        branch_channels = branch_channels or max(channels // 4, 32)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation, groups=channels, bias=True),
                build_norm(norm_type, channels),
                nn.LeakyReLU(negative_slope, inplace=True),
                nn.Conv2d(channels, branch_channels, 1, 1, 0, bias=True),
                build_norm(norm_type, branch_channels),
                nn.LeakyReLU(negative_slope, inplace=True),
            )
            for dilation in dilations
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * len(dilations), channels, 1, 1, 0, bias=True),
            build_norm(norm_type, channels),
            nn.LeakyReLU(negative_slope, inplace=True),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x):
        context = torch.cat([branch(x) for branch in self.branches], dim=1)
        return x + self.residual_scale * self.fuse(context)


class ScalarFiLM(nn.Module):
    def __init__(self, channels, scalar_channels, strength=0.1):
        super().__init__()
        self.strength = float(strength)
        self.affine = nn.Linear(scalar_channels, channels * 2)
        self.zero_init()

    def zero_init(self):
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x, scalar_embedding):
        scale_shift = self.affine(scalar_embedding).view(x.size(0), 2, x.size(1), 1, 1)
        scale = scale_shift[:, 0]
        shift = scale_shift[:, 1]
        return x * (1.0 + self.strength * scale) + self.strength * shift


class ScalarAttentionSkipGate(nn.Module):
    def __init__(self, skip_channels, decoder_channels, scalar_channels, norm_type="instance", negative_slope=0.2):
        super().__init__()
        inter_channels = max(skip_channels // 2, 8)
        self.skip_proj = nn.Conv2d(skip_channels, inter_channels, 1, 1, 0, bias=False)
        self.decoder_proj = nn.Conv2d(decoder_channels, inter_channels, 1, 1, 0, bias=False)
        self.spatial_norm = build_norm(norm_type, inter_channels)
        self.scalar_proj = nn.Linear(scalar_channels, inter_channels)
        self.activation = nn.LeakyReLU(negative_slope, inplace=True)
        self.gate_logits = nn.Conv2d(inter_channels, 1, 1, 1, 0, bias=True)
        self.zero_init_gate()

    def zero_init_gate(self):
        nn.init.zeros_(self.gate_logits.weight)
        nn.init.zeros_(self.gate_logits.bias)

    def forward(self, skip, decoder_feature, scalar_embedding):
        if decoder_feature.shape[-2:] != skip.shape[-2:]:
            decoder_feature = F.interpolate(decoder_feature, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        spatial = self.skip_proj(skip) + self.decoder_proj(decoder_feature)
        spatial = self.spatial_norm(spatial)
        scalar = self.scalar_proj(scalar_embedding).view(scalar_embedding.size(0), -1, 1, 1)
        logits = self.gate_logits(self.activation(spatial + scalar))
        gate = 2.0 * torch.sigmoid(logits)
        return skip * gate


class ScaleHead(nn.Module):
    def __init__(self, bottleneck_channels, scalar_channels, hidden_channels=64, out_channels=2, negative_slope=0.2):
        super().__init__()
        self.output = nn.Linear(hidden_channels, out_channels)
        self.net = nn.Sequential(
            nn.Linear(bottleneck_channels + scalar_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
            self.output,
        )
        self.zero_init_output()

    def zero_init_output(self):
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, bottleneck, scalar_embedding):
        pooled = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)
        return self.net(torch.cat([pooled, scalar_embedding], dim=1)).view(bottleneck.size(0), -1, 1, 1)


class FusionGateHead(nn.Module):
    def __init__(
        self,
        bottleneck_channels,
        scalar_channels,
        hidden_channels=64,
        out_channels=2,
        gate_min=0.0,
        gate_max=0.2,
        gate_init=0.02,
        negative_slope=0.2,
    ):
        super().__init__()
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.gate_init = float(gate_init)
        if self.gate_max < self.gate_min:
            raise ValueError("gate_max must be >= gate_min")
        self.output = nn.Linear(hidden_channels, out_channels)
        self.net = nn.Sequential(
            nn.Linear(bottleneck_channels + scalar_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
            self.output,
        )
        self.reset_gate()

    def reset_gate(self):
        nn.init.zeros_(self.output.weight)
        if self.gate_max <= self.gate_min:
            nn.init.zeros_(self.output.bias)
            return
        ratio = (self.gate_init - self.gate_min) / (self.gate_max - self.gate_min)
        ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.output.bias, math.log(ratio / (1.0 - ratio)))

    def forward(self, bottleneck, scalar_embedding):
        if self.gate_max <= self.gate_min:
            return bottleneck.new_full((bottleneck.size(0), self.output.out_features, 1, 1), self.gate_min)
        pooled = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)
        logits = self.net(torch.cat([pooled, scalar_embedding], dim=1)).view(bottleneck.size(0), -1, 1, 1)
        return self.gate_min + (self.gate_max - self.gate_min) * torch.sigmoid(logits)


class ScalarGatedResUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=2,
        base_channels=32,
        out_activation="softplus",
        norm_type="instance",
        negative_slope=0.2,
        scalar_channels=4,
        scalar_hidden_channels=32,
        scalar_embedding_channels=64,
        scale_log_clamp=1.5,
        require_scalar=True,
        use_scalar_film=True,
        scalar_film_layers=("e3", "b", "d3"),
        scalar_film_strength=0.1,
        use_aspp=True,
        aspp_dilations=(1, 2, 4, 8),
        aspp_branch_channels=None,
        aspp_res_scale=0.2,
        use_dual_stem=False,
        relative_in_channels=3,
        physics_in_channels=7,
        use_dual_head=False,
        dual_head_gate_min=0.0,
        dual_head_gate_max=0.2,
        dual_head_gate_init=0.02,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.scalar_channels = scalar_channels
        self.scale_log_clamp = scale_log_clamp
        self.require_scalar = require_scalar
        self.use_scalar_film = as_bool(use_scalar_film)
        self.scalar_film_layers = parse_name_set(scalar_film_layers)
        self.use_aspp = as_bool(use_aspp)
        self.use_dual_stem = as_bool(use_dual_stem)
        self.use_dual_head = as_bool(use_dual_head)

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        aspp_dilations = parse_int_tuple(aspp_dilations)

        self.scalar_encoder = ScalarEncoder(
            scalar_channels=scalar_channels,
            hidden_channels=scalar_hidden_channels,
            embedding_channels=scalar_embedding_channels,
            negative_slope=negative_slope,
        )

        if self.use_dual_stem:
            self.input_stem = DualInputStem(
                relative_in_channels,
                physics_in_channels,
                c1,
                norm_type=norm_type,
                negative_slope=negative_slope,
            )
            e1_in_channels = c1
        else:
            self.input_stem = nn.Identity()
            e1_in_channels = in_channels

        self.e1 = MultiScaleResBlock(e1_in_channels, c1, norm_type, negative_slope)
        self.down1 = nn.MaxPool2d(2)
        self.e2 = MultiScaleResBlock(c1, c2, norm_type, negative_slope)
        self.down2 = nn.MaxPool2d(2)
        self.e3 = MultiScaleResBlock(c2, c3, norm_type, negative_slope)
        self.down3 = nn.MaxPool2d(2)
        self.bottleneck = MultiScaleResBlock(c3, c4, norm_type, negative_slope)
        if self.use_aspp:
            self.aspp = LightASPP(
                c4,
                branch_channels=aspp_branch_channels,
                dilations=aspp_dilations,
                residual_scale=aspp_res_scale,
                norm_type=norm_type,
                negative_slope=negative_slope,
            )
        else:
            self.aspp = nn.Identity()

        self.film_e1 = ScalarFiLM(c1, scalar_embedding_channels, scalar_film_strength)
        self.film_e2 = ScalarFiLM(c2, scalar_embedding_channels, scalar_film_strength)
        self.film_e3 = ScalarFiLM(c3, scalar_embedding_channels, scalar_film_strength)
        self.film_b = ScalarFiLM(c4, scalar_embedding_channels, scalar_film_strength)
        self.film_d3 = ScalarFiLM(c3, scalar_embedding_channels, scalar_film_strength)
        self.film_d2 = ScalarFiLM(c2, scalar_embedding_channels, scalar_film_strength)
        self.film_d1 = ScalarFiLM(c1, scalar_embedding_channels, scalar_film_strength)

        self.up3 = UpBlock(c4, c3, norm_type, negative_slope)
        self.gate3 = ScalarAttentionSkipGate(c3, c3, scalar_embedding_channels, norm_type, negative_slope)
        self.d3 = MultiScaleResBlock(c3 + c3, c3, norm_type, negative_slope)

        self.up2 = UpBlock(c3, c2, norm_type, negative_slope)
        self.gate2 = ScalarAttentionSkipGate(c2, c2, scalar_embedding_channels, norm_type, negative_slope)
        self.d2 = MultiScaleResBlock(c2 + c2, c2, norm_type, negative_slope)

        self.up1 = UpBlock(c2, c1, norm_type, negative_slope)
        self.gate1 = ScalarAttentionSkipGate(c1, c1, scalar_embedding_channels, norm_type, negative_slope)
        self.d1 = MultiScaleResBlock(c1 + c1, c1, norm_type, negative_slope)

        self.map_head = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, 3, 1, 1, bias=True),
            build_norm(norm_type, c1 // 2),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(c1 // 2, out_channels, 1, 1, 0, bias=True),
            build_output_activation(out_activation),
        )
        self.scale_head = ScaleHead(
            bottleneck_channels=c4,
            scalar_channels=scalar_embedding_channels,
            hidden_channels=scalar_embedding_channels,
            out_channels=out_channels,
            negative_slope=negative_slope,
        )
        if self.use_dual_head:
            self.aux_map_head = nn.Sequential(
                nn.Conv2d(c1, c1 // 2, 3, 1, 1, bias=True),
                build_norm(norm_type, c1 // 2),
                nn.LeakyReLU(negative_slope, inplace=True),
                nn.Conv2d(c1 // 2, out_channels, 1, 1, 0, bias=True),
                build_output_activation(out_activation),
            )
            self.aux_scale_head = ScaleHead(
                bottleneck_channels=c4,
                scalar_channels=scalar_embedding_channels,
                hidden_channels=scalar_embedding_channels,
                out_channels=out_channels,
                negative_slope=negative_slope,
            )
            self.fusion_gate_head = FusionGateHead(
                bottleneck_channels=c4,
                scalar_channels=scalar_embedding_channels,
                hidden_channels=scalar_embedding_channels,
                out_channels=out_channels,
                gate_min=dual_head_gate_min,
                gate_max=dual_head_gate_max,
                gate_init=dual_head_gate_init,
                negative_slope=negative_slope,
            )
        else:
            self.aux_map_head = None
            self.aux_scale_head = None
            self.fusion_gate_head = None

    def apply_film(self, name, module, x, scalar_embedding):
        if not self.use_scalar_film or name not in self.scalar_film_layers:
            return x
        return module(x, scalar_embedding)

    def zero_init_conditioning(self):
        self.scale_head.zero_init_output()
        if self.aux_scale_head is not None:
            self.aux_scale_head.zero_init_output()
        if self.fusion_gate_head is not None:
            self.fusion_gate_head.reset_gate()
        self.gate1.zero_init_gate()
        self.gate2.zero_init_gate()
        self.gate3.zero_init_gate()
        for module in [
            self.film_e1,
            self.film_e2,
            self.film_e3,
            self.film_b,
            self.film_d3,
            self.film_d2,
            self.film_d1,
        ]:
            module.zero_init()

    def split_inputs(self, inputs):
        if isinstance(inputs, dict):
            x = inputs.get("map", inputs.get("feature"))
            scalar = inputs.get("scalar")
        elif isinstance(inputs, (tuple, list)):
            x, scalar = inputs
        else:
            x = inputs
            scalar = None

        if x is None:
            raise ValueError("ScalarGatedResUNet input must contain a map tensor")

        if scalar is None:
            if self.require_scalar:
                raise ValueError("ScalarGatedResUNet requires scalar input; check dataset scalar_input_stats")
            scalar = x.new_zeros((x.size(0), self.scalar_channels))
        elif scalar.dim() > 2:
            scalar = scalar.view(scalar.size(0), -1)
        return x, scalar.to(device=x.device, dtype=x.dtype)

    def apply_prediction_scale(self, base_map, scale_head, bottleneck, scalar_embedding):
        log_scale = scale_head(bottleneck, scalar_embedding)
        if self.scale_log_clamp is not None:
            scale_log_clamp = float(self.scale_log_clamp)
            if scale_log_clamp <= 0:
                log_scale = torch.zeros_like(log_scale)
            else:
                log_scale = scale_log_clamp * torch.tanh(log_scale / scale_log_clamp)
        return base_map * torch.exp(log_scale)

    def forward(self, inputs):
        x, scalar = self.split_inputs(inputs)
        scalar_embedding = self.scalar_encoder(scalar)
        x = self.input_stem(x)

        e1 = self.apply_film("e1", self.film_e1, self.e1(x), scalar_embedding)
        e2 = self.apply_film("e2", self.film_e2, self.e2(self.down1(e1)), scalar_embedding)
        e3 = self.apply_film("e3", self.film_e3, self.e3(self.down2(e2)), scalar_embedding)
        b = self.bottleneck(self.down3(e3))
        b = self.aspp(b)
        b = self.apply_film("b", self.film_b, b, scalar_embedding)

        x = self.up3(b, e3)
        x = self.apply_film(
            "d3",
            self.film_d3,
            self.d3(torch.cat([x, self.gate3(e3, x, scalar_embedding)], dim=1)),
            scalar_embedding,
        )
        x = self.up2(x, e2)
        x = self.apply_film(
            "d2",
            self.film_d2,
            self.d2(torch.cat([x, self.gate2(e2, x, scalar_embedding)], dim=1)),
            scalar_embedding,
        )
        x = self.up1(x, e1)
        x = self.apply_film(
            "d1",
            self.film_d1,
            self.d1(torch.cat([x, self.gate1(e1, x, scalar_embedding)], dim=1)),
            scalar_embedding,
        )

        main = self.apply_prediction_scale(self.map_head(x), self.scale_head, b, scalar_embedding)
        if not self.use_dual_head:
            return main

        aux = self.apply_prediction_scale(self.aux_map_head(x), self.aux_scale_head, b, scalar_embedding)
        gate = self.fusion_gate_head(b, scalar_embedding)
        return {
            "main": main,
            "aux": aux,
            "gate": gate,
        }

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
            self.zero_init_conditioning()
        else:
            raise TypeError(
                "'pretrained' must be a str or None. "
                "But received {}".format(type(pretrained))
            )
