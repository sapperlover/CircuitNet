import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['L1Loss', 'MSELoss', 'L1CorrLoss', 'TargetIRL1Loss', 'TargetPhysCorrLoss', 'DualHeadPhysLoss']


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()

    return loss.sum()


def mask_reduce_loss(loss, weight=None, reduction='mean', sample_wise=False):
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        if weight.size(1) == 1:
            weight = weight.expand_as(loss)
        eps = 1e-12

        if sample_wise:
            weight = weight.sum(dim=[1, 2, 3], keepdim=True)
            loss = (loss / (weight + eps)).sum() / weight.size(0)
        else:
            loss = loss.sum() / (weight.sum() + eps)

    return loss

def masked_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                sample_wise=False,
                **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = mask_reduce_loss(loss, weight, reduction, sample_wise)
        return loss

    return wrapper

@masked_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@masked_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


def corr_loss(pred, target, eps=1e-6):
    pred = pred.flatten(1)
    target = target.flatten(1)

    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    numerator = (pred * target).sum(dim=1)
    denominator = torch.sqrt((pred ** 2).sum(dim=1) + eps) * torch.sqrt((target ** 2).sum(dim=1) + eps)
    corr = numerator / denominator
    return 1.0 - corr.mean()


def hotspot_weighted_smooth_l1(
    pred,
    target,
    gt_ir,
    hotspot_percentile=95.0,
    hotspot_weight=1.0,
    beta=1.0,
    eps=1e-12,
    positive_only=True,
    min_value=None,
    min_positive_pixels=1,
):
    with torch.no_grad():
        b, c = gt_ir.shape[:2]
        gt = gt_ir.detach()
        if positive_only:
            threshold = eps if min_value is None else float(min_value)
            mask = torch.zeros_like(gt, dtype=torch.bool)
            quantile = float(hotspot_percentile) / 100.0
            for sample_idx in range(b):
                for channel_idx in range(c):
                    channel = gt[sample_idx, channel_idx]
                    positive = channel > threshold
                    if int(positive.sum().item()) < int(min_positive_pixels):
                        continue
                    q = torch.quantile(channel[positive].flatten(), quantile)
                    mask[sample_idx, channel_idx] = positive & (channel >= q)
        else:
            flat = gt.flatten(2)
            q = torch.quantile(flat, hotspot_percentile / 100.0, dim=2)
            q = q.view(b, c, 1, 1)
            mask = gt >= q
        mask = mask.to(pred.dtype)
        weight = 1.0 + float(hotspot_weight) * mask

    loss = F.smooth_l1_loss(pred, target, beta=float(beta), reduction='none')
    loss = loss * weight
    return loss.mean()


def gradient_loss(pred, target, beta=1.0, eps=1e-12):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_x = F.smooth_l1_loss(pred_dx, target_dx, beta=float(beta), reduction='mean')
    loss_y = F.smooth_l1_loss(pred_dy, target_dy, beta=float(beta), reduction='mean')
    return loss_x + loss_y

class L1Loss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction='mean', sample_wise=False):
        super().__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        loss = l1_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)
        total = self.loss_weight * loss
        self.last_components = {
            'target_l1': loss.detach(),
            'weighted_total': total.detach(),
        }
        return total



class MSELoss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction='mean', sample_wise=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        loss = mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)
        total = self.loss_weight * loss
        self.last_components = {
            'target_mse': loss.detach(),
            'weighted_total': total.detach(),
        }
        return total


class L1CorrLoss(nn.Module):
    def __init__(self, corr_loss_weight=0.1, corr_eps=1e-6, loss_weight=100.0):
        super().__init__()
        self.corr_loss_weight = float(corr_loss_weight)
        self.corr_eps = float(corr_eps)
        self.loss_weight = float(loss_weight)

    def forward(self, pred, target, **kwargs):
        if isinstance(target, dict):
            target = target['target']
        l1 = F.l1_loss(pred, target, reduction='mean')
        corr = corr_loss(pred, target, eps=self.corr_eps)
        total = l1 + self.corr_loss_weight * corr
        self.last_components = {
            'target_l1': l1.detach(),
            'corr_loss': corr.detach(),
            'corr_contrib': (self.corr_loss_weight * corr).detach(),
            'unweighted_total': total.detach(),
            'weighted_total': (self.loss_weight * total).detach(),
        }
        return self.loss_weight * total


class TargetIRL1Loss(nn.Module):
    def __init__(
        self,
        ir_loss_weight=0.1,
        target_scale_factor=1.0,
        loss_weight=100.0,
        ir_norm='mean',
        ir_norm_eps=1e-12,
    ):
        super().__init__()
        self.ir_loss_weight = float(ir_loss_weight)
        self.target_scale_factor = float(target_scale_factor)
        self.loss_weight = float(loss_weight)
        self.ir_norm = ir_norm
        self.ir_norm_eps = float(ir_norm_eps)

    def normalize_ir(self, pred_ir, gt_ir):
        if self.ir_norm in (None, 'none'):
            return pred_ir, gt_ir
        if self.ir_norm == 'mean':
            scale = gt_ir.detach().abs().mean(dim=(2, 3), keepdim=True).clamp_min(self.ir_norm_eps)
        elif self.ir_norm == 'rms':
            scale = torch.sqrt((gt_ir.detach() ** 2).mean(dim=(2, 3), keepdim=True) + self.ir_norm_eps)
        else:
            raise ValueError('Unsupported ir_norm: {}'.format(self.ir_norm))
        return pred_ir / scale, gt_ir / scale

    def forward(self, pred, target, **kwargs):
        if not isinstance(target, dict):
            raise ValueError('TargetIRL1Loss expects target dict with target, ir, and label_scale')
        target_map = target['target']
        gt_ir = target['ir']
        label_scale = target['label_scale']

        target_loss = F.l1_loss(pred, target_map, reduction='mean')
        pred_ir = pred / self.target_scale_factor * label_scale
        pred_ir_norm, gt_ir_norm = self.normalize_ir(pred_ir, gt_ir)
        ir_loss = F.l1_loss(pred_ir_norm, gt_ir_norm, reduction='mean')
        total = target_loss + self.ir_loss_weight * ir_loss
        self.last_components = {
            'target_l1': target_loss.detach(),
            'ir_norm_l1': ir_loss.detach(),
            'ir_contrib': (self.ir_loss_weight * ir_loss).detach(),
            'unweighted_total': total.detach(),
            'weighted_total': (self.loss_weight * total).detach(),
        }
        return self.loss_weight * total


class TargetPhysCorrLoss(nn.Module):
    def __init__(
        self,
        corr_loss_weight=0.1,
        ir_loss_weight=0.1,
        phys_loss_weight=0.1,
        target_scale_factor=1.0,
        loss_weight=100.0,
        ir_norm='mean',
        phys_norm='mean',
        norm_eps=1e-12,
        corr_eps=1e-6,
        hotspot_loss_weight=0.0,
        hotspot_percentile=95.0,
        hotspot_weight=1.0,
        hotspot_beta=1.0,
        hotspot_positive_only=True,
        hotspot_min_value=None,
        hotspot_min_positive_pixels=1,
        gradient_loss_weight=0.0,
        gradient_beta=1.0,
    ):
        super().__init__()
        self.corr_loss_weight = float(corr_loss_weight)
        self.ir_loss_weight = float(ir_loss_weight)
        self.phys_loss_weight = float(phys_loss_weight)
        self.target_scale_factor = float(target_scale_factor)
        self.loss_weight = float(loss_weight)
        self.ir_norm = ir_norm
        self.phys_norm = phys_norm
        self.norm_eps = float(norm_eps)
        self.corr_eps = float(corr_eps)
        self.hotspot_loss_weight = float(hotspot_loss_weight)
        self.hotspot_percentile = float(hotspot_percentile)
        self.hotspot_weight = float(hotspot_weight)
        self.hotspot_beta = float(hotspot_beta)
        self.hotspot_positive_only = bool(hotspot_positive_only)
        self.hotspot_min_value = hotspot_min_value
        self.hotspot_min_positive_pixels = int(hotspot_min_positive_pixels)
        self.gradient_loss_weight = float(gradient_loss_weight)
        self.gradient_beta = float(gradient_beta)

    def normalize_pair(self, pred, target, mode):
        if mode in (None, 'none'):
            return pred, target
        if mode == 'mean':
            scale = target.detach().abs().mean(dim=(2, 3), keepdim=True).clamp_min(self.norm_eps)
        elif mode == 'rms':
            scale = torch.sqrt((target.detach() ** 2).mean(dim=(2, 3), keepdim=True) + self.norm_eps)
        else:
            raise ValueError('Unsupported norm mode: {}'.format(mode))
        return pred / scale, target / scale

    def forward(self, pred, target, **kwargs):
        if not isinstance(target, dict):
            raise ValueError('TargetPhysCorrLoss expects target dict with target, ir, label_scale, and aux_label_scale')
        target_map = target['target']
        gt_ir = target['ir']
        label_scale = target['label_scale']
        aux_label_scale = target['aux_label_scale']

        target_loss = F.l1_loss(pred, target_map, reduction='mean')
        corr = corr_loss(pred, target_map, eps=self.corr_eps)

        pred_ir = pred / self.target_scale_factor * label_scale
        pred_ir_norm, gt_ir_norm = self.normalize_pair(pred_ir, gt_ir, self.ir_norm)
        ir_loss = F.l1_loss(pred_ir_norm, gt_ir_norm, reduction='mean')

        pred_phys = pred * label_scale / aux_label_scale
        gt_phys = gt_ir / aux_label_scale * self.target_scale_factor
        pred_phys_norm, gt_phys_norm = self.normalize_pair(pred_phys, gt_phys, self.phys_norm)
        phys_loss = F.l1_loss(pred_phys_norm, gt_phys_norm, reduction='mean')

        hotspot_loss = pred.new_tensor(0.0)
        if self.hotspot_loss_weight > 0:
            hotspot_loss = hotspot_weighted_smooth_l1(
                pred,
                target_map,
                gt_ir,
                hotspot_percentile=self.hotspot_percentile,
                hotspot_weight=self.hotspot_weight,
                beta=self.hotspot_beta,
                eps=self.norm_eps,
                positive_only=self.hotspot_positive_only,
                min_value=self.hotspot_min_value,
                min_positive_pixels=self.hotspot_min_positive_pixels,
            )

        edge_loss = pred.new_tensor(0.0)
        if self.gradient_loss_weight > 0:
            edge_loss = gradient_loss(pred_ir_norm, gt_ir_norm, beta=self.gradient_beta, eps=self.norm_eps)

        total = target_loss
        total = total + self.corr_loss_weight * corr
        total = total + self.ir_loss_weight * ir_loss
        total = total + self.phys_loss_weight * phys_loss
        total = total + self.hotspot_loss_weight * hotspot_loss
        total = total + self.gradient_loss_weight * edge_loss
        self.last_components = {
            'target_l1': target_loss.detach(),
            'corr_loss': corr.detach(),
            'corr_contrib': (self.corr_loss_weight * corr).detach(),
            'ir_norm_l1': ir_loss.detach(),
            'ir_contrib': (self.ir_loss_weight * ir_loss).detach(),
            'phys_norm_l1': phys_loss.detach(),
            'phys_contrib': (self.phys_loss_weight * phys_loss).detach(),
            'hotspot_smooth_l1': hotspot_loss.detach(),
            'hotspot_contrib': (self.hotspot_loss_weight * hotspot_loss).detach(),
            'gradient_smooth_l1': edge_loss.detach(),
            'gradient_contrib': (self.gradient_loss_weight * edge_loss).detach(),
            'unweighted_total': total.detach(),
            'weighted_total': (self.loss_weight * total).detach(),
        }
        return self.loss_weight * total


class DualHeadPhysLoss(nn.Module):
    def __init__(
        self,
        corr_loss_weight=0.1,
        ir_loss_weight=0.1,
        aux_target_loss_weight=0.2,
        aux_ir_loss_weight=0.1,
        fused_ir_loss_weight=0.5,
        fused_corr_loss_weight=0.1,
        consistency_loss_weight=0.05,
        target_scale_factor=1.0,
        target_clip_max=None,
        loss_weight=100.0,
        ir_norm='mean',
        norm_eps=1e-12,
        corr_eps=1e-6,
    ):
        super().__init__()
        self.corr_loss_weight = float(corr_loss_weight)
        self.ir_loss_weight = float(ir_loss_weight)
        self.aux_target_loss_weight = float(aux_target_loss_weight)
        self.aux_ir_loss_weight = float(aux_ir_loss_weight)
        self.fused_ir_loss_weight = float(fused_ir_loss_weight)
        self.fused_corr_loss_weight = float(fused_corr_loss_weight)
        self.consistency_loss_weight = float(consistency_loss_weight)
        self.target_scale_factor = float(target_scale_factor)
        self.target_clip_max = target_clip_max
        self.loss_weight = float(loss_weight)
        self.ir_norm = ir_norm
        self.norm_eps = float(norm_eps)
        self.corr_eps = float(corr_eps)

    def normalize_pair(self, pred, target):
        if self.ir_norm in (None, 'none'):
            return pred, target
        if self.ir_norm == 'mean':
            scale = target.detach().abs().mean(dim=(2, 3), keepdim=True).clamp_min(self.norm_eps)
        elif self.ir_norm == 'rms':
            scale = torch.sqrt((target.detach() ** 2).mean(dim=(2, 3), keepdim=True) + self.norm_eps)
        else:
            raise ValueError('Unsupported ir_norm: {}'.format(self.ir_norm))
        return pred / scale, target / scale

    def normalize_by_reference(self, pred, reference):
        normalized, _ = self.normalize_pair(pred, reference)
        return normalized

    def broadcast_gate(self, gate, ref):
        if gate is None:
            return ref.new_zeros((ref.size(0), ref.size(1), 1, 1))
        if gate.dim() == 2:
            gate = gate.view(gate.size(0), gate.size(1), 1, 1)
        if gate.size(1) == 1 and ref.size(1) != 1:
            gate = gate.expand(-1, ref.size(1), -1, -1)
        return gate.to(device=ref.device, dtype=ref.dtype)

    def forward(self, pred, target, **kwargs):
        if not isinstance(pred, dict):
            raise ValueError('DualHeadPhysLoss expects prediction dict with main, aux, and gate')
        if not isinstance(target, dict):
            raise ValueError('DualHeadPhysLoss expects target dict with target, ir, label_scale, and aux_label_scale')

        main_pred = pred['main']
        aux_pred = pred['aux']
        gate = self.broadcast_gate(pred.get('gate'), main_pred)

        target_map = target['target']
        gt_ir = target['ir']
        label_scale = target['label_scale']
        aux_label_scale = target['aux_label_scale']

        aux_target = gt_ir / aux_label_scale * self.target_scale_factor
        if self.target_clip_max is not None:
            aux_target = torch.clamp(aux_target, max=float(self.target_clip_max))

        main_target_loss = F.l1_loss(main_pred, target_map, reduction='mean')
        main_corr = corr_loss(main_pred, target_map, eps=self.corr_eps)
        main_ir = main_pred / self.target_scale_factor * label_scale
        main_ir_norm, gt_ir_norm = self.normalize_pair(main_ir, gt_ir)
        main_ir_loss = F.l1_loss(main_ir_norm, gt_ir_norm, reduction='mean')

        aux_target_loss = F.l1_loss(aux_pred, aux_target, reduction='mean')
        aux_ir = aux_pred / self.target_scale_factor * aux_label_scale
        aux_ir_norm, _ = self.normalize_pair(aux_ir, gt_ir)
        aux_ir_loss = F.l1_loss(aux_ir_norm, gt_ir_norm, reduction='mean')

        aux_as_main = aux_pred * aux_label_scale / label_scale
        fused_target = (1.0 - gate) * main_pred + gate * aux_as_main
        fused_ir = fused_target / self.target_scale_factor * label_scale
        fused_ir_norm, _ = self.normalize_pair(fused_ir, gt_ir)
        fused_ir_loss = F.l1_loss(fused_ir_norm, gt_ir_norm, reduction='mean')
        fused_corr = corr_loss(fused_target, target_map, eps=self.corr_eps)

        consistency_loss = F.l1_loss(
            self.normalize_by_reference(main_ir, gt_ir),
            self.normalize_by_reference(aux_ir, gt_ir),
            reduction='mean',
        )

        total = main_target_loss
        total = total + self.corr_loss_weight * main_corr
        total = total + self.ir_loss_weight * main_ir_loss
        total = total + self.aux_target_loss_weight * aux_target_loss
        total = total + self.aux_ir_loss_weight * aux_ir_loss
        total = total + self.fused_ir_loss_weight * fused_ir_loss
        total = total + self.fused_corr_loss_weight * fused_corr
        total = total + self.consistency_loss_weight * consistency_loss
        self.last_components = {
            'main_target_l1': main_target_loss.detach(),
            'main_corr_loss': main_corr.detach(),
            'main_corr_contrib': (self.corr_loss_weight * main_corr).detach(),
            'main_ir_norm_l1': main_ir_loss.detach(),
            'main_ir_contrib': (self.ir_loss_weight * main_ir_loss).detach(),
            'aux_target_l1': aux_target_loss.detach(),
            'aux_target_contrib': (self.aux_target_loss_weight * aux_target_loss).detach(),
            'aux_ir_norm_l1': aux_ir_loss.detach(),
            'aux_ir_contrib': (self.aux_ir_loss_weight * aux_ir_loss).detach(),
            'fused_ir_norm_l1': fused_ir_loss.detach(),
            'fused_ir_contrib': (self.fused_ir_loss_weight * fused_ir_loss).detach(),
            'fused_corr_loss': fused_corr.detach(),
            'fused_corr_contrib': (self.fused_corr_loss_weight * fused_corr).detach(),
            'consistency_l1': consistency_loss.detach(),
            'consistency_contrib': (self.consistency_loss_weight * consistency_loss).detach(),
            'gate_mean': gate.detach().mean(),
            'unweighted_total': total.detach(),
            'weighted_total': (self.loss_weight * total).detach(),
        }
        return self.loss_weight * total
