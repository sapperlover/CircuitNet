import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['L1Loss', 'MSELoss', 'L1CorrLoss', 'TargetIRL1Loss', 'TargetPhysCorrLoss']


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

class L1Loss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction='mean', sample_wise=False):
        super().__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)



class MSELoss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction='mean', sample_wise=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


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
        return self.loss_weight * (l1 + self.corr_loss_weight * corr)


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
        return self.loss_weight * (target_loss + self.ir_loss_weight * ir_loss)


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

        total = target_loss
        total = total + self.corr_loss_weight * corr
        total = total + self.ir_loss_weight * ir_loss
        total = total + self.phys_loss_weight * phys_loss
        return self.loss_weight * total
