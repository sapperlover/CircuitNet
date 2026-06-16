import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from datasets.build_dataset import build_dataset
import utils.losses as losses
from models.build_model import build_model
from utils.arg_parser import Parser
from utils.logger import build_logger
from math import cos, pi

 
def checkpoint(
    logger,
    model,
    epoch,
    save_path,
    label_norm_stats=None,
    scalar_norm_stats=None,
    filename=None,
    extra=None,
):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if filename is None:
        filename = "model_iters_{}.pth".format(epoch)
    model_out_path = os.path.join(save_path, filename)
    checkpoint_data = {'state_dict': model.state_dict(), 'iter': epoch}
    if label_norm_stats is not None:
        checkpoint_data['label_norm_stats'] = label_norm_stats
    if scalar_norm_stats is not None:
        checkpoint_data['scalar_norm_stats'] = scalar_norm_stats
    if extra:
        checkpoint_data.update(extra)
    torch.save(checkpoint_data, model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))


def save_train_loss_min(logger, model, iter_num, train_loss, save_path, label_norm_stats=None, scalar_norm_stats=None):
    checkpoint(
        logger,
        model,
        iter_num,
        save_path,
        label_norm_stats,
        scalar_norm_stats,
        filename='model_train_loss_min.pth',
        extra={'train_loss_min': train_loss},
    )
    record_path = os.path.join(save_path, 'train_loss_min.json')
    with open(record_path, 'wt') as f:
        json.dump({'iter': iter_num, 'train_loss_min': train_loss}, f, indent=4)
    logger.info("Train loss min updated at iter {}: {:.6f}".format(iter_num, train_loss))


def should_save_periodic_checkpoint(iter_num, max_iters, early_ratio=0.75, early_freq=5000, late_freq=2000):
    if iter_num <= 0:
        return False
    if iter_num >= max_iters:
        return True
    switch_iter = int(max_iters * early_ratio)
    if iter_num <= switch_iter:
        return iter_num % early_freq == 0
    return iter_num % late_freq == 0
        
def build_loss(args):
    loss_type = args.pop('loss_type')
    if loss_type == 'TargetIRL1Loss':
        return losses.__dict__[loss_type](
            ir_loss_weight=args.get('ir_loss_weight', 0.1),
            target_scale_factor=args.get('target_scale_factor', 1.0),
            ir_norm=args.get('ir_norm', 'mean'),
            ir_norm_eps=args.get('ir_norm_eps', 1e-12),
        )
    if loss_type == 'L1CorrLoss':
        return losses.__dict__[loss_type](
            corr_loss_weight=args.get('corr_loss_weight', 0.1),
            corr_eps=args.get('corr_eps', 1e-6),
            loss_weight=args.get('loss_weight', 100.0),
        )
    if loss_type == 'TargetPhysCorrLoss':
        return losses.__dict__[loss_type](
            corr_loss_weight=args.get('corr_loss_weight', 0.1),
            ir_loss_weight=args.get('ir_loss_weight', 0.1),
            phys_loss_weight=args.get('phys_loss_weight', 0.1),
            target_scale_factor=args.get('target_scale_factor', 1.0),
            loss_weight=args.get('loss_weight', 100.0),
            ir_norm=args.get('ir_norm', 'mean'),
            phys_norm=args.get('phys_norm', 'mean'),
            norm_eps=args.get('norm_eps', 1e-12),
            corr_eps=args.get('corr_eps', 1e-6),
            hotspot_loss_weight=args.get('hotspot_loss_weight', 0.0),
            hotspot_percentile=args.get('hotspot_percentile', 95.0),
            hotspot_weight=args.get('hotspot_weight', 1.0),
            hotspot_beta=args.get('hotspot_beta', 1.0),
            hotspot_positive_only=args.get('hotspot_positive_only', True),
            hotspot_min_value=args.get('hotspot_min_value', None),
            hotspot_min_positive_pixels=args.get('hotspot_min_positive_pixels', 1),
            gradient_loss_weight=args.get('gradient_loss_weight', 0.0),
            gradient_beta=args.get('gradient_beta', 1.0),
        )
    if loss_type == 'DualHeadPhysLoss':
        return losses.__dict__[loss_type](
            corr_loss_weight=args.get('corr_loss_weight', 0.1),
            ir_loss_weight=args.get('ir_loss_weight', 0.1),
            aux_target_loss_weight=args.get('aux_target_loss_weight', 0.2),
            aux_ir_loss_weight=args.get('aux_ir_loss_weight', 0.1),
            fused_ir_loss_weight=args.get('fused_ir_loss_weight', 0.5),
            fused_corr_loss_weight=args.get('fused_corr_loss_weight', 0.1),
            consistency_loss_weight=args.get('consistency_loss_weight', 0.05),
            target_scale_factor=args.get('target_scale_factor', 1.0),
            target_clip_max=args.get('target_clip_max', None),
            loss_weight=args.get('loss_weight', 100.0),
            ir_norm=args.get('ir_norm', 'mean'),
            norm_eps=args.get('norm_eps', 1e-12),
            corr_eps=args.get('corr_eps', 1e-6),
        )
    return losses.__dict__[loss_type]()

def to_jsonable(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value

def move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(move_to_device(value, device) for value in data)
    return data

class CosineRestartLr(object):
    def __init__(self,
                 base_lr,
                 periods,
                 restart_weights = [1],
                 min_lr = None,
                 min_lr_ratio = None):
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        super().__init__()

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

        self.base_lr = base_lr

    def annealing_cos(self, start: float,
                    end: float,
                    factor: float,
                    weight: float = 1.) -> float:
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_position_from_periods(self, iteration: int, cumulative_periods):
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i
        raise ValueError(f'Current iteration {iteration} exceeds '
                        f'cumulative_periods {cumulative_periods}')


    def get_lr(self, iter_num, base_lr: float):
        target_lr = self.min_lr  # type:ignore

        idx = self.get_position_from_periods(iter_num, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((iter_num - nearest_restart) / current_periods, 1)
        return self.annealing_cos(base_lr, target_lr, alpha, current_weight)

    
    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups,
                                        lr_groups):
                param_group['lr'] = lr

    def get_regular_lr(self, iter_num):
        return [self.get_lr(iter_num, _base_lr) for _base_lr in self.base_lr]  # iters

    def set_init_lr(self, optimizer):
        for group in optimizer.param_groups:  # type: ignore
            group.setdefault('initial_lr', group['lr'])
            self.base_lr = [group['initial_lr'] for group in optimizer.param_groups  # type: ignore
        ]


def train():
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)

    gpu = None
    if arg.gpu is not None:
        gpu = int(arg.gpu)

    # Initialize hyperparams from json
    if arg.args is not None:
        with open(arg.args, 'rt') as f:
            arg_dict.update(json.load(f))

    arg_dict['max_iters'] = int(arg_dict['max_iters'])
    arg_dict.setdefault('ckpt_save_early_ratio', 0.75)
    arg_dict.setdefault('ckpt_save_early_freq', 5000)
    arg_dict.setdefault('ckpt_save_late_freq', 2000)
    arg_dict.setdefault('train_loss_min_freq', 500)
    arg_dict['ckpt_save_early_ratio'] = min(max(float(arg_dict['ckpt_save_early_ratio']), 0.0), 1.0)
    arg_dict['ckpt_save_early_freq'] = max(1, int(arg_dict['ckpt_save_early_freq']))
    arg_dict['ckpt_save_late_freq'] = max(1, int(arg_dict['ckpt_save_late_freq']))
    arg_dict['train_loss_min_freq'] = max(1, int(arg_dict['train_loss_min_freq']))

    arg_dict['test_mode'] = False 

    logger, log_dir = build_logger(arg_dict)
    logger.info(arg_dict)

    if gpu is not None:
        arg_dict['gpu'] = gpu
    
    if arg_dict['cpu']:
        device = torch.device("cpu")
        logger.info('using cpu for training')
    elif arg_dict['gpu'] is not None:
        torch.cuda.set_device(arg_dict['gpu'])
        device = torch.device("cuda", arg_dict['gpu'])
        logger.info('using gpu {} for training'.format(arg_dict['gpu']))

    saved_arg_dict = dict(arg_dict)
    with open(os.path.join(log_dir, 'train.json'), 'wt') as f:
      json.dump(saved_arg_dict, f, indent=4)

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)

    logger.info('===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(arg_dict)
    train_dataset = getattr(dataset, 'dataset', None)
    label_norm_stats = getattr(train_dataset, 'label_norm_stats', None)
    power_epsilon = getattr(train_dataset, 'power_epsilon', None)
    scalar_norm_stats = getattr(train_dataset, 'scalar_norm_stats', None)
    map_feature_norm_stats = getattr(train_dataset, 'map_feature_norm_stats', None)
    target_clip_max = getattr(train_dataset, 'target_clip_max', None)
    rewrite_train_config = False
    if power_epsilon is not None:
        arg_dict['power_epsilon'] = power_epsilon
        saved_arg_dict['power_epsilon'] = to_jsonable(power_epsilon)
        logger.info('power epsilon: {}'.format(power_epsilon))
        rewrite_train_config = True
    if target_clip_max is not None:
        arg_dict['target_clip_max'] = target_clip_max
        saved_arg_dict['target_clip_max'] = to_jsonable(target_clip_max)
        logger.info('target clip max: {}'.format(target_clip_max))
        rewrite_train_config = True
    if scalar_norm_stats is not None:
        arg_dict['scalar_norm_stats'] = scalar_norm_stats
        saved_arg_dict['scalar_norm_stats'] = to_jsonable(scalar_norm_stats)
        logger.info('scalar norm stats: {}'.format(scalar_norm_stats))
        rewrite_train_config = True
    if map_feature_norm_stats is not None:
        arg_dict['map_feature_norm_stats'] = map_feature_norm_stats
        saved_arg_dict['map_feature_norm_stats'] = to_jsonable(map_feature_norm_stats)
        logger.info('map feature norm stats: {}'.format(map_feature_norm_stats))
        rewrite_train_config = True
    if rewrite_train_config:
        with open(os.path.join(log_dir, 'train.json'), 'wt') as f:
            json.dump(saved_arg_dict, f, indent=4)
    if label_norm_stats is not None:
        logger.info('label norm stats: {}'.format(label_norm_stats))
        with open(os.path.join(log_dir, 'label_norm.json'), 'wt') as f:
            json.dump(label_norm_stats, f, indent=4)

    logger.info('===> Building model')
    # Initialize model parameters
    model = build_model(arg_dict)
    model = model.to(device)
    
    # Build loss
    loss = build_loss(arg_dict)

    # Build Optimzer
    optimizer = optim.AdamW(model.parameters(), lr=arg_dict['lr'],  betas=(0.9, 0.999), weight_decay=arg_dict['weight_decay'])

    # Build lr scheduler
    cosine_lr = CosineRestartLr(arg_dict['lr'], [arg_dict['max_iters']], [1], 1e-7)
    cosine_lr.set_init_lr(optimizer)

    epoch_loss = 0
    epoch_loss_count = 0
    loss_component_sums = {}
    loss_component_count = 0
    train_loss_min_sum = 0
    train_loss_min_count = 0
    best_train_loss = float('inf')
    iter_num = 0
    print_freq = min(100, int(arg_dict['max_iters']/10))
    print_freq = max(1, print_freq)
    train_loss_min_freq = max(1, min(arg_dict['train_loss_min_freq'], arg_dict['max_iters']))
    logger.info(
        'checkpoint schedule: first {:.0%} every {} iters, last {:.0%} every {} iters, train_loss_min every {} iters'.format(
            arg_dict['ckpt_save_early_ratio'],
            arg_dict['ckpt_save_early_freq'],
            1.0 - arg_dict['ckpt_save_early_ratio'],
            arg_dict['ckpt_save_late_freq'],
            train_loss_min_freq,
        )
    )

    while iter_num < arg_dict['max_iters']:
        with tqdm(total=print_freq) as bar:
            for feature, label, _ in dataset:        
                if arg_dict['cpu']:
                    input, target = feature, label
                else:
                    input, target = move_to_device(feature, device), move_to_device(label, device)

                regular_lr = cosine_lr.get_regular_lr(iter_num)
                cosine_lr._set_lr(optimizer, regular_lr)

                prediction = model(input)

                optimizer.zero_grad()
                pixel_loss = loss(prediction, target)

                loss_value = pixel_loss.item()
                epoch_loss += loss_value
                epoch_loss_count += 1
                train_loss_min_sum += loss_value
                train_loss_min_count += 1
                loss_components = getattr(loss, 'last_components', None)
                if loss_components:
                    for name, value in loss_components.items():
                        loss_component_sums[name] = loss_component_sums.get(name, 0.0) + float(value.detach().cpu())
                    loss_component_count += 1
                pixel_loss.backward()
                optimizer.step()

                iter_num += 1
                
                bar.update(1)
                if should_save_periodic_checkpoint(
                    iter_num,
                    arg_dict['max_iters'],
                    arg_dict['ckpt_save_early_ratio'],
                    arg_dict['ckpt_save_early_freq'],
                    arg_dict['ckpt_save_late_freq'],
                ):
                    checkpoint(logger, model, iter_num, log_dir, label_norm_stats, scalar_norm_stats)

                if iter_num % train_loss_min_freq == 0 or iter_num >= arg_dict['max_iters']:
                    if train_loss_min_count > 0:
                        train_loss = train_loss_min_sum / train_loss_min_count
                        writer.add_scalar('Loss/train_loss_min_window', train_loss, iter_num)
                        if train_loss < best_train_loss:
                            best_train_loss = train_loss
                            save_train_loss_min(
                                logger,
                                model,
                                iter_num,
                                best_train_loss,
                                log_dir,
                                label_norm_stats,
                                scalar_norm_stats,
                            )
                    train_loss_min_sum = 0
                    train_loss_min_count = 0

                if iter_num % print_freq == 0:
                    break
                if iter_num >= arg_dict['max_iters']:
                    break

        avg_epoch_loss = epoch_loss / max(epoch_loss_count, 1)
        logger.info("===> Iters[{}]({}/{}): Loss: {:.4f}".format(iter_num, iter_num, arg_dict['max_iters'], avg_epoch_loss))
        writer.add_scalar('Loss/training loss', avg_epoch_loss, iter_num)
        if loss_component_count > 0:
            component_msg = []
            for name in sorted(loss_component_sums.keys()):
                value = loss_component_sums[name] / loss_component_count
                component_msg.append('{}: {:.6f}'.format(name, value))
                writer.add_scalar('LossComponents/{}'.format(name), value, iter_num)
            logger.info('===> Loss components: {}'.format(', '.join(component_msg)))
        epoch_loss = 0
        epoch_loss_count = 0
        loss_component_sums = {}
        loss_component_count = 0


    writer.close()

if __name__ == "__main__":
    train()
