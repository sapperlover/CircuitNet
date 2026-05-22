import os
import copy
import numpy as np
from pathlib import Path
from scipy import ndimage
from torchvision.transforms import Compose

class TrainDataset(object):
    def __init__(self, ann_file, dataroot, pipeline=None, test_mode=False, **kwargs):
        super().__init__()
        self.ann_file = ann_file
        self.dataroot = dataroot
        self.test_mode = test_mode
        self.label_norm = kwargs.get('label_norm', True)
        self.label_clip_min = kwargs.get('label_clip_min', None)
        self.label_clip_max = kwargs.get('label_clip_max', None)
        self.label_scale_mode = kwargs.get('label_scale_mode', 'none')
        self.power_smooth_sigma = kwargs.get('power_smooth_sigma', 5.0)
        self.effres_smooth_sigma = kwargs.get('effres_smooth_sigma', 5.0)
        self.power_epsilon = kwargs.get('power_epsilon', None)
        self.power_epsilon_mode = kwargs.get('power_epsilon_mode', 'sample_max')
        self.power_epsilon_ratio = kwargs.get('power_epsilon_ratio', 0.01)
        self.target_scale_factor = kwargs.get('target_scale_factor', 1.0)
        self.target_clip_max = kwargs.get('target_clip_max', None)
        if pipeline:
            self.pipeline = Compose(pipeline)
            if self.uses_label_scale():
                for transform in self.pipeline.transforms:
                    if hasattr(transform, 'keys') and 'label_scale' not in transform.keys:
                        transform.keys.append('label_scale')
        else:
            self.pipeline = None

        self.data_infos = self.load_annotations()
        self.power_epsilon = self.resolve_power_epsilon()
        self.target_clip_max = self.resolve_target_clip_max()
        self.label_min = None
        self.label_max = None
        self.label_norm_stats = None
        if self.label_norm:
            self.label_min, self.label_max = self.compute_label_norm_stats()
            self.label_norm_stats = {
                'min': self.label_min.tolist(),
                'max': self.label_max.tolist(),
            }

    def load_annotations(self):
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                if not line.strip():
                    continue
                else:
                    feature, label = line.strip().split(',')
                    if self.dataroot is not None:
                        feature_path = os.path.join(self.dataroot, feature)
                        label_path = os.path.join(self.dataroot, label)
                    data_infos.append(dict(feature_path=feature_path, label_path=label_path))
        return data_infos

    def compute_label_norm_stats(self):
        label_min = None
        label_max = None
        for data_info in self.data_infos:
            label = np.load(data_info['label_path'])
            label = self.clip_label(label)
            if self.uses_label_scale():
                label = self.apply_label_scale(label, data_info['feature_path'])
            channel_min = label.min(axis=(0, 1))
            channel_max = label.max(axis=(0, 1))
            if label_min is None:
                label_min = channel_min
                label_max = channel_max
            else:
                label_min = np.minimum(label_min, channel_min)
                label_max = np.maximum(label_max, channel_max)

        return label_min.astype(np.float32), label_max.astype(np.float32)

    def uses_label_scale(self):
        return self.label_scale_mode not in (None, 'none')

    def uses_fixed_power_epsilon(self):
        return self.power_epsilon_mode in ('train_nonzero_mean', 'fixed')

    def resize_map(self, data, out_shape):
        if data.shape == tuple(out_shape):
            return data
        return ndimage.zoom(data, (out_shape[0] / data.shape[0], out_shape[1] / data.shape[1]), order=1)

    def resolve_out_feature_path(self, feature_path, feature_name):
        path = Path(feature_path)
        parts = path.parts
        if 'training_set' not in parts:
            raise ValueError('feature path does not contain training_set: {}'.format(feature_path))
        index = parts.index('training_set')
        tech = parts[index + 1]
        filename = path.name
        root = Path(*parts[:index]) if index > 0 else Path('.')
        return root / 'out' / tech / 'features' / feature_name / filename

    def load_smooth_power(self, feature_path, out_shape):
        power_path = self.resolve_out_feature_path(feature_path, 'total_power')
        power = np.load(power_path).astype(np.float32)
        power = self.resize_map(power, out_shape)
        power = np.maximum(power, 0.0)
        return ndimage.gaussian_filter(power, sigma=self.power_smooth_sigma, mode='nearest')

    def load_smooth_feature(self, feature_path, feature_name, out_shape, sigma):
        feature_map_path = self.resolve_out_feature_path(feature_path, feature_name)
        feature_map = np.load(feature_map_path).astype(np.float32)
        feature_map = self.resize_map(feature_map, out_shape)
        feature_map = np.maximum(feature_map, 0.0)
        return ndimage.gaussian_filter(feature_map, sigma=sigma, mode='nearest')

    def build_scale_base(self, feature_path, out_shape):
        if self.label_scale_mode == 'smooth_power':
            return self.load_smooth_power(feature_path, out_shape)[:, :, None]
        if self.label_scale_mode == 'power_effres':
            power = self.load_smooth_power(feature_path, out_shape)
            eff_vdd = self.load_smooth_feature(feature_path, 'eff_res_VDD', out_shape, self.effres_smooth_sigma)
            eff_vss = self.load_smooth_feature(feature_path, 'eff_res_VSS', out_shape, self.effres_smooth_sigma)
            scale_vdd = power * eff_vdd
            scale_vss = power * eff_vss
            return np.stack([scale_vdd, scale_vss], axis=2)
        raise ValueError('Unsupported label_scale_mode: {}'.format(self.label_scale_mode))

    def compute_train_nonzero_scale_mean(self):
        total = None
        count = None
        for data_info in self.data_infos:
            label = np.load(data_info['label_path'])
            scale_base = self.build_scale_base(data_info['feature_path'], label.shape[:2])
            channels = scale_base.shape[2]
            if total is None:
                total = np.zeros(channels, dtype=np.float64)
                count = np.zeros(channels, dtype=np.int64)
            for channel in range(channels):
                nonzero_scale = scale_base[:, :, channel][scale_base[:, :, channel] > 0]
                total[channel] += float(nonzero_scale.sum())
                count[channel] += int(nonzero_scale.size)
        if total is None:
            return 0.0
        mean = total / np.maximum(count, 1)
        mean = np.maximum(mean, 0.0)
        if mean.size == 1:
            return float(mean[0])
        return mean.astype(np.float32)

    def resolve_power_epsilon(self):
        if not self.uses_label_scale():
            return self.power_epsilon
        if self.power_epsilon is not None:
            power_epsilon = np.asarray(self.power_epsilon, dtype=np.float32)
            if power_epsilon.ndim == 0:
                return float(power_epsilon)
            return power_epsilon
        if self.power_epsilon_mode == 'sample_max':
            return None
        if self.power_epsilon_mode == 'train_nonzero_mean':
            epsilon = np.asarray(self.compute_train_nonzero_scale_mean(), dtype=np.float32) * self.power_epsilon_ratio
            epsilon = np.maximum(epsilon, 1e-12)
            if epsilon.ndim == 0:
                return float(epsilon)
            return epsilon
        raise ValueError('Unsupported power_epsilon_mode: {}'.format(self.power_epsilon_mode))

    def parse_target_clip_percentile(self):
        if not isinstance(self.target_clip_max, str):
            return None
        value = self.target_clip_max.strip().lower()
        if not value.startswith('p'):
            return None
        return float(value[1:])

    def resolve_target_clip_max(self):
        if self.target_clip_max is None:
            return None
        percentile = self.parse_target_clip_percentile()
        if percentile is None:
            return float(self.target_clip_max)

        samples = []
        max_pixels_per_case = 4096
        for data_info in self.data_infos:
            label = np.load(data_info['label_path'])
            label = self.clip_label(label)
            if self.uses_label_scale():
                label = self.apply_label_scale(label, data_info['feature_path'], apply_clip=False)
            flat = label.reshape(-1)
            if flat.size > max_pixels_per_case:
                index = np.linspace(0, flat.size - 1, max_pixels_per_case).astype(np.int64)
                flat = flat[index]
            samples.append(flat.astype(np.float32))
        if not samples:
            return None
        return float(np.percentile(np.concatenate(samples), percentile))

    def build_label_scale(self, feature_path, out_shape):
        if not self.uses_label_scale():
            return None
        scale_base = self.build_scale_base(feature_path, out_shape)

        if self.power_epsilon is None:
            epsilon = np.maximum(scale_base.max(axis=(0, 1)) * self.power_epsilon_ratio, 1e-12)
        else:
            epsilon = np.asarray(self.power_epsilon, dtype=np.float32)
        return np.ascontiguousarray((scale_base + epsilon.reshape(1, 1, -1)).astype(np.float32))

    def apply_label_scale(self, label, feature_path, apply_clip=True):
        label = label / self.build_label_scale(feature_path, label.shape[:2]) * self.target_scale_factor
        if apply_clip and self.target_clip_max is not None:
            label = np.minimum(label, self.target_clip_max)
        return label

    def clip_label(self, label):
        if self.label_clip_min is None and self.label_clip_max is None:
            return label
        if self.label_clip_min is not None:
            label = np.maximum(label, self.label_clip_min)
        if self.label_clip_max is not None:
            label = np.minimum(label, self.label_clip_max)
        return label

    def normalize_label(self, label):
        scale = self.label_max - self.label_min
        scale = np.where(scale == 0, 1.0, scale)
        return (label - self.label_min.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        results['feature'] = np.load(results['feature_path'])
        results['label'] = np.load(results['label_path'])
        if self.uses_label_scale():
            results['label_scale'] = self.build_label_scale(results['feature_path'], results['label'].shape[:2])

        results = self.pipeline(results) if self.pipeline else results

        results['label'] = self.clip_label(results['label'])
        if self.uses_label_scale():
            results['label'] = results['label'] / results['label_scale'] * self.target_scale_factor
            if self.target_clip_max is not None:
                results['label'] = np.minimum(results['label'], self.target_clip_max)

        if self.label_norm:
            results['label'] = self.normalize_label(results['label'])
        
        feature =  results['feature'].transpose(2, 0, 1).astype(np.float32)
        label = results['label'].transpose(2, 0, 1).astype(np.float32)

        return feature, label, results['label_path']

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)
