import os
import copy
import numpy as np
from pathlib import Path

from .map_features import MapFeatureBuilder


class TestDataset(object):
    def __init__(self, ann_file, dataroot, test_mode=False, **kwargs):
        super().__init__()
        self.ann_file = ann_file
        self.dataroot = dataroot
        self.test_mode = test_mode
        self.power_smooth_sigma = kwargs.get('power_smooth_sigma', 5.0)
        self.map_feature_norm_stats = kwargs.get('map_feature_norm_stats', None)
        self.scalar_input_stats = self.parse_scalar_input_stats(kwargs.get('scalar_input_stats', []))
        self.scalar_norm = kwargs.get('scalar_norm', True)
        self.scalar_norm_stats = kwargs.get('scalar_norm_stats', None)
        self.map_feature_builder = MapFeatureBuilder(
            kwargs.get('map_input_features', None),
            power_smooth_sigma=self.power_smooth_sigma,
            local_window_size=kwargs.get('local_window_size', 9),
            local_z_clip=kwargs.get('local_z_clip', 5.0),
            norm_stats=self.map_feature_norm_stats,
            std_clip=kwargs.get('map_feature_std_clip', None),
        )
        if self.scalar_norm_stats is not None:
            self.scalar_norm_stats = self.normalize_scalar_norm_stats(self.scalar_norm_stats)
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                if not line.strip():
                    continue
                else:
                    feature, label, instance_count, instance_IR_drop, instance_name = line.strip().split(',')
                    if self.dataroot is not None:
                        feature_path = os.path.join(self.dataroot, feature)
                        label_path = os.path.join(self.dataroot, label)
                        instance_count_path = os.path.join(self.dataroot, instance_count)
                        instance_IR_drop_path = os.path.join(self.dataroot, instance_IR_drop)
                        instance_name_path = os.path.join(self.dataroot, instance_name)
                    data_infos.append(dict(feature_path=feature_path, label_path=label_path, instance_count_path=instance_count_path, instance_IR_drop_path=instance_IR_drop_path, instance_name_path=instance_name_path))
        return data_infos

    def parse_scalar_input_stats(self, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(',') if item.strip()]
        return list(value)

    def normalize_scalar_norm_stats(self, stats):
        mean = np.asarray(stats['mean'], dtype=np.float32)
        std = np.asarray(stats['std'], dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        return {'mean': mean.tolist(), 'std': std.tolist()}

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

    def positive_values(self, data):
        values = np.asarray(data, dtype=np.float64)
        return values[np.isfinite(values) & (values > 0)]

    def safe_log(self, value, eps=1e-30):
        return float(np.log(max(float(value), eps)))

    def log_positive_percentile(self, data, percentile):
        values = self.positive_values(data)
        if values.size == 0:
            return 0.0
        return self.safe_log(np.percentile(values, percentile))

    def load_out_feature(self, feature_path, feature_name):
        return np.load(self.resolve_out_feature_path(feature_path, feature_name))

    def build_scalar_input(self, feature_path):
        cache = {}

        def get_feature(name):
            if name not in cache:
                cache[name] = self.load_out_feature(feature_path, name)
            return cache[name]

        values = []
        for stat in self.scalar_input_stats:
            if stat == 'log_total_power_sum':
                values.append(self.safe_log(np.sum(self.positive_values(get_feature('total_power')))))
            elif stat == 'log_eff_res_VDD_p50':
                values.append(self.log_positive_percentile(get_feature('eff_res_VDD'), 50))
            elif stat == 'log_eff_res_VDD_p95':
                values.append(self.log_positive_percentile(get_feature('eff_res_VDD'), 95))
            elif stat == 'log_eff_res_VSS_p50':
                values.append(self.log_positive_percentile(get_feature('eff_res_VSS'), 50))
            elif stat == 'log_eff_res_VSS_p95':
                values.append(self.log_positive_percentile(get_feature('eff_res_VSS'), 95))
            elif stat == 'occupancy_ratio':
                total_power = get_feature('total_power')
                values.append(float(np.count_nonzero(total_power) / total_power.size) if total_power.size else 0.0)
            else:
                raise ValueError('Unsupported scalar input stat: {}'.format(stat))

        scalar = np.asarray(values, dtype=np.float32)
        if self.scalar_norm and self.scalar_norm_stats is not None:
            mean = np.asarray(self.scalar_norm_stats['mean'], dtype=np.float32)
            std = np.asarray(self.scalar_norm_stats['std'], dtype=np.float32)
            scalar = (scalar - mean) / std
        return scalar.astype(np.float32)

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        
        base_feature = np.load(results['feature_path'])
        feature = self.map_feature_builder.build(
            results['feature_path'], base_feature, self.resolve_out_feature_path
        ).transpose(2, 0, 1).astype(np.float32)
        label = np.load(results['label_path']).transpose(2, 0, 1).astype(np.float32)
        if self.scalar_input_stats:
            feature = {'map': feature, 'scalar': self.build_scalar_input(results['feature_path'])}
        return feature, label, results['instance_count_path'], results['instance_IR_drop_path'], results['instance_name_path'], results['feature_path']


    def __len__(self):
        return len(self.data_infos)


    def __getitem__(self, idx):
        return self.prepare_data(idx)
