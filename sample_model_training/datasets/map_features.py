import numpy as np
from scipy import ndimage


def parse_map_input_features(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    return list(value)


class MapFeatureBuilder:
    def __init__(
        self,
        feature_names=None,
        power_smooth_sigma=5.0,
        local_window_size=9,
        local_z_clip=5.0,
        norm_stats=None,
        std_clip=None,
        log_eps=1e-30,
    ):
        self.feature_names = parse_map_input_features(feature_names)
        self.power_smooth_sigma = float(power_smooth_sigma)
        self.local_window_size = int(local_window_size)
        self.local_z_clip = float(local_z_clip)
        self.norm_stats = self.normalize_norm_stats(norm_stats)
        self.std_clip = std_clip
        self.log_eps = float(log_eps)

    def enabled(self):
        return bool(self.feature_names)

    def normalize_norm_stats(self, stats):
        if stats is None:
            return None
        normalized = {}
        for key, value in stats.items():
            std = float(value.get('std', 1.0))
            if std < 1e-6:
                std = 1.0
            normalized[key] = {'mean': float(value.get('mean', 0.0)), 'std': std}
        return normalized

    def set_norm_stats(self, stats):
        self.norm_stats = self.normalize_norm_stats(stats)

    def requires_norm_stats(self):
        return [name for name in self.feature_names if name.strip() in {
            'log_smooth_power_eff_res_VDD_std',
            'log_smooth_power_eff_res_VSS_std',
        }]

    def resize_map(self, data, out_shape, order=1):
        if data.shape == tuple(out_shape):
            return data.astype(np.float32)
        zoom = (out_shape[0] / data.shape[0], out_shape[1] / data.shape[1])
        return ndimage.zoom(data, zoom, order=order).astype(np.float32)

    def finite_nonnegative(self, data):
        data = np.asarray(data, dtype=np.float32)
        data = np.where(np.isfinite(data), data, 0.0)
        return np.maximum(data, 0.0)

    def minmax_norm(self, data):
        data = np.asarray(data, dtype=np.float32)
        finite = np.isfinite(data)
        out = np.zeros_like(data, dtype=np.float32)
        if not np.any(finite):
            return out
        values = data[finite]
        data_min = float(values.min())
        data_max = float(values.max())
        denom = data_max - data_min
        if denom < 1e-12:
            return out
        out[finite] = (data[finite] - data_min) / denom
        return out.astype(np.float32)

    def log_positive_minmax(self, data):
        data = self.finite_nonnegative(data)
        positive = data > 0
        out = np.zeros_like(data, dtype=np.float32)
        if not np.any(positive):
            return out
        logs = np.log(np.maximum(data[positive], self.log_eps)).astype(np.float32)
        log_min = float(logs.min())
        log_max = float(logs.max())
        denom = log_max - log_min
        if denom < 1e-12:
            return out
        out[positive] = (logs - log_min) / denom
        return out.astype(np.float32)

    def log_positive_map(self, data):
        data = self.finite_nonnegative(data)
        positive = data > 0
        out = np.zeros_like(data, dtype=np.float32)
        out[positive] = np.log(np.maximum(data[positive], self.log_eps)).astype(np.float32)
        return out

    def standardize(self, name, data):
        if self.norm_stats is None or name not in self.norm_stats:
            raise ValueError('Missing map feature norm stats for {}'.format(name))
        stats = self.norm_stats[name]
        out = (data.astype(np.float32) - stats['mean']) / stats['std']
        if self.std_clip is not None:
            clip = float(self.std_clip)
            if clip > 0:
                out = np.clip(out, -clip, clip)
        return out.astype(np.float32)

    def local_occupancy(self, total_power_raw, out_shape):
        mask = (self.finite_nonnegative(total_power_raw) > 0).astype(np.float32)
        mask = self.resize_map(mask, out_shape, order=0)
        return ndimage.uniform_filter(mask, size=self.local_window_size, mode='nearest').astype(np.float32)

    def local_z_log_positive(self, data):
        data = self.finite_nonnegative(data)
        positive = data > 0
        logs = np.zeros_like(data, dtype=np.float32)
        logs[positive] = np.log(np.maximum(data[positive], self.log_eps)).astype(np.float32)
        mask = positive.astype(np.float32)

        local_weight = ndimage.uniform_filter(mask, size=self.local_window_size, mode='nearest')
        local_sum = ndimage.uniform_filter(logs * mask, size=self.local_window_size, mode='nearest')
        local_sq_sum = ndimage.uniform_filter(logs * logs * mask, size=self.local_window_size, mode='nearest')
        denom = np.maximum(local_weight, 1e-6)
        local_mean = local_sum / denom
        local_var = np.maximum(local_sq_sum / denom - local_mean * local_mean, 0.0)
        local_std = np.sqrt(local_var + 1e-6)

        z_map = np.zeros_like(data, dtype=np.float32)
        valid = positive & (local_weight > 1e-6)
        z_map[valid] = (logs[valid] - local_mean[valid]) / local_std[valid]
        if self.local_z_clip > 0:
            z_map = np.clip(z_map, -self.local_z_clip, self.local_z_clip) / self.local_z_clip
        return z_map.astype(np.float32)

    def raw_named_feature(self, name, feature_path, base_feature, resolve_out_feature_path, cache=None):
        base_feature = np.asarray(base_feature, dtype=np.float32)
        out_shape = base_feature.shape[:2]
        if cache is None:
            cache = {}

        def raw_feature(feature_name):
            if feature_name not in cache:
                raw = np.load(resolve_out_feature_path(feature_path, feature_name)).astype(np.float32)
                cache[feature_name] = self.resize_map(self.finite_nonnegative(raw), out_shape)
            return cache[feature_name]

        def smooth_power():
            if 'smooth_power' not in cache:
                cache['smooth_power'] = ndimage.gaussian_filter(
                    raw_feature('total_power'), sigma=self.power_smooth_sigma, mode='nearest'
                ).astype(np.float32)
            return cache['smooth_power']

        key = name.strip()
        if key in ('total_power_norm', 'relative_total_power_norm', 'base_0'):
            return base_feature[:, :, 0]
        if key in ('eff_res_VDD_norm', 'relative_eff_res_VDD_norm', 'base_1'):
            return base_feature[:, :, 1]
        if key in ('eff_res_VSS_norm', 'relative_eff_res_VSS_norm', 'base_2'):
            return base_feature[:, :, 2]
        if key == 'smooth_power_norm':
            return self.minmax_norm(smooth_power())
        if key == 'local_occupancy_map':
            total_power_path = resolve_out_feature_path(feature_path, 'total_power')
            total_power_raw = np.load(total_power_path).astype(np.float32)
            return self.local_occupancy(total_power_raw, out_shape)
        if key == 'local_z_power':
            return self.local_z_log_positive(raw_feature('total_power'))
        if key == 'local_z_eff_res_VDD':
            return self.local_z_log_positive(raw_feature('eff_res_VDD'))
        if key == 'local_z_eff_res_VSS':
            return self.local_z_log_positive(raw_feature('eff_res_VSS'))
        if key in ('log_smooth_power_eff_res_VDD', 'log(smooth_power_raw*eff_res_VDD_raw)'):
            return self.log_positive_minmax(smooth_power() * raw_feature('eff_res_VDD'))
        if key in ('log_smooth_power_eff_res_VSS', 'log(smooth_power_raw*eff_res_VSS_raw)'):
            return self.log_positive_minmax(smooth_power() * raw_feature('eff_res_VSS'))
        if key == 'log_smooth_power_eff_res_VDD_std':
            return self.log_positive_map(smooth_power() * raw_feature('eff_res_VDD'))
        if key == 'log_smooth_power_eff_res_VSS_std':
            return self.log_positive_map(smooth_power() * raw_feature('eff_res_VSS'))
        raise ValueError('Unsupported map input feature: {}'.format(name))

    def compute_norm_stats(self, feature_paths, load_base_feature, resolve_out_feature_path):
        stat_names = self.requires_norm_stats()
        if not stat_names:
            return None

        accum = {name: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0} for name in stat_names}
        for feature_path in feature_paths:
            base_feature = load_base_feature(feature_path)
            cache = {}
            for name in stat_names:
                data = self.raw_named_feature(name, feature_path, base_feature, resolve_out_feature_path, cache)
                values = data[np.isfinite(data)].astype(np.float64)
                if values.size == 0:
                    continue
                accum[name]['sum'] += float(values.sum())
                accum[name]['sum_sq'] += float(np.square(values).sum())
                accum[name]['count'] += int(values.size)

        stats = {}
        for name, item in accum.items():
            count = max(item['count'], 1)
            mean = item['sum'] / count
            var = max(item['sum_sq'] / count - mean * mean, 0.0)
            std = max(float(np.sqrt(var)), 1e-6)
            stats[name] = {'mean': float(mean), 'std': std}
        return stats

    def build(self, feature_path, base_feature, resolve_out_feature_path):
        base_feature = np.asarray(base_feature, dtype=np.float32)
        if not self.enabled():
            return base_feature

        channels = []
        cache = {}
        for name in self.feature_names:
            key = name.strip()
            raw_feature = self.raw_named_feature(name, feature_path, base_feature, resolve_out_feature_path, cache)
            if key in ('log_smooth_power_eff_res_VDD_std', 'log_smooth_power_eff_res_VSS_std'):
                channels.append(self.standardize(key, raw_feature))
            else:
                channels.append(raw_feature)

        return np.stack(channels, axis=2).astype(np.float32)
