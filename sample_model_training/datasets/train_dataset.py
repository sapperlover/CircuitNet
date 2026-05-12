import os
import copy
import numpy as np
from torchvision.transforms import Compose

class TrainDataset(object):
    def __init__(self, ann_file, dataroot, pipeline=None, test_mode=False, **kwargs):
        super().__init__()
        self.ann_file = ann_file
        self.dataroot = dataroot
        self.test_mode = test_mode
        self.label_norm = kwargs.get('label_norm', True)
        if pipeline:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None

        self.data_infos = self.load_annotations()
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
            channel_min = label.min(axis=(0, 1))
            channel_max = label.max(axis=(0, 1))
            if label_min is None:
                label_min = channel_min
                label_max = channel_max
            else:
                label_min = np.minimum(label_min, channel_min)
                label_max = np.maximum(label_max, channel_max)

        return label_min.astype(np.float32), label_max.astype(np.float32)

    def normalize_label(self, label):
        scale = self.label_max - self.label_min
        scale = np.where(scale == 0, 1.0, scale)
        return (label - self.label_min.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        results['feature'] = np.load(results['feature_path'])
        results['label'] = np.load(results['label_path'])

        results = self.pipeline(results) if self.pipeline else results

        if self.label_norm:
            results['label'] = self.normalize_label(results['label'])
        
        feature =  results['feature'].transpose(2, 0, 1).astype(np.float32)
        label = results['label'].transpose(2, 0, 1).astype(np.float32)

        return feature, label, results['label_path']

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)
