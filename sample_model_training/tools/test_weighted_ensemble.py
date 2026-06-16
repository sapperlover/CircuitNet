import argparse
import csv
import datetime
import gzip
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
SAMPLE_MODEL_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SAMPLE_MODEL_ROOT.parent

sys.path.insert(0, str(SAMPLE_MODEL_ROOT))

from datasets.build_dataset import build_dataset  # noqa: E402
import utils.metrics as metrics  # noqa: E402
from models.build_model import build_model  # noqa: E402
from test import (  # noqa: E402
    build_label_scale,
    denormalize_channels,
    load_label_norm_stats,
    load_train_config,
    move_to_device,
    resize,
)


CHECKPOINT_CONFIG_KEYS = [
    "model_type",
    "in_channels",
    "out_channels",
    "base_channels",
    "norm_type",
    "negative_slope",
    "scalar_channels",
    "scalar_hidden_channels",
    "scalar_embedding_channels",
    "scale_log_clamp",
    "require_scalar",
    "use_scalar_film",
    "scalar_film_layers",
    "scalar_film_strength",
    "use_aspp",
    "aspp_dilations",
    "aspp_branch_channels",
    "aspp_res_scale",
    "use_dual_stem",
    "use_dual_head",
    "dual_head_gate_min",
    "dual_head_gate_max",
    "dual_head_gate_init",
    "relative_in_channels",
    "physics_in_channels",
    "map_input_features",
    "local_window_size",
    "local_z_clip",
    "scalar_input_stats",
    "scalar_norm",
    "scalar_norm_stats",
    "out_activation",
    "label_norm",
    "label_norm_stats",
    "label_scale_mode",
    "aux_label_scale_mode",
    "effres_alpha",
    "aux_effres_alpha",
    "power_smooth_sigma",
    "effres_smooth_sigma",
    "power_epsilon_mode",
    "power_epsilon_ratio",
    "target_scale_factor",
    "power_epsilon",
    "map_feature_norm_stats",
    "map_feature_std_clip",
    "test_output_mode",
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def resolve_path(path, base=SAMPLE_MODEL_ROOT):
    if path is None:
        return None
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    for candidate in (Path.cwd() / path, base / path, REPO_ROOT / path):
        if candidate.exists():
            return candidate.resolve()
    return (base / path).resolve()


def read_json(path):
    with open(path, "rt") as f:
        return json.load(f)


def build_metric(metric_name):
    return metrics.__dict__[metric_name.lower()]


def design_name(path):
    name = os.path.basename(path)
    if "Vortex-small" in name:
        return "Vortex-small"
    if "nvdla-small" in name:
        return "nvdla-small"
    if "FPU" in name:
        return "RISCY-FPU"
    return name.split("_")[0]


def build_logger(log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("weighted_ensemble_test")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_dir / "test.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def load_base_config(args):
    cfg = {
        "gpu": 0,
        "cpu": False,
        "ann_file": str(SAMPLE_MODEL_ROOT / "../feature_extraction/test.csv"),
        "dataroot": str(SAMPLE_MODEL_ROOT / "../feature_extraction"),
        "dataset_type": "TestDataset",
        "eval_metric": ["MAE", "corrcoef"],
        "final_test": False,
        "save_report": False,
        "plot": False,
        "test_mode": True,
    }
    if args.args is not None:
        args_path = resolve_path(args.args)
        cfg.update(read_json(args_path))
        cfg["args"] = str(args_path)

    if args.ann_file is not None:
        cfg["ann_file"] = args.ann_file
    if args.dataroot is not None:
        cfg["dataroot"] = args.dataroot
    if args.gpu is not None:
        cfg["gpu"] = args.gpu
    if args.cpu:
        cfg["cpu"] = True
    else:
        cfg["cpu"] = bool(cfg.get("cpu", False))
    if args.final_test is not None:
        cfg["final_test"] = args.final_test
    if args.save_report is not None:
        cfg["save_report"] = args.save_report

    cfg["test_mode"] = True
    cfg["dataset_type"] = "TestDataset"
    cfg["ann_file"] = str(resolve_path(cfg["ann_file"]))
    cfg["dataroot"] = str(resolve_path(cfg["dataroot"]))
    return cfg


def merge_checkpoint_config(base_cfg, checkpoint_path):
    cfg = dict(base_cfg)
    train_config = load_train_config(str(checkpoint_path))
    for key in CHECKPOINT_CONFIG_KEYS:
        if key in train_config:
            cfg[key] = train_config[key]
    cfg["pretrained"] = str(checkpoint_path)
    cfg["test_mode"] = True
    cfg["dataset_type"] = "TestDataset"
    cfg["ann_file"] = base_cfg["ann_file"]
    cfg["dataroot"] = base_cfg["dataroot"]
    cfg["cpu"] = base_cfg.get("cpu", False)
    cfg["gpu"] = base_cfg.get("gpu", None)

    label_norm_stats = cfg.get("label_norm_stats", None)
    if label_norm_stats is None and cfg.get("label_norm", True):
        label_norm_stats = load_label_norm_stats(str(checkpoint_path))
    if label_norm_stats is not None:
        cfg["label_norm_stats"] = label_norm_stats
    return cfg, label_norm_stats


@dataclass
class ModelBundle:
    name: str
    checkpoint: Path
    cfg: dict
    label_norm_stats: dict
    model: torch.nn.Module
    loader: object


def build_bundle(name, checkpoint_path, base_cfg, device):
    cfg, label_norm_stats = merge_checkpoint_config(base_cfg, checkpoint_path)
    model_args = dict(cfg)
    model = build_model(model_args)
    model.to(device)
    model.eval()
    loader = build_dataset(dict(cfg))
    return ModelBundle(name, checkpoint_path, cfg, label_norm_stats, model, loader)


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_ir_map(bundle, feature, feature_path, device):
    model_input = move_to_device(feature, device)
    with torch.no_grad():
        output = bundle.model(model_input)[0].detach().cpu().numpy()
    output = denormalize_channels(output, bundle.label_norm_stats)
    label_scale = build_label_scale(feature_path, output.shape[1:], bundle.cfg)
    if label_scale is not None:
        output = output / bundle.cfg.get("target_scale_factor", 1.0) * label_scale.transpose(2, 0, 1)
    return output.astype(np.float32)


def match_map_shape(data, out_shape):
    if data.shape[1:] == tuple(out_shape):
        return data
    return np.stack([resize(data[channel], out_shape) for channel in range(data.shape[0])], axis=0).astype(np.float32)


def get_single_path(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def load_instance_count(path, final_test):
    if final_test:
        path = path.replace("instance_count", "instance_count_from_power_rpt")
    return np.load(path).astype(int)


def load_instance_name(path, final_test):
    if final_test:
        path = path.replace("instance_name", "instance_name_from_power_rpt")
    return np.load(path)["instance_name"]


def write_report(log_dir, file_name, pred_instance_vdd, pred_instance_vss, instance_name):
    report_dir = log_dir / "pred_static_ir_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "pred_static_ir_{}.gz".format(file_name)
    with gzip.open(report_path, "wt") as f:
        for vdd, vss, name in zip(pred_instance_vdd, pred_instance_vss, instance_name):
            f.write("{} {}\n".format(vdd + vss, name))


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else math.nan


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Weighted ensemble test for final-target and changed-target checkpoints.")
    parser.add_argument("--args", "--arg_file", dest="args", default="args/test.json")
    parser.add_argument("--final-pretrained", required=True, help="Checkpoint for the final model.")
    parser.add_argument("--target-pretrained", required=True, help="Checkpoint for the changed-target model.")
    parser.add_argument("--final-weight", type=float, default=0.5, help="Weight for final model prediction. Target weight is 1 - final_weight.")
    parser.add_argument("--ann_file", default=None)
    parser.add_argument("--dataroot", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--final_test", type=str2bool, default=None)
    parser.add_argument("--save_report", type=str2bool, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.final_weight <= 1.0:
        raise ValueError("--final-weight must be in [0, 1]")
    target_weight = 1.0 - args.final_weight

    final_checkpoint = resolve_path(args.final_pretrained)
    target_checkpoint = resolve_path(args.target_pretrained)
    base_cfg = load_base_config(args)

    if base_cfg.get("cpu", False) or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        gpu = int(base_cfg.get("gpu", 0) or 0)
        torch.cuda.set_device(gpu)
        device = torch.device("cuda", gpu)

    if args.out_dir is None:
        ensemble_name = "{}__{}__w{:.2f}".format(
            final_checkpoint.parent.name,
            target_checkpoint.parent.name,
            args.final_weight,
        )
        out_root = SAMPLE_MODEL_ROOT / "work_dir" / "weighted_ensemble" / ensemble_name
    else:
        out_root = resolve_path(args.out_dir)
    log_dir = out_root / "test-{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    logger = build_logger(log_dir)

    logger.info("Using device: %s", device)
    logger.info("Final checkpoint: %s", final_checkpoint)
    logger.info("Changed-target checkpoint: %s", target_checkpoint)
    logger.info("Weights: final=%.4f, changed_target=%.4f", args.final_weight, target_weight)
    logger.info("Test CSV: %s", base_cfg["ann_file"])
    logger.info("Output: %s", log_dir)

    final_bundle = build_bundle("final", final_checkpoint, base_cfg, device)
    target_bundle = build_bundle("changed_target", target_checkpoint, base_cfg, device)

    with open(log_dir / "ensemble_test.json", "wt") as f:
        json.dump(
            {
                "base_cfg": base_cfg,
                "final_checkpoint": str(final_checkpoint),
                "target_checkpoint": str(target_checkpoint),
                "final_weight": args.final_weight,
                "target_weight": target_weight,
                "final_model_cfg": final_bundle.cfg,
                "target_model_cfg": target_bundle.cfg,
            },
            f,
            indent=4,
            default=str,
        )

    metric_fns = {name: build_metric(name) for name in base_cfg.get("eval_metric", ["MAE", "corrcoef"])}
    metric_sums = {name: 0.0 for name in metric_fns}
    metric_values = {name: [] for name in metric_fns}
    split_metrics = {name: {} for name in metric_fns}
    case_rows = []

    total = min(len(final_bundle.loader), len(target_bundle.loader))
    if args.limit is not None:
        total = min(total, args.limit)

    final_iter = iter(final_bundle.loader)
    target_iter = iter(target_bundle.loader)
    processed = 0
    for index in tqdm(range(total), total=total):
        final_batch = next(final_iter)
        target_batch = next(target_iter)
        (
            final_feature,
            _final_label,
            final_instance_count_path,
            final_instance_ir_path,
            final_instance_name_path,
            final_feature_path,
        ) = final_batch
        (
            target_feature,
            _target_label,
            target_instance_count_path,
            target_instance_ir_path,
            _target_instance_name_path,
            target_feature_path,
        ) = target_batch

        final_feature_path = get_single_path(final_feature_path)
        target_feature_path = get_single_path(target_feature_path)
        final_instance_count_path = get_single_path(final_instance_count_path)
        final_instance_ir_path = get_single_path(final_instance_ir_path)
        final_instance_name_path = get_single_path(final_instance_name_path)
        target_instance_ir_path = get_single_path(target_instance_ir_path)
        if os.path.abspath(final_feature_path) != os.path.abspath(target_feature_path):
            raise ValueError("Dataset order mismatch: {} vs {}".format(final_feature_path, target_feature_path))
        if os.path.abspath(final_instance_ir_path) != os.path.abspath(target_instance_ir_path):
            raise ValueError("Label order mismatch: {} vs {}".format(final_instance_ir_path, target_instance_ir_path))

        sync_if_needed(device)
        start_time = time.time()
        final_ir_map = predict_ir_map(final_bundle, final_feature, final_feature_path, device)
        target_ir_map = predict_ir_map(target_bundle, target_feature, target_feature_path, device)
        sync_if_needed(device)
        end_time = time.time()

        target_ir_map = match_map_shape(target_ir_map, final_ir_map.shape[1:])
        ensemble_ir_map = args.final_weight * final_ir_map + target_weight * target_ir_map

        instance_count = load_instance_count(final_instance_count_path, base_cfg.get("final_test", False))
        pred_vdd_drop = resize(ensemble_ir_map[0], instance_count.shape)
        pred_gnd_bounce = resize(ensemble_ir_map[1], instance_count.shape)
        pred_instance_vdd = np.repeat(pred_vdd_drop.ravel(), instance_count.ravel())
        pred_instance_vss = np.repeat(pred_gnd_bounce.ravel(), instance_count.ravel())
        pred_instance_ir = pred_instance_vdd + pred_instance_vss

        file_name = os.path.splitext(os.path.basename(final_instance_ir_path))[0]
        logger.info("#%d %s, ensemble inference time %.4fs", index + 1, file_name, end_time - start_time)

        if base_cfg.get("save_report", False):
            instance_name = load_instance_name(final_instance_name_path, base_cfg.get("final_test", False))
            write_report(log_dir, file_name, pred_instance_vdd, pred_instance_vss, instance_name)

        row = {
            "case": file_name,
            "design": design_name(final_instance_ir_path),
            "final_weight": args.final_weight,
            "target_weight": target_weight,
        }

        if not base_cfg.get("final_test", False):
            instance_ir_drop = np.load(final_instance_ir_path)
            for metric_name, metric_func in metric_fns.items():
                result = float(metric_func(instance_ir_drop, pred_instance_ir))
                logger.info("%s: %s", metric_name, result)
                metric_sums[metric_name] += result
                metric_values[metric_name].append(result)
                split_metrics[metric_name].setdefault(row["design"], [0.0, 0])
                split_metrics[metric_name][row["design"]][0] += result
                split_metrics[metric_name][row["design"]][1] += 1
                row[metric_name] = result
        case_rows.append(row)
        processed += 1

    summary_rows = []
    if not base_cfg.get("final_test", False) and processed > 0:
        for metric_name in metric_fns:
            avg_value = metric_sums[metric_name] / processed
            finite_avg = finite_mean(metric_values[metric_name])
            logger.info("===> Avg. %s: %.4f", metric_name, avg_value)
            summary_rows.append({"design": "ALL", "metric": metric_name, "avg": avg_value, "finite_avg": finite_avg, "count": processed})
        for metric_name, design_values in split_metrics.items():
            if len(design_values) <= 1:
                continue
            for name, (total_value, count) in design_values.items():
                avg_value = total_value / max(count, 1)
                logger.info("===> %s %s: %.4f", name, metric_name, avg_value)
                summary_rows.append({"design": name, "metric": metric_name, "avg": avg_value, "finite_avg": avg_value, "count": count})

    if case_rows:
        case_fields = ["case", "design", "final_weight", "target_weight", *metric_fns.keys()]
        write_csv(log_dir / "case_metrics.csv", case_rows, case_fields)
    if summary_rows:
        write_csv(log_dir / "summary_metrics.csv", summary_rows, ["design", "metric", "avg", "finite_avg", "count"])

    if base_cfg.get("save_report", False):
        logger.info("Predicted static_ir report saved in %s.", log_dir / "pred_static_ir_report")
    logger.info("Weighted ensemble test finished. Results saved in %s", log_dir)


if __name__ == "__main__":
    main()
