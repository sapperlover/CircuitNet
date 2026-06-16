import argparse
import datetime
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
SAMPLE_MODEL_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(SAMPLE_MODEL_ROOT))

from test import compute_map_level_metrics, tensor_to_numpy
from test_weighted_ensemble import (
    build_bundle,
    build_logger,
    build_metric,
    design_name,
    finite_mean,
    get_single_path,
    load_base_config,
    load_instance_count,
    load_instance_name,
    match_map_shape,
    predict_ir_map,
    resize,
    resolve_path,
    str2bool,
    sync_if_needed,
    write_csv,
    write_report,
)


DEFAULT_FINAL_PATH = (
    "/home/lc/class/实践题目资料-2026春/CircuitNet/sample_model_training/"
    "work_dir/scalar_gated_resunet_e02_final_smooth_target_film_physcorr/"
    "train-20260613_075249"
)
DEFAULT_TARGET_PATH = (
    "/home/lc/class/实践题目资料-2026春/CircuitNet/sample_model_training/"
    "work_dir/scalar_gated_resunet_dualstem10_aspp_no_film_e02_alpha025_std_iraux"
)


def parse_weights(value):
    if value is None or value == "":
        return [round(i / 10.0, 4) for i in range(11)]
    weights = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        weight = float(item)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]: {}".format(weight))
        weights.append(weight)
    if not weights:
        raise ValueError("No valid weights were provided")
    return weights


def iter_number(path):
    stem = path.stem
    prefix = "model_iters_"
    if stem.startswith(prefix):
        try:
            return int(stem[len(prefix):])
        except ValueError:
            return -1
    return -1


def latest_iter_checkpoint(directory):
    candidates = list(directory.glob("model_iters_*.pth"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (iter_number(p), p.stat().st_mtime), reverse=True)[0]


def resolve_checkpoint(path_value):
    path = resolve_path(path_value)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError("Checkpoint path does not exist: {}".format(path))

    direct_min = path / "model_train_loss_min.pth"
    if direct_min.exists():
        return direct_min

    direct_latest = latest_iter_checkpoint(path)
    if direct_latest is not None:
        return direct_latest

    train_dirs = sorted([p for p in path.glob("train-*") if p.is_dir()], key=lambda p: p.name, reverse=True)
    for train_dir in train_dirs:
        train_min = train_dir / "model_train_loss_min.pth"
        if train_min.exists():
            return train_min

    latest_candidates = [latest_iter_checkpoint(train_dir) for train_dir in train_dirs]
    latest_candidates = [p for p in latest_candidates if p is not None]
    if latest_candidates:
        return sorted(latest_candidates, key=lambda p: (p.parent.name, iter_number(p), p.stat().st_mtime), reverse=True)[0]

    raise FileNotFoundError("No .pth checkpoint found under {}".format(path))


def add_value(summary, split_summary, weight, metric_name, design, value):
    if not np.isfinite(value):
        return
    summary[weight].setdefault(metric_name, []).append(float(value))
    split_summary[weight].setdefault(metric_name, {}).setdefault(design, []).append(float(value))


def mean_or_nan(values):
    if not values:
        return math.nan
    return float(np.mean(values))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch test IR-space weighted ensembles for final-target and changed-target checkpoints."
    )
    parser.add_argument("--args", "--arg_file", dest="args", default="args/test.json")
    parser.add_argument("--final-path", default=DEFAULT_FINAL_PATH, help="Final model checkpoint, train dir, or work_dir.")
    parser.add_argument("--target-path", default=DEFAULT_TARGET_PATH, help="Changed-target model checkpoint, train dir, or work_dir.")
    parser.add_argument(
        "--weights",
        default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated final-model weights. Target-model weight is 1 - final_weight.",
    )
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
    weights = parse_weights(args.weights)
    final_checkpoint = resolve_checkpoint(args.final_path)
    target_checkpoint = resolve_checkpoint(args.target_path)
    base_cfg = load_base_config(args)

    if base_cfg.get("cpu", False) or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        gpu = int(base_cfg.get("gpu", 0) or 0)
        torch.cuda.set_device(gpu)
        device = torch.device("cuda", gpu)

    if args.out_dir is None:
        run_name = "{}__{}".format(final_checkpoint.parent.name, target_checkpoint.parent.name)
        out_root = SAMPLE_MODEL_ROOT / "work_dir" / "weighted_ensemble_batch" / run_name
    else:
        out_root = resolve_path(args.out_dir)
    log_dir = out_root / "sweep-{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    logger = build_logger(log_dir)

    logger.info("Using device: %s", device)
    logger.info("Final checkpoint: %s", final_checkpoint)
    logger.info("Changed-target checkpoint: %s", target_checkpoint)
    logger.info("Final weights: %s", ", ".join("{:.4f}".format(w) for w in weights))
    logger.info("Test CSV: %s", base_cfg["ann_file"])
    logger.info("Output: %s", log_dir)

    final_bundle = build_bundle("final", final_checkpoint, base_cfg, device)
    target_bundle = build_bundle("changed_target", target_checkpoint, base_cfg, device)

    with open(log_dir / "batch_ensemble_test.json", "wt") as f:
        json.dump(
            {
                "base_cfg": base_cfg,
                "final_checkpoint": str(final_checkpoint),
                "target_checkpoint": str(target_checkpoint),
                "weights": weights,
                "final_model_cfg": final_bundle.cfg,
                "target_model_cfg": target_bundle.cfg,
            },
            f,
            indent=4,
            default=str,
        )

    metric_fns = {name: build_metric(name) for name in base_cfg.get("eval_metric", ["MAE", "corrcoef"])}
    map_metric_names = ["map_sum_MAE", "map_sum_CC", "map_flat2_CC", "map_channel_avg_CC"]
    metric_names = list(metric_fns.keys())
    if not base_cfg.get("final_test", False):
        metric_names.extend(map_metric_names)

    summary = {weight: {name: [] for name in metric_names} for weight in weights}
    split_summary = {weight: {name: {} for name in metric_names} for weight in weights}
    case_rows = []

    total = min(len(final_bundle.loader), len(target_bundle.loader))
    if args.limit is not None:
        total = min(total, args.limit)

    final_iter = iter(final_bundle.loader)
    target_iter = iter(target_bundle.loader)
    for index in tqdm(range(total), total=total):
        final_batch = next(final_iter)
        target_batch = next(target_iter)
        (
            final_feature,
            final_label,
            final_instance_count_path,
            final_instance_ir_path,
            final_instance_name_path,
            final_feature_path,
        ) = final_batch
        (
            target_feature,
            _target_label,
            _target_instance_count_path,
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
        final_ir_map = predict_ir_map(final_bundle, final_feature, final_feature_path, device)
        target_ir_map = predict_ir_map(target_bundle, target_feature, target_feature_path, device)
        sync_if_needed(device)

        target_ir_map = match_map_shape(target_ir_map, final_ir_map.shape[1:])
        instance_count = load_instance_count(final_instance_count_path, base_cfg.get("final_test", False))
        file_name = os.path.splitext(os.path.basename(final_instance_ir_path))[0]
        design = design_name(final_instance_ir_path)
        instance_ir_drop = None if base_cfg.get("final_test", False) else np.load(final_instance_ir_path)
        label_map = None
        if not base_cfg.get("final_test", False):
            label_map = tensor_to_numpy(final_label)
            if label_map.ndim == 4:
                label_map = label_map[0]

        instance_name = None
        if base_cfg.get("save_report", False):
            instance_name = load_instance_name(final_instance_name_path, base_cfg.get("final_test", False))

        logger.info("#%d %s", index + 1, file_name)
        for final_weight in weights:
            target_weight = 1.0 - final_weight
            ensemble_ir_map = final_weight * final_ir_map + target_weight * target_ir_map
            pred_vdd_drop = resize(ensemble_ir_map[0], instance_count.shape)
            pred_gnd_bounce = resize(ensemble_ir_map[1], instance_count.shape)
            pred_instance_vdd = np.repeat(pred_vdd_drop.ravel(), instance_count.ravel())
            pred_instance_vss = np.repeat(pred_gnd_bounce.ravel(), instance_count.ravel())
            pred_instance_ir = pred_instance_vdd + pred_instance_vss

            if base_cfg.get("save_report", False):
                weight_dir = log_dir / "reports_w{:.4f}".format(final_weight).replace(".", "p")
                write_report(weight_dir, file_name, pred_instance_vdd, pred_instance_vss, instance_name)

            row = {
                "case": file_name,
                "design": design,
                "final_weight": final_weight,
                "target_weight": target_weight,
            }

            if not base_cfg.get("final_test", False):
                for metric_name, metric_func in metric_fns.items():
                    result = float(metric_func(instance_ir_drop, pred_instance_ir))
                    row[metric_name] = result
                    add_value(summary, split_summary, final_weight, metric_name, design, result)
                map_values = compute_map_level_metrics(ensemble_ir_map, label_map)
                for metric_name, result in map_values.items():
                    row[metric_name] = result
                    add_value(summary, split_summary, final_weight, metric_name, design, result)
            case_rows.append(row)

    summary_rows = []
    best_rows = []
    if not base_cfg.get("final_test", False):
        for final_weight in weights:
            for metric_name in metric_names:
                values = summary[final_weight].get(metric_name, [])
                summary_rows.append(
                    {
                        "final_weight": final_weight,
                        "target_weight": 1.0 - final_weight,
                        "design": "ALL",
                        "metric": metric_name,
                        "avg": mean_or_nan(values),
                        "finite_avg": finite_mean(values),
                        "count": len(values),
                    }
                )
                for design, design_values in split_summary[final_weight].get(metric_name, {}).items():
                    summary_rows.append(
                        {
                            "final_weight": final_weight,
                            "target_weight": 1.0 - final_weight,
                            "design": design,
                            "metric": metric_name,
                            "avg": mean_or_nan(design_values),
                            "finite_avg": finite_mean(design_values),
                            "count": len(design_values),
                        }
                    )

        grouped = {}
        for row in summary_rows:
            if row["count"] <= 0 or not np.isfinite(row["finite_avg"]):
                continue
            grouped.setdefault((row["design"], row["metric"]), []).append(row)
        for (design, metric_name), rows in grouped.items():
            reverse = "MAE" not in metric_name
            best = sorted(rows, key=lambda r: r["finite_avg"], reverse=reverse)[0]
            best_rows.append(dict(best))

        for row in summary_rows:
            if row["design"] == "ALL":
                logger.info(
                    "w_final=%.4f w_target=%.4f Avg. %s: %.6f",
                    row["final_weight"],
                    row["target_weight"],
                    row["metric"],
                    row["finite_avg"],
                )

    if case_rows:
        case_fields = ["case", "design", "final_weight", "target_weight", *metric_names]
        write_csv(log_dir / "case_metrics.csv", case_rows, case_fields)
    if summary_rows:
        fields = ["final_weight", "target_weight", "design", "metric", "avg", "finite_avg", "count"]
        write_csv(log_dir / "summary_metrics.csv", summary_rows, fields)
        write_csv(log_dir / "best_by_metric.csv", best_rows, fields)

    logger.info("Batch weighted ensemble test finished. Results saved in %s", log_dir)


if __name__ == "__main__":
    main()
