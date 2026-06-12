import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.stats import spearmanr


SCRIPT_PATH = Path(__file__).resolve()
SAMPLE_MODEL_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SAMPLE_MODEL_ROOT.parent
DATA_ROOT = REPO_ROOT / "feature_extraction"

sys.path.insert(0, str(SAMPLE_MODEL_ROOT))

from models.build_model import build_model  # noqa: E402


INPUT_STATS = [
    "log_total_power_sum",
    "log_total_power_p50",
    "log_total_power_p95",
    "log_eff_res_VDD_p50",
    "log_eff_res_VDD_p95",
    "log_eff_res_VSS_p50",
    "log_eff_res_VSS_p95",
    "log_instance_count",
    "occupancy_ratio",
]


def resolve_path(path, base=None):
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = []
    if base is not None:
        candidates.append(Path(base) / path)
    candidates.extend([Path.cwd() / path, SAMPLE_MODEL_ROOT / path, REPO_ROOT / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def read_json(path):
    with open(path, "rt") as f:
        return json.load(f)


def resize_map(data, out_shape, order):
    if data.shape == tuple(out_shape):
        return data
    return ndimage.zoom(data, (out_shape[0] / data.shape[0], out_shape[1] / data.shape[1]), order=order)


def resolve_out_feature_path(feature_path, feature_name):
    path = Path(feature_path)
    parts = path.parts
    if "training_set" not in parts:
        raise ValueError(f"feature path does not contain training_set: {feature_path}")
    index = parts.index("training_set")
    tech = parts[index + 1]
    root = Path(*parts[:index]) if index > 0 else Path(".")
    return root / "out" / tech / "features" / feature_name / path.name


def load_smooth_feature(feature_path, feature_name, out_shape, sigma):
    feature_map = np.load(resolve_out_feature_path(feature_path, feature_name)).astype(np.float32)
    feature_map = resize_map(feature_map, out_shape, order=1)
    feature_map = np.maximum(feature_map, 0.0)
    return ndimage.gaussian_filter(feature_map, sigma=sigma, mode="nearest")


def build_scale_base(feature_path, out_shape, cfg):
    mode = cfg.get("label_scale_mode", "none")
    if mode in (None, "none"):
        return None
    if mode == "smooth_power":
        return load_smooth_feature(
            feature_path,
            "total_power",
            out_shape,
            cfg.get("power_smooth_sigma", 5.0),
        )[:, :, None]
    if mode == "power_effres":
        power = load_smooth_feature(
            feature_path,
            "total_power",
            out_shape,
            cfg.get("power_smooth_sigma", 5.0),
        )
        eff_vdd = load_smooth_feature(
            feature_path,
            "eff_res_VDD",
            out_shape,
            cfg.get("effres_smooth_sigma", 5.0),
        )
        eff_vss = load_smooth_feature(
            feature_path,
            "eff_res_VSS",
            out_shape,
            cfg.get("effres_smooth_sigma", 5.0),
        )
        return np.stack([power * eff_vdd, power * eff_vss], axis=2)
    raise ValueError(f"Unsupported label_scale_mode: {mode}")


def build_label_scale(feature_path, out_shape, cfg):
    scale_base = build_scale_base(feature_path, out_shape, cfg)
    if scale_base is None:
        return None

    power_epsilon = cfg.get("power_epsilon")
    if power_epsilon is None:
        epsilon = np.maximum(scale_base.max(axis=(0, 1)) * cfg.get("power_epsilon_ratio", 0.01), 1e-12)
    else:
        epsilon = np.asarray(power_epsilon, dtype=np.float32)
    return (scale_base + epsilon.reshape(1, 1, -1)).astype(np.float32)


def denormalize_channels(data, stats):
    if stats is None:
        return data
    label_min = np.asarray(stats["min"], dtype=np.float32)
    label_max = np.asarray(stats["max"], dtype=np.float32)
    scale = label_max - label_min
    scale = np.where(scale == 0, 1.0, scale)
    return data * scale.reshape((-1, 1, 1)) + label_min.reshape((-1, 1, 1))


def design_name(path):
    name = Path(path).name
    if "Vortex-small" in name:
        return "Vortex-small"
    if "nvdla-small" in name:
        return "nvdla-small"
    if "FPU" in name:
        return "RISCY-FPU"
    return name.split("_")[0]


def tech_name(feature_path):
    parts = Path(feature_path).parts
    if "training_set" in parts:
        return parts[parts.index("training_set") + 1]
    return "unknown"


def process_name(path):
    match = re.search(r"_fi_([^_]+)_", Path(path).name)
    return match.group(1) if match else "unknown"


def safe_log(value, eps=1e-30):
    return float(np.log(max(float(value), eps)))


def positive_values(data):
    values = np.asarray(data, dtype=np.float64)
    return values[np.isfinite(values) & (values > 0)]


def log_positive_percentile(data, percentile):
    values = positive_values(data)
    if values.size == 0:
        return math.nan
    return safe_log(np.percentile(values, percentile))


def input_stats(row):
    total_power = np.load(resolve_out_feature_path(row["feature"], "total_power"))
    eff_vdd = np.load(resolve_out_feature_path(row["feature"], "eff_res_VDD"))
    eff_vss = np.load(resolve_out_feature_path(row["feature"], "eff_res_VSS"))
    instance_count = np.load(row["instance_count"])
    return {
        "log_total_power_sum": safe_log(np.sum(positive_values(total_power))),
        "log_total_power_p50": log_positive_percentile(total_power, 50),
        "log_total_power_p95": log_positive_percentile(total_power, 95),
        "log_eff_res_VDD_p50": log_positive_percentile(eff_vdd, 50),
        "log_eff_res_VDD_p95": log_positive_percentile(eff_vdd, 95),
        "log_eff_res_VSS_p50": log_positive_percentile(eff_vss, 50),
        "log_eff_res_VSS_p95": log_positive_percentile(eff_vss, 95),
        "log_instance_count": safe_log(np.sum(instance_count)),
        "occupancy_ratio": float(np.count_nonzero(total_power) / total_power.size) if total_power.size else math.nan,
    }


def load_rows(csv_path, dataroot, design_filter):
    rows = []
    with open(csv_path, newline="") as f:
        for parts in csv.reader(f):
            if not parts:
                continue
            if len(parts) == 2:
                feature, label = parts
                tech = tech_name(feature)
                name = Path(feature).name
                instance_count = f"./out/{tech}/features/instance_count/{name}"
                instance_ir_drop = f"./out/{tech}/features/instance_IR_drop/{name}"
                instance_name = f"./out/{tech}/features/instance_name/{Path(name).with_suffix('.npz').name}"
            else:
                feature, label, instance_count, instance_ir_drop, instance_name = parts[:5]

            row = {
                "feature": dataroot / feature,
                "label": dataroot / label,
                "instance_count": dataroot / instance_count,
                "instance_ir_drop": dataroot / instance_ir_drop,
                "instance_name": dataroot / instance_name,
                "case": Path(feature).name,
                "design": design_name(instance_ir_drop),
                "tech": tech_name(feature),
                "process": process_name(feature),
            }
            if design_filter and row["design"] != design_filter:
                continue
            rows.append(row)
    return rows


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint_path = resolve_path(checkpoint_path, SAMPLE_MODEL_ROOT)
    train_config_path = checkpoint_path.parent / "train.json"
    cfg = read_json(train_config_path)
    cfg = dict(cfg)
    cfg["test_mode"] = True
    cfg["pretrained"] = str(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    label_norm_stats = cfg.get("label_norm_stats")
    if label_norm_stats is None and cfg.get("label_norm", True):
        label_norm_stats = checkpoint.get("label_norm_stats") if isinstance(checkpoint, dict) else None

    model_args = dict(cfg)
    model = build_model(model_args)
    model.to(device)
    model.eval()
    return model, cfg, label_norm_stats, checkpoint_path


def build_model_input(feature, row, cfg, device):
    feature_tensor = torch.from_numpy(feature[None]).to(device)
    scalar_names = cfg.get("scalar_input_stats") or []
    if not scalar_names:
        return feature_tensor

    stats = input_stats(row)
    scalar = np.asarray([stats[name] for name in scalar_names], dtype=np.float32)
    if cfg.get("scalar_norm", True) and cfg.get("scalar_norm_stats") is not None:
        norm_stats = cfg["scalar_norm_stats"]
        mean = np.asarray(norm_stats["mean"], dtype=np.float32)
        std = np.asarray(norm_stats["std"], dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        scalar = (scalar - mean) / std
    return {"map": feature_tensor, "scalar": torch.from_numpy(scalar[None]).to(device)}


def predict_instance_ir(model, cfg, label_norm_stats, row, device):
    feature = np.load(row["feature"]).transpose(2, 0, 1).astype(np.float32)
    with torch.no_grad():
        output = model(build_model_input(feature, row, cfg, device))[0].detach().cpu().numpy()

    output = denormalize_channels(output, label_norm_stats)
    label_scale = build_label_scale(row["feature"], output.shape[1:], cfg)
    if label_scale is not None:
        output = output / cfg.get("target_scale_factor", 1.0) * label_scale.transpose(2, 0, 1)

    instance_count = np.load(row["instance_count"]).astype(int)
    pred_vdd = resize_map(output[0], instance_count.shape, order=3)
    pred_vss = resize_map(output[1], instance_count.shape, order=3)
    return np.repeat((pred_vdd + pred_vss).ravel(), instance_count.ravel()).astype(np.float32)


def corrcoef(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if gt.size < 2 or np.std(gt) == 0 or np.std(pred) == 0:
        return math.nan
    return float(np.corrcoef(gt, pred)[0, 1])


def spearman(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if gt.size < 2 or np.std(gt) == 0 or np.std(pred) == 0:
        return math.nan
    return float(spearmanr(gt, pred).correlation)


def linear_fit(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if gt.size < 2 or np.var(gt) == 0:
        return math.nan, math.nan
    slope, intercept = np.polyfit(gt, pred, 1)
    return float(slope), float(intercept)


def case_metrics(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    gt_mean = float(np.mean(gt))
    pred_mean = float(np.mean(pred))
    gt_std = float(np.std(gt))
    pred_std = float(np.std(pred))
    slope, intercept = linear_fit(gt, pred)
    return {
        "case_mean_ratio": float(pred_mean / gt_mean) if gt_mean != 0 else math.nan,
        "case_std_ratio": float(pred_std / gt_std) if gt_std != 0 else math.nan,
        "case_MAE": float(np.mean(np.abs(pred - gt))),
        "case_bias": float(np.mean(pred - gt)),
        "case_CC": corrcoef(gt, pred),
        "case_spearman": spearman(gt, pred),
        "case_slope": slope,
        "case_intercept": intercept,
        "gt_mean": gt_mean,
        "pred_mean": pred_mean,
        "gt_std": gt_std,
        "pred_std": pred_std,
        "instance_count_total": int(gt.size),
    }


def mean_calibrate(gt, pred):
    metrics = case_metrics(gt, pred)
    ratio = metrics["case_mean_ratio"]
    if not np.isfinite(ratio) or ratio == 0:
        return pred
    return pred / ratio


def affine_calibrate(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    pred_std = np.std(pred)
    gt_std = np.std(gt)
    if pred_std == 0 or not np.isfinite(pred_std) or not np.isfinite(gt_std):
        return pred
    return (pred - np.mean(pred)) / pred_std * gt_std + np.mean(gt)


class GlobalStats:
    def __init__(self):
        self.gt = []
        self.pred = []

    def update(self, gt, pred):
        self.gt.append(np.asarray(gt, dtype=np.float64))
        self.pred.append(np.asarray(pred, dtype=np.float64))

    def result(self):
        if not self.gt:
            return {}
        gt = np.concatenate(self.gt)
        pred = np.concatenate(self.pred)
        gt_mean = float(np.mean(gt))
        pred_mean = float(np.mean(pred))
        gt_std = float(np.std(gt))
        pred_std = float(np.std(pred))
        slope, intercept = linear_fit(gt, pred)
        return {
            "global_count": int(gt.size),
            "MAE": float(np.mean(np.abs(pred - gt))),
            "bias": float(np.mean(pred - gt)),
            "mean_ratio": float(pred_mean / gt_mean) if gt_mean != 0 else math.nan,
            "std_ratio": float(pred_std / gt_std) if gt_std != 0 else math.nan,
            "Global_CC": corrcoef(gt, pred),
            "Global_spearman": spearman(gt, pred),
            "slope": slope,
            "intercept": intercept,
            "gt_mean": gt_mean,
            "pred_mean": pred_mean,
            "gt_std": gt_std,
            "pred_std": pred_std,
        }


def summarize(records, pred_key):
    case_rows = []
    global_stats = GlobalStats()
    for record in records:
        gt = record["gt"]
        pred = record[pred_key]
        metrics = case_metrics(gt, pred)
        case_rows.append(metrics)
        global_stats.update(gt, pred)

    summary = global_stats.result()
    for metric in [
        "case_mean_ratio",
        "case_std_ratio",
        "case_MAE",
        "case_bias",
        "case_CC",
        "case_spearman",
        "case_slope",
    ]:
        values = np.asarray([row[metric] for row in case_rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        summary[f"{metric}_avg"] = float(np.mean(values)) if values.size else math.nan
        summary[f"{metric}_median"] = float(np.median(values)) if values.size else math.nan
    summary["case_count"] = len(records)
    return summary


def numeric_corr(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return math.nan, math.nan, int(xs.size)
    return float(np.corrcoef(xs, ys)[0, 1]), float(spearmanr(xs, ys).correlation), int(xs.size)


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def diagnose(summary):
    original = summary["original"]
    mean_cal = summary["per_case_mean_cal"]
    affine_cal = summary["per_case_mean_std_cal"]
    lines = []

    if original["case_CC_avg"] < 0.5:
        lines.append("case 内 CC 偏低，说明 nvdla 不只是幅值偏了，实例排序/空间形状也明显失真。")
    else:
        lines.append("case 内 CC 仍有一定水平，说明空间排序还保留了一部分，主要矛盾可能在幅值或动态范围。")

    mae_gain_mean = 1.0 - mean_cal["MAE"] / original["MAE"] if original["MAE"] > 0 else math.nan
    mae_gain_affine = 1.0 - affine_cal["MAE"] / original["MAE"] if original["MAE"] > 0 else math.nan
    if np.isfinite(mae_gain_mean) and mae_gain_mean > 0.4:
        lines.append("只修正每个 case 的均值后 MAE 大幅下降，说明绝对幅值偏置是主要错误之一。")
    elif np.isfinite(mae_gain_mean) and mae_gain_mean > 0.15:
        lines.append("只修正每个 case 的均值后 MAE 有一定下降，说明存在幅值偏置，但不是全部问题。")
    else:
        lines.append("只修正每个 case 的均值后 MAE 改善有限，说明错误不主要是平均幅值。")

    if np.isfinite(mae_gain_affine) and mae_gain_affine - mae_gain_mean > 0.15:
        lines.append("进一步匹配标准差后还能明显改善，说明动态范围/contrast 也错了。")

    if original["std_ratio"] < 0.7:
        lines.append("全局 std_ratio 小于 1，预测动态范围被压扁。")
    elif original["std_ratio"] > 1.3:
        lines.append("全局 std_ratio 大于 1，预测动态范围被放大。")

    if original["mean_ratio"] > 1.3:
        lines.append("全局 mean_ratio 大于 1，整体高估 IR。")
    elif original["mean_ratio"] < 0.7:
        lines.append("全局 mean_ratio 小于 1，整体低估 IR。")

    if original["Global_CC"] < original["case_CC_avg"] - 0.15:
        lines.append("Global CC 明显低于 case 平均 CC，跨 case 的幅值排序也有问题。")
    return lines


def write_markdown(path, args, checkpoint_path, summaries, corr_rows, worst_cc, worst_ratio):
    original = summaries["original"]
    with open(path, "w") as f:
        f.write("# Error Type Analysis\n\n")
        f.write(f"Checkpoint: `{checkpoint_path}`\n\n")
        f.write(f"CSV: `{args.csv}`\n\n")
        f.write(f"Design filter: `{args.design_filter}`\n\n")
        f.write(
            "All metrics use the same instance-level pixels after model output is resized to the instance-count grid "
            "and repeated by `instance_count`. `slope` is `pred ~ gt` with an intercept. "
            "`mean_ratio = mean(pred) / mean(gt)`, `std_ratio = std(pred) / std(gt)`.\n\n"
        )

        f.write("## Summary\n\n")
        f.write("| Method | Cases | MAE | CC(avg) | mean_ratio | std_ratio | Global CC | slope | bias |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for key, label in [
            ("original", "original"),
            ("global_mean_cal", "global mean calibrated"),
            ("per_case_mean_cal", "per-case mean calibrated"),
            ("per_case_mean_std_cal", "per-case mean+std calibrated"),
        ]:
            row = summaries[key]
            f.write(
                f"| {label} | {row['case_count']} | {fmt(row['MAE'], 6)} | "
                f"{fmt(row['case_CC_avg'])} | {fmt(row['mean_ratio'])} | {fmt(row['std_ratio'])} | "
                f"{fmt(row['Global_CC'])} | {fmt(row['slope'])} | {fmt(row['bias'], 6)} |\n"
            )

        f.write("\n## Diagnosis\n\n")
        for line in diagnose(summaries):
            f.write(f"- {line}\n")

        f.write("\n## Strongest Input Correlations\n\n")
        f.write("| Metric | Input | Pearson | Spearman | Cases |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for row in corr_rows:
            f.write(
                f"| {row['metric']} | {row['input_stat']} | {fmt(row['pearson'])} | "
                f"{fmt(row['spearman'])} | {row['case_count']} |\n"
            )

        f.write("\n## Worst Case CC\n\n")
        f.write("| Case | Process | MAE | CC | mean_ratio | std_ratio | slope |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in worst_cc:
            f.write(
                f"| {row['case']} | {row['process']} | {fmt(row['case_MAE'], 6)} | "
                f"{fmt(row['case_CC'])} | {fmt(row['case_mean_ratio'])} | "
                f"{fmt(row['case_std_ratio'])} | {fmt(row['case_slope'])} |\n"
            )

        f.write("\n## Largest Mean-Ratio Errors\n\n")
        f.write("| Case | Process | MAE | CC | mean_ratio | std_ratio | slope |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in worst_ratio:
            f.write(
                f"| {row['case']} | {row['process']} | {fmt(row['case_MAE'], 6)} | "
                f"{fmt(row['case_CC'])} | {fmt(row['case_mean_ratio'])} | "
                f"{fmt(row['case_std_ratio'])} | {fmt(row['case_slope'])} |\n"
            )

        f.write("\nFull tables are in `case_metrics.csv`, `summary_metrics.csv`, and `input_correlations.csv`.\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a model checkpoint. Its sibling train.json is used for model and target-scale config.",
    )
    parser.add_argument("--csv", default=str(DATA_ROOT / "test_nvdla_vortex.csv"))
    parser.add_argument("--dataroot", default=str(DATA_ROOT))
    parser.add_argument("--design-filter", default="nvdla-small")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "cuda:0"])
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device(args.device)

    dataroot = resolve_path(args.dataroot, REPO_ROOT)
    csv_path = resolve_path(args.csv, REPO_ROOT)
    checkpoint_path = resolve_path(args.checkpoint, SAMPLE_MODEL_ROOT)
    if args.out_dir is None:
        out_dir = SAMPLE_MODEL_ROOT / "work_dir" / "error_type_analysis" / checkpoint_path.parent.name / args.design_filter
    else:
        out_dir = resolve_path(args.out_dir, SAMPLE_MODEL_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"Output: {out_dir}", flush=True)

    model, cfg, label_norm_stats, checkpoint_path = load_model_from_checkpoint(checkpoint_path, device)
    rows = load_rows(csv_path, dataroot, args.design_filter)
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"Cases: {len(rows)}", flush=True)

    records = []
    case_rows = []
    for idx, row in enumerate(rows, 1):
        if idx == 1 or idx % 10 == 0 or idx == len(rows):
            print(f"  {idx}/{len(rows)} {row['case']}", flush=True)
        gt = np.load(row["instance_ir_drop"]).astype(np.float32)
        pred = predict_instance_ir(model, cfg, label_norm_stats, row, device)
        metrics = case_metrics(gt, pred)
        stats = input_stats(row)
        record = {
            **row,
            **metrics,
            **stats,
            "gt": gt,
            "pred": pred,
        }
        record["pred_global_mean_cal"] = pred
        record["pred_per_case_mean_cal"] = mean_calibrate(gt, pred)
        record["pred_per_case_mean_std_cal"] = affine_calibrate(gt, pred)
        records.append(record)
        case_rows.append(
            {
                "design": row["design"],
                "tech": row["tech"],
                "process": row["process"],
                "case": row["case"],
                **metrics,
                **stats,
            }
        )

    summaries = {"original": summarize(records, "pred")}
    global_scale = summaries["original"]["mean_ratio"]
    for record in records:
        record["pred_global_mean_cal"] = record["pred"] / global_scale if np.isfinite(global_scale) and global_scale != 0 else record["pred"]
    summaries["global_mean_cal"] = summarize(records, "pred_global_mean_cal")
    summaries["per_case_mean_cal"] = summarize(records, "pred_per_case_mean_cal")
    summaries["per_case_mean_std_cal"] = summarize(records, "pred_per_case_mean_std_cal")

    summary_rows = []
    for key, label in [
        ("original", "original"),
        ("global_mean_cal", "global mean calibrated"),
        ("per_case_mean_cal", "per-case mean calibrated"),
        ("per_case_mean_std_cal", "per-case mean+std calibrated"),
    ]:
        summary_rows.append({"method": label, **summaries[key]})

    corr_rows = []
    for metric in ["case_mean_ratio", "case_std_ratio", "case_MAE", "case_CC", "case_slope"]:
        rows_for_metric = []
        for stat in INPUT_STATS:
            pearson, sp, count = numeric_corr([r[stat] for r in records], [r[metric] for r in records])
            rows_for_metric.append(
                {
                    "metric": metric,
                    "input_stat": stat,
                    "pearson": pearson,
                    "spearman": sp,
                    "case_count": count,
                }
            )
        rows_for_metric = sorted(
            rows_for_metric,
            key=lambda r: abs(r["pearson"]) if np.isfinite(r["pearson"]) else -1,
            reverse=True,
        )
        corr_rows.extend(rows_for_metric[:3])

    worst_cc = sorted(case_rows, key=lambda r: r["case_CC"] if np.isfinite(r["case_CC"]) else 1.0)[:10]
    worst_ratio = sorted(
        case_rows,
        key=lambda r: abs(math.log(r["case_mean_ratio"])) if r["case_mean_ratio"] > 0 and np.isfinite(r["case_mean_ratio"]) else -1,
        reverse=True,
    )[:10]

    case_fields = [
        "design",
        "tech",
        "process",
        "case",
        "case_mean_ratio",
        "case_std_ratio",
        "case_MAE",
        "case_bias",
        "case_CC",
        "case_spearman",
        "case_slope",
        "case_intercept",
        "gt_mean",
        "pred_mean",
        "gt_std",
        "pred_std",
        "instance_count_total",
        *INPUT_STATS,
    ]
    summary_fields = [
        "method",
        "case_count",
        "global_count",
        "MAE",
        "bias",
        "mean_ratio",
        "std_ratio",
        "Global_CC",
        "Global_spearman",
        "slope",
        "intercept",
        "gt_mean",
        "pred_mean",
        "gt_std",
        "pred_std",
        "case_mean_ratio_avg",
        "case_mean_ratio_median",
        "case_std_ratio_avg",
        "case_std_ratio_median",
        "case_MAE_avg",
        "case_MAE_median",
        "case_bias_avg",
        "case_bias_median",
        "case_CC_avg",
        "case_CC_median",
        "case_spearman_avg",
        "case_spearman_median",
        "case_slope_avg",
        "case_slope_median",
    ]
    corr_fields = ["metric", "input_stat", "pearson", "spearman", "case_count"]
    write_csv(out_dir / "case_metrics.csv", case_rows, case_fields)
    write_csv(out_dir / "summary_metrics.csv", summary_rows, summary_fields)
    write_csv(out_dir / "input_correlations.csv", corr_rows, corr_fields)
    write_markdown(out_dir / "summary.md", args, checkpoint_path, summaries, corr_rows, worst_cc, worst_ratio)

    print(f"Wrote {out_dir}", flush=True)
    print(
        "original: MAE={:.6f} CC(avg)={:.4f} mean_ratio={:.4f} std_ratio={:.4f} Global_CC={:.4f} slope={:.4f}".format(
            summaries["original"]["MAE"],
            summaries["original"]["case_CC_avg"],
            summaries["original"]["mean_ratio"],
            summaries["original"]["std_ratio"],
            summaries["original"]["Global_CC"],
            summaries["original"]["slope"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
