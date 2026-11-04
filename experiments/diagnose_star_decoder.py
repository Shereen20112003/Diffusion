"""Frozen-reference, true-coordinate tripod decoder audit. Never trains a DDPM."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from conditional_star import (
    H_RATIOS, SCENARIOS, THRESHOLDS, GaussianFrechet, StarEncoder, jr_arrays,
    law_probabilities, radial_cdf, sample_pairs, stream_seed, tensor,
    threshold_points, to_planar, validate_points,
)
from conditional_star_models import decode_array
from run_experiments import save_json, write_csv


EXPERIMENT = "conditional_tripod_decoder_only_debug"
ENCODERS = ("primary_sample_medoid", "controlled_center_endpoints")
NUMERICAL_AUDIT_TOLERANCE = 1e-12


def balanced_dictionary(references):
    """Symmetrize only B, preserving its total prior mass and center multiplicity."""
    validate_points(references)
    radius = references.sum(axis=1)
    positive = radius[radius > 0]
    if not len(positive):
        raise ValueError("Balanced diagnostic needs positive reference radii.")
    centers = np.zeros((int((radius == 0).sum()), 3))
    arms = [positive[:, None] * np.eye(3)[j] for j in range(3)]
    points = np.concatenate([centers, *arms])
    prior = np.r_[np.ones(len(centers)), np.full(3 * len(positive), 1 / 3)]
    return points, prior


class DiagnosticDecoder(GaussianFrechet):
    def __init__(self, encoder, references, ratio, common_h_scale, prior=None):
        super().__init__(encoder, references, ratio)
        self.native_h_scale = self.h_scale
        self.h_scale = common_h_scale
        self.h = ratio * common_h_scale
        if prior is not None:
            prior = np.asarray(prior, dtype=np.float64)
            if prior.shape != (len(references),) or np.any(prior <= 0) or not np.isfinite(prior).all():
                raise ValueError("Reference prior must have one finite positive weight per object.")
        self.register_buffer("prior", None if prior is None else tensor(prior))

    def weights(self, z):
        weights = super().weights(z)
        if self.prior is None:
            return weights
        weighted = weights * self.prior
        return weighted / weighted.sum(dim=1, keepdim=True)


def reference_records(references, prior=None):
    j, r = jr_arrays(references)
    prior = np.ones(len(r)) if prior is None else np.asarray(prior)
    records = []
    for arm in range(4):
        selected = j == arm
        radii = r[selected]
        records.append({
            "arm": arm, "count": int(selected.sum()),
            "prior_mass": float(prior[selected].sum()),
            "prior_radial_moment": float((prior[selected] * radii).sum()),
            "minimum_radius": float(radii.min()) if len(radii) else None,
            "median_radius": float(np.median(radii)) if len(radii) else None,
            "maximum_radius": float(radii.max()) if len(radii) else None,
        })
    return records


@torch.no_grad()
def center_balance(decoder):
    coordinate = decoder.encoder(torch.zeros((1, 3), dtype=torch.float64))
    weights = decoder.weights(coordinate)
    moment = (weights @ decoder.references)[0].numpy()
    g = 2 * moment - moment.sum()
    output = decoder(coordinate).numpy()[0]
    j, radius = jr_arrays(output[None])
    reference = decoder.references.numpy()
    ref_j, ref_r = jr_arrays(reference)
    result = {
        "h_ratio": decoder.h_ratio, "h": decoder.h, "h_scale": decoder.h_scale,
        "dictionary_native_h_scale_not_used": decoder.native_h_scale,
        "reference_count": len(reference), "reference_center_count": int((ref_j == 0).sum()),
        "minimum_positive_reference_radius": float(ref_r[ref_r > 0].min()),
        "encoded_center": coordinate[0].tolist(), "T": float(moment.sum()),
        "max_g": float(g.max()), "center_decoded_J": int(j[0]),
        "center_decoded_R": float(radius[0]),
        "strict_center_rule_satisfied": bool((g.max() <= 0) == (radius[0] == 0)),
        "near_zero_max_g": bool(abs(g.max()) <= NUMERICAL_AUDIT_TOLERANCE),
        "numerical_audit_tolerance_not_snapping": NUMERICAL_AUDIT_TOLERANCE,
        "raw_center_tolerance": 0.,
    }
    w = weights[0].numpy()
    for arm in range(1, 4):
        selected = ref_j == arm
        result[f"T_{arm}"] = float(moment[arm - 1])
        result[f"g_{arm}"] = float(g[arm - 1])
        result[f"reference_arm_{arm}_count"] = int(selected.sum())
        result[f"center_kernel_arm_{arm}_weight"] = float(w[selected].sum())
    return result


def confusion_record(truth, predicted):
    validate_points(truth)
    validate_points(predicted)
    if len(truth) != len(predicted):
        raise ValueError("Confusion diagnostics require paired inputs and outputs.")
    true_j, true_r = jr_arrays(truth)
    pred_j, pred_r = jr_arrays(predicted)
    true_center, decoded_center = true_r == 0, pred_r == 0
    tp = int((true_center & decoded_center).sum())
    fn = int((true_center & ~decoded_center).sum())
    fp = int((~true_center & decoded_center).sum())
    tn = int((~true_center & ~decoded_center).sum())
    result = {
        "n": len(truth), "true_center_count": int(true_center.sum()),
        "decoded_center_count": int(decoded_center.sum()), "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "center_retention_rate": tp / (tp + fn) if tp + fn else None,
        "noncenter_false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_center_mass": fp / len(truth), "lost_center_mass": fn / len(truth),
        "true_center_fraction": float(true_center.mean()),
        "decoded_center_fraction": float(decoded_center.mean()),
        "paired_signed_center_mass_error": (fp - fn) / len(truth),
        "intrinsic_reconstruction_mse": float(np.mean(np.abs(truth - predicted).sum(axis=1) ** 2)),
        "decoded_endpoint_mass": float((pred_r == 1).mean()),
    }
    for arm in range(1, 4):
        result[f"paired_arm_{arm}_probability_error"] = float((pred_j == arm).mean() - (true_j == arm).mean())
    return result


@torch.no_grad()
def identical_center_audit(encoder, decoder, truth, coordinates, decoded):
    mask = truth.sum(axis=1) == 0
    repeated = np.zeros((35, 3))
    encoded = encoder(tensor(repeated)).numpy()
    result = decode_array(decoder, encoded, chunk=17)
    direct = decoder(encoder(torch.zeros((1, 3), dtype=torch.float64))).numpy()[0]
    unique = len(np.unique(encoded, axis=0))
    delta = float(np.abs(result - direct).max())
    encoded_true_unique = len(np.unique(coordinates[mask], axis=0)) if mask.any() else 0
    heldout_delta = float(np.abs(decoded[mask] - direct).max()) if mask.any() else None
    labels = decoded[mask].sum(axis=1) == 0
    label_consistent = bool(not mask.any() or np.all(labels == (direct.sum() == 0)))
    if (unique != 1 or delta > NUMERICAL_AUDIT_TOLERANCE or not label_consistent
            or encoded_true_unique > 1
            or (heldout_delta is not None and heldout_delta > NUMERICAL_AUDIT_TOLERANCE)):
        raise AssertionError("Fixed center inputs have inconsistent encoding/decoding outputs.")
    return {
        "synthetic_center_repetitions": len(repeated), "center_encoded_unique_rows": unique,
        "cross_batch_max_output_difference": delta,
        "heldout_true_center_count": int(mask.sum()),
        "heldout_center_encoded_unique_rows": encoded_true_unique,
        "heldout_center_max_output_difference": heldout_delta,
        "heldout_center_labels_consistent": label_consistent, "passed": True,
    }


def probability_metrics(truth, predicted, x, scenario):
    record = confusion_record(truth, predicted)
    p0, pi = law_probabilities([x], scenario)
    j, _ = jr_arrays(predicted)
    record["analytic_center_probability"] = float(p0[0])
    record["population_signed_center_mass_error"] = record["decoded_center_fraction"] - p0[0]
    for arm in range(1, 4):
        target = float((1 - p0[0]) * pi[0, arm - 1])
        record[f"analytic_arm_{arm}_probability"] = target
        record[f"population_arm_{arm}_probability_error"] = float((j == arm).mean() - target)
    return record


def plots(output, metrics, balance, ref_data):
    for scenario, filename, title in (
        (SCENARIOS[1], "S1_center_mass_vs_bandwidth.png", "S1: genuine center atom"),
        (SCENARIOS[2], "S2_false_center_mass_vs_bandwidth.png", "S2: no center atom"),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
        x_grid = sorted({row["x"] for row in metrics})
        for axis_row, dictionary in zip(axes, ("original_B", "balanced_B_oracle")):
            for axis, x in zip(axis_row, x_grid):
                group = [r for r in metrics if r["scenario"] == scenario and r["x"] == x
                         and r["encoder"] == ENCODERS[0] and r["dictionary"] == dictionary]
                for a in THRESHOLDS:
                    subset = sorted([r for r in group if r["threshold"] == a], key=lambda r: r["h_ratio"])
                    axis.plot([r["h_ratio"] for r in subset], [r["decoded_center_fraction"] for r in subset],
                              marker=".", label=f"a={a:g}")
                axis.axhline(group[0]["analytic_center_probability"], color="black", linestyle="--",
                             label="Population truth")
                axis.axhline(group[0]["true_center_fraction"], color="gray", linestyle=":",
                             label="Sample truth")
                axis.set(title=f"{dictionary}, X={x:g}", xlabel="h / original-B h_scale",
                         ylabel="Decoded center fraction", ylim=(-.015, 1.015))
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{title} — true-coordinate decoder only; primary encoder")
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, scenario in zip(axes, SCENARIOS):
        for encoder in ENCODERS:
            for dictionary in ("original_B", "balanced_B_oracle"):
                group = sorted([r for r in balance if r["scenario"] == scenario and r["encoder"] == encoder
                                and r["dictionary"] == dictionary], key=lambda r: r["h_ratio"])
                axis.plot([r["h_ratio"] for r in group], [r["center_decoded_R"] for r in group],
                          marker=".", label=f"{encoder} / {dictionary}")
        axis.set(title=scenario, xlabel="h / original-B h_scale", ylabel="D(E(center)) radius")
    axes[0].legend(fontsize=6)
    fig.suptitle("Where does an exact center input go? Raw decoder, no snapping")
    fig.tight_layout()
    fig.savefig(output / "exact_center_displacement.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for column, scenario in enumerate(SCENARIOS):
        original, balanced, prior = ref_data[scenario]
        for dictionary, points, weights, style in (
            ("original B", original, np.ones(len(original)), "-"),
            ("symmetrized B (prior-weighted)", balanced, prior, "--"),
        ):
            j, radius = jr_arrays(points)
            for arm in range(1, 4):
                select = j == arm
                axes[1, column].hist(radius[select], bins=np.linspace(0, 1, 11),
                                     weights=weights[select], histtype="step",
                                     linestyle=style, label=f"{dictionary}, arm {arm}")
        xy = to_planar(original)
        axes[0, column].scatter(*xy.T, s=12)
        records = reference_records(original)
        counts = [r["count"] for r in records]
        axes[0, column].set(title=f"{scenario}\ncenter/arms counts={counts}", aspect="equal")
        axes[1, column].set(xlabel="Radius", ylabel="Reference prior mass per bin")
    axes[1, 0].legend(fontsize=6)
    fig.suptitle("Frozen references: counts and radial coverage; balanced control changes B only")
    fig.tight_layout()
    fig.savefig(output / "reference_balance.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for axis, scenario in zip(axes, SCENARIOS):
        group = [r for r in metrics if r["scenario"] == scenario and r["encoder"] == ENCODERS[0]
                 and r["dictionary"] == "original_B" and r["threshold"] == 0]
        for x in sorted({r["x"] for r in group}):
            subset = sorted([r for r in group if r["x"] == x], key=lambda r: r["h_ratio"])
            axis.plot([r["h_ratio"] for r in subset], [r["false_center_mass"] for r in subset],
                      marker=".", label=f"FP/n, X={x:g}")
            axis.plot([r["h_ratio"] for r in subset], [r["lost_center_mass"] for r in subset],
                      marker="x", linestyle="--", label=f"FN/n, X={x:g}")
        axis.set(title=scenario, xlabel="h / h_scale", ylabel="Fraction of all inputs")
    axes[1].legend(fontsize=7)
    fig.suptitle("Raw decoder: lost true centers versus spurious centers (no DDPM)")
    fig.tight_layout()
    fig.savefig(output / "raw_center_confusion.png", dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(r"experiments\results\conditional_star_smoke_20261103"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-x", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--x", nargs=3, type=float, default=[.1, .5, .9])
    args = parser.parse_args(argv)
    if (args.samples_per_x < 1000 or args.seed < 0 or len(set(args.x)) != 3
            or any(not np.isfinite(x) or not 0 <= x <= 1 for x in args.x)):
        parser.error("Use at least 1000 samples per X, a nonnegative seed and three distinct conditions in [0,1].")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new empty diagnostic output directory.")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    started = time.perf_counter()
    source_files = ("diagnose_star_decoder.py", "conditional_star.py", "conditional_star_models.py",
                    "metric_spaces.py", "run_experiments.py")
    config = {
        "experiment": EXPERIMENT, "started_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source.resolve()), "seed": args.seed, "samples_per_x": args.samples_per_x,
        "x": args.x, "h_ratios": H_RATIOS, "thresholds": THRESHOLDS,
        "no_ddpm_training_or_sampling": True, "reference_seed": args.seed,
        "float_dtype": "float64", "raw_center_tolerance": 0.,
        "audit_tolerance_not_snapping": NUMERICAL_AUDIT_TOLERANCE,
        "balanced_dictionary": (
            "Replicate every original positive radius on each arm with prior 1/3 per copy; "
            "retain original center copies with prior 1. Total prior and center prior masses unchanged. "
            "Raw dictionary count increases. Oracle diagnostic only, not a learned baseline."),
        "bandwidth_control": "Use original-B h_scale within each frozen encoder for both B variants.",
        "encoder_control": "Fit primary encoder once on original references; no refit when B changes.",
        "base_sensitivity": "deferred until B/bandwidth diagnostics understood",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                   text=True, check=True).stdout.strip(),
        "git_status_porcelain": subprocess.run(["git", "status", "--porcelain"],
                                              capture_output=True, text=True, check=True).stdout,
        "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in source_files},
        "environment": {"numpy": np.__version__, "torch": torch.__version__},
        "status": "decoder_only_development_diagnostic_not_benchmark",
    }
    save_json(output / "config.json", config)
    metrics, balances, audits, original_audit, references_table, bias_rows = [], [], [], [], [], []
    ref_data = {}
    source_hashes = {}
    try:
        for scenario in SCENARIOS:
            print(f"{scenario}: frozen-reference decoder-only audit", flush=True)
            folder = output / scenario
            folder.mkdir()
            source = args.source / scenario / f"seed_{args.seed}"
            data_file = source / "independent_data.npz"
            source_hashes[str(data_file)] = hashlib.sha256(data_file.read_bytes()).hexdigest()
            with np.load(data_file) as saved:
                ref_y, old_y = saved["reference_y"].copy(), saved["validation_y"].copy()
            primary = StarEncoder(ref_y)
            saved_geometry = json.loads((source / "references_medoid_k3.json").read_text())
            description = primary.describe()
            for key in ("base", "landmark_indices", "mean", "scale"):
                if description[key] != saved_geometry[key]:
                    raise AssertionError(f"Frozen encoder mismatch in {scenario}: {key}")
            save_json(folder / "frozen_encoder.json", description)
            balanced, prior = balanced_dictionary(ref_y)
            ref_data[scenario] = (ref_y, balanced, prior)
            np.savez_compressed(folder / "frozen_reference_dictionaries.npz", original=ref_y,
                                balanced=balanced, balanced_prior=prior)
            for name, points, p in (("original_B", ref_y, None), ("balanced_B_oracle", balanced, prior)):
                for record in reference_records(points, p):
                    references_table.append({"scenario": scenario, "dictionary": name, **record})
            all_x, all_y, seeds = [], [], []
            for index, x in enumerate(args.x):
                seed = stream_seed(args.seed, scenario, 20261104, index)
                xx, yy = sample_pairs(args.samples_per_x, scenario, seed, x=[x])
                seeds.append(seed)
                all_x.append(xx)
                all_y.append(yy)
            x_values, true_y = np.concatenate(all_x), np.concatenate(all_y)
            true_j, true_r = jr_arrays(true_y)
            np.savez_compressed(folder / "independent_true_inputs.npz",
                                x=x_values, points=true_y, J=true_j, R=true_r, stream_seeds=seeds)
            for a in THRESHOLDS:
                snapped = threshold_points(true_y, a)
                for x in args.x:
                    select = x_values == x
                    p0 = float(law_probabilities([x], scenario)[0][0])
                    additional = 0. if a == 0 else float((1 - p0) * radial_cdf(a, [x], scenario)[0])
                    bias_rows.append({"scenario": scenario, "x": x, "threshold": a,
                                      "analytic_added_center_mass": additional,
                                      "analytic_thresholded_center_mass": p0 + additional,
                                      **probability_metrics(true_y[select], snapped[select], x, scenario)})
            for encoder_name, encoder in zip(ENCODERS, (primary, StarEncoder.controlled())):
                with torch.no_grad():
                    z = encoder(tensor(true_y)).numpy()
                np.savez_compressed(folder / f"coordinates_{encoder_name}.npz", coordinates=z)
                common_scale = GaussianFrechet(encoder, ref_y, .1).h_scale
                for dictionary, points, p in (
                    ("original_B", ref_y, None), ("balanced_B_oracle", balanced, prior),
                ):
                    for ratio in H_RATIOS:
                        decoder = DiagnosticDecoder(encoder, points, ratio, common_scale, p)
                        prefix = {"scenario": scenario, "reference_seed": args.seed,
                                  "encoder": encoder_name, "dictionary": dictionary, "h_ratio": ratio}
                        balance = center_balance(decoder)
                        balances.append({**prefix, **balance})
                        decoded = decode_array(decoder, z, chunk=256)
                        validate_points(decoded)
                        audit = identical_center_audit(encoder, decoder, true_y, z, decoded)
                        audits.append({**prefix, **audit})
                        pred_j, pred_r = jr_arrays(decoded)
                        np.savez_compressed(
                            folder / f"decoded_{encoder_name}_{dictionary}_h{ratio:g}.npz",
                            J=pred_j, R=pred_r, points=decoded)
                        if encoder_name == ENCODERS[0] and dictionary == "original_B":
                            with torch.no_grad():
                                old_z = encoder(tensor(old_y)).numpy()
                            old_decoded = decode_array(decoder, old_z)
                            original_audit.append({
                                **prefix, **confusion_record(old_y, old_decoded),
                                **{k: balance[k] for k in ("T_1", "T_2", "T_3", "g_1", "g_2", "g_3",
                                                          "center_decoded_J", "center_decoded_R")},
                            })
                        previous_centers = np.zeros(len(decoded), dtype=bool)
                        for a in THRESHOLDS:
                            predicted = threshold_points(decoded, a)
                            is_center = predicted.sum(axis=1) == 0
                            if np.any(previous_centers & ~is_center):
                                raise AssertionError("Increasing a threshold removed an existing center.")
                            previous_centers = is_center
                            for x in args.x:
                                select = x_values == x
                                metrics.append({
                                    **prefix, "x": x, "threshold": a, "h": decoder.h,
                                    "h_scale": common_scale,
                                    **{k: balance[k] for k in ("T_1", "T_2", "T_3", "g_1", "g_2", "g_3",
                                                              "max_g", "center_decoded_J", "center_decoded_R")},
                                    **probability_metrics(true_y[select], predicted[select], x, scenario),
                                })
            write_csv(output / "per_x_decoder_confusion.csv", metrics)
            write_csv(output / "center_branch_balance.csv", balances)
            write_csv(output / "identical_center_audits.csv", audits)
            write_csv(output / "original_48_observation_audit.csv", original_audit)
            write_csv(output / "reference_balance.csv", references_table)
            write_csv(output / "perfect_sample_threshold_bias.csv", bias_rows)
        save_json(output / "input_sha256.json", source_hashes)
        plots(output, metrics, balances, ref_data)
        save_json(output / "completion.json", {
            "experiment": EXPERIMENT, "complete": True, "ddpm_runs": 0,
            "seconds": time.perf_counter() - started, "metric_rows": len(metrics),
            "center_invariant_audits_passed": len(audits),
            "new_true_observations": len(SCENARIOS) * len(args.x) * args.samples_per_x,
            "scientific_claim": "decoder diagnosis only; no superiority or conditional-generation claim",
        })
        print(f"Completed decoder-only diagnostic: {output}", flush=True)
    except Exception as error:
        save_json(output / "last_failure.json", {"type": type(error).__name__, "message": str(error),
                                                "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
