"""Staged conditional metric-star pilot; no Appendix-J certification claims."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t
import torch

from conditional_star import (
    H_RATIOS, RADIUS_GRID, SCENARIOS, THRESHOLDS, GaussianFrechet, StarEncoder,
    brier_score, conditional_resample, decoder_diagnostics, intrinsic_w2, jr_arrays,
    probability_record, project_planar,
    sample_pairs, stream_seed, tensor, threshold_points, to_planar, validate_points,
)
from conditional_star_models import (
    ProjectedMLP, decode_array, fit_categorical, fit_ddpm, fit_decoder,
    predicted_events, sample_ddpm,
)
from run_experiments import save_json, write_csv


EXPERIMENT = "conditional_metric_star_pilot"
METHOD_COORD = "conditional_coordinate_ddpm_gaussian_frechet"
METHOD_DIRECT = "conditional_planar_ddpm_segment_projection"
METHOD_EMPIRICAL = "conditional_empirical_gaussian_x_resampling"
METHOD_IGNORED = "x_ignored_coordinate_ddpm_gaussian_frechet"
METHOD_NAMES = {
    METHOD_COORD: "Conditional coordinate DDPM + Gaussian Frechet",
    METHOD_DIRECT: "Conditional planar DDPM + segment projection",
    METHOD_EMPIRICAL: "Conditional empirical resampling",
    METHOD_IGNORED: "X-ignored coordinate DDPM + Gaussian Frechet",
    "real_real_mc_reference": "Real-real Monte Carlo reference",
}


class Budget:
    def __init__(self, seconds):
        self.started = time.perf_counter()
        self.seconds = seconds

    def check(self):
        if time.perf_counter() - self.started > self.seconds:
            raise TimeoutError(
                f"Conditional-star execution exceeded {self.seconds:g}s. "
                "Partial artifacts are preserved; no completion claim is made."
            )


def source_hashes():
    root = Path(__file__).parent
    names = ("conditional_star.py", "conditional_star_models.py", "run_conditional_star.py",
             "metric_spaces.py", "run_experiments.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def artifact_stem(name):
    return f"{name[:24]}_{hashlib.sha256(name.encode('ascii')).hexdigest()[:8]}"


def latent_validation(model, val_x, args, seed, budget):
    return sample_ddpm(model, val_x, args.validation_draws, args.timesteps, seed, budget.check)


def choose_coordinate(model, decoders, val_x, val_y, args, seed, budget):
    coordinates = latent_validation(model, val_x, args, seed, budget)
    candidates = []
    for decoder in decoders:
        budget.check()
        samples = decode_array(decoder, coordinates)
        score = brier_score(predicted_events(samples), val_y)
        candidates.append({"h_ratio": decoder.h_ratio, "h": decoder.h,
                           "validation_brier": score})
    best = min(candidates, key=lambda row: row["validation_brier"])
    return best["validation_brier"], best["h_ratio"], candidates


def choose_direct(model, val_x, val_y, args, seed, budget):
    ambient = latent_validation(model, val_x, args, seed, budget)
    samples, _ = project_planar(ambient.reshape(-1, 2))
    samples = samples.reshape(len(val_x), args.validation_draws, 3)
    score = brier_score(predicted_events(samples), val_y)
    return score, None, [{"validation_brier": score}]


def select_empirical(train_x, train_y, val_x, val_y, args, seed, budget):
    started = time.perf_counter()
    candidates = []
    for h in args.x_bandwidths:
        budget.check()
        samples = conditional_resample(train_x, train_y, val_x, args.validation_draws, h, seed)
        candidates.append({"x_bandwidth": h,
                           "validation_brier": brier_score(predicted_events(samples), val_y)})
    selected = min(candidates, key=lambda record: record["validation_brier"])
    return selected["x_bandwidth"], {
        "candidates": candidates, "selected": selected["x_bandwidth"],
        "fit_seconds": time.perf_counter() - started,
    }


def fit_unit(args, scenario, master, folder, data, budget, primary_encoder):
    train_x, train_y, val_x, val_y, ref_y = data
    selection_x = val_x[:args.validation_conditions]
    selection_y = val_y[:args.validation_conditions]
    encoders = {"medoid_k3": primary_encoder}
    if args.second_stage:
        encoders.update({f"medoid_k{k}": StarEncoder(ref_y, k=k) for k in (6, 12)})
        encoders["frechet_base_k3"] = StarEncoder(
            ref_y, base_kind="frechet", landmarks=encoders["medoid_k3"].indices)
    models, decoder_sets, selected_h, records = {}, {}, {}, {}
    coordinates = {}
    for index, (tag, encoder) in enumerate(encoders.items()):
        budget.check()
        with torch.no_grad():
            train_z, val_z = encoder(tensor(train_y)).numpy(), encoder(tensor(val_y)).numpy()
        coordinates[tag] = (train_z, val_z)
        decoders = [GaussianFrechet(encoder, ref_y, h) for h in H_RATIOS]
        decoder_sets[tag] = decoders
        method = METHOD_COORD if tag == "medoid_k3" else f"{METHOD_COORD}__{tag}"
        print(f"{scenario}, seed {master}: fitting {method}", flush=True)
        model, h, record = fit_ddpm(
            train_z, train_x, args, stream_seed(
                master, scenario, 20, 0 if tag == "frechet_base_k3" else index),
            lambda net, ds=decoders: choose_coordinate(
                net, ds, selection_x, selection_y, args,
                stream_seed(master, scenario, 30), budget),
            folder / f"{artifact_stem(method)}.pt", budget.check,
        )
        models[tag], selected_h[tag], records[method] = model, h, record
        save_json(folder / f"references_{tag}.json", encoder.describe())
    primary_z = coordinates["medoid_k3"][0]
    print(f"{scenario}, seed {master}: fitting X-ignored control", flush=True)
    model, h, record = fit_ddpm(
        primary_z, train_x, args, stream_seed(master, scenario, 21),
        lambda net: choose_coordinate(net, decoder_sets["medoid_k3"], selection_x, selection_y,
                                      args, stream_seed(master, scenario, 30), budget),
        folder / f"{artifact_stem(METHOD_IGNORED)}.pt", budget.check, ignore_x=True,
    )
    models["x_ignored"], selected_h["x_ignored"], records[METHOD_IGNORED] = model, h, record
    print(f"{scenario}, seed {master}: fitting direct planar comparator", flush=True)
    model, _, record = fit_ddpm(
        to_planar(train_y), train_x, args, stream_seed(master, scenario, 22),
        lambda net: choose_direct(net, selection_x, selection_y, args,
                                  stream_seed(master, scenario, 30), budget),
        folder / f"{artifact_stem(METHOD_DIRECT)}.pt", budget.check,
    )
    models["direct"], records[METHOD_DIRECT] = model, record
    empirical_h, records[METHOD_EMPIRICAL] = select_empirical(
        train_x, train_y, selection_x, selection_y, args,
        stream_seed(master, scenario, 30), budget)
    extras = {}
    if args.second_stage:
        encoder = encoders["medoid_k3"]
        train_z, val_z = coordinates["medoid_k3"]
        learned = []
        for index, h in enumerate(H_RATIOS):
            torch.manual_seed(stream_seed(master, scenario, 23, index))
            decoder = GaussianFrechet(encoder, ref_y, h, args.width, learned=True)
            record = fit_decoder(decoder, train_z, train_y, val_z, val_y, args,
                                 stream_seed(master, scenario, 23, index), budget.check)
            learned.append(decoder)
            records[f"bounded_residual_frechet_h{h:g}"] = record
            torch.save(decoder.state_dict(), folder / f"bounded_residual_frechet_h{h:g}.pt")
        primary = models["medoid_k3"]
        score, h, candidates = choose_coordinate(
            primary, learned, selection_x, selection_y, args,
            stream_seed(master, scenario, 30), budget)
        extras["learned"] = learned
        selected_h["learned"] = h
        records["bounded_residual_frechet_selection"] = {
            "validation_brier": score, "selected": h, "candidates": candidates,
            "residual_logits": "2*tanh(network(z)); pairwise logit residual range <=4",
            "localization_caveat": "bounded residual preserves relative Gaussian localization; no empirical theorem certificate",
        }
        torch.manual_seed(stream_seed(master, scenario, 24))
        mlp = ProjectedMLP(train_z.shape[1], args.width)
        records["coordinate_mlp_explicit_segment_projection"] = fit_decoder(
            mlp, train_z, train_y, val_z, val_y, args,
            stream_seed(master, scenario, 24), budget.check)
        extras["mlp"] = mlp
        torch.save(mlp.state_dict(), folder / "coordinate_mlp_explicit_segment_projection.pt")
        categorical, record = fit_categorical(
            train_x, train_y, selection_x, selection_y, args,
            stream_seed(master, scenario, 25), budget.check)
        extras["categorical"] = categorical
        records["conditional_categorical_radial_histogram"] = record
        torch.save(categorical.state_dict(), folder / "conditional_categorical_radial_histogram.pt")
    save_json(folder / "selection.json", records)
    return encoders, models, decoder_sets, selected_h, empirical_h, extras, records


def reference_plot(folder, ref_y, encoders):
    fig, axis = plt.subplots(figsize=(6, 5))
    for endpoint in to_planar(np.eye(3)):
        axis.plot([0, endpoint[0]], [0, endpoint[1]], color="gray")
    planar = to_planar(ref_y)
    axis.scatter(*planar.T, s=10, alpha=.3, label="B (multiplicities retained)")
    encoder = encoders["medoid_k3"]
    axis.scatter(*to_planar(encoder.base.numpy()).T, marker="*", s=160, label="Medoid base")
    axis.scatter(*to_planar(encoder.landmarks.numpy()).T, marker="x", s=70, label="FPS A")
    axis.legend(fontsize=8)
    axis.set_aspect("equal")
    axis.set_title("Conditional metric-star pilot: fixed reference geometry")
    fig.tight_layout()
    fig.savefig(folder / "reference_geometry.png", dpi=150)
    plt.close(fig)


def reconstruction_diagnostics(folder, ref_y, val_y, encoder, budget):
    records = []
    sample_z = encoder(tensor(val_y)).detach().numpy()
    controlled = StarEncoder.controlled()
    for name, current in (("primary_sample_medoid", encoder), ("controlled_center_endpoints", controlled)):
        for h in H_RATIOS:
            budget.check()
            decoder = GaussianFrechet(current, ref_y, h)
            z = current(tensor(val_y)).detach().numpy()
            start = time.perf_counter()
            reconstructed = decode_array(decoder, z)
            margin = decoder.margins(tensor(z)).detach().numpy()
            records.append({
                "encoder": name, "h_ratio": h, "h_scale": decoder.h_scale,
                "paired_intrinsic_mse": float(np.mean(np.abs(reconstructed - val_y).sum(axis=1) ** 2)),
                "true_center_mass": float(np.mean(val_y.sum(axis=1) == 0)),
                "decoded_exact_center_mass": float(np.mean(reconstructed.sum(axis=1) == 0)),
                "minimum_absolute_sign_margin": float(np.min(np.abs(margin))),
                "positive_margin_le_1e_12_count": int(np.sum((margin > 0) & (margin <= 1e-12))),
                "decoding_seconds": time.perf_counter() - start,
            })
    np.savez_compressed(folder / "held_out_paired_coordinates.npz",
                        points=val_y, coordinates=sample_z)
    write_csv(folder / "held_out_reconstruction.csv", records)
    save_json(folder / "controlled_encoder.json", controlled.describe())


def summarize_rows(rows):
    groups = {}
    keys = ("scenario", "method", "stage", "h_ratio", "threshold", "perturbation_sigma", "selected")
    metrics = ("w2", "label_tv", "center_absolute_error", "center_signed_error",
               "false_endpoint_mass", "validity")
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in groups.items():
        integrated = {}
        for seed in sorted({row["seed"] for row in group}):
            draws = []
            seed_rows = [row for row in group if row["seed"] == seed]
            for draw in sorted({row["draw"] for row in seed_rows}):
                subset = [row for row in seed_rows if row["draw"] == draw]
                values = {m: float(np.mean([row[m] for row in subset])) for m in metrics}
                values["w2"] = float(np.sqrt(np.mean([row["w2"] ** 2 for row in subset])))
                draws.append(values)
            integrated[seed] = {m: float(np.mean([v[m] for v in draws])) for m in metrics}
        record = dict(zip(keys, key))
        record["independent_training_seed_count"] = len(integrated)
        record["evaluation_draw_count_per_seed"] = len({row["draw"] for row in group})
        for metric in metrics:
            values = np.asarray([v[metric] for v in integrated.values()])
            record[metric] = float(values.mean())
            record[f"{metric}_between_seed_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
            half = (float(student_t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1 else None)
            record[f"{metric}_seed_mean_ci95_halfwidth"] = half
        result.append(record)
    return result


def paired_comparisons(rows):
    primary = {tuple(row[k] for k in ("scenario", "seed", "draw", "x", "threshold")): row
               for row in rows if row["method"] == METHOD_COORD and row["selected"]
               and row["stage"] == "learned_generator"}
    comparisons = []
    for row in rows:
        if (row["stage"] != "learned_generator" or not row["selected"]
                or row["method"] == METHOD_COORD):
            continue
        key = tuple(row[k] for k in ("scenario", "seed", "draw", "x", "threshold"))
        match = primary.get(key)
        if match is not None:
            comparisons.append({
                **{k: row[k] for k in ("scenario", "seed", "draw", "x", "threshold")},
                "comparator": row["method"], "primary_minus_comparator_w2": match["w2"] - row["w2"],
                "primary_minus_comparator_tv": match["label_tv"] - row["label_tv"],
                "pairing": "same seed/condition/true OT sample; common coordinate draws for decoder variants",
            })
    return comparisons


def summarize_selected(rows):
    return summarize_rows([{**row, "h_ratio": "validation_selected"}
                           for row in rows if row["selected"]])


def integrated_paired_comparisons(rows):
    result = []
    for seed in sorted({row["seed"] for row in rows}):
        summary = summarize_selected([r for r in rows if r["seed"] == seed
                                      and r["stage"] == "learned_generator"])
        primary = {(r["scenario"], r["threshold"]): r for r in summary if r["method"] == METHOD_COORD}
        for row in summary:
            if row["method"] == METHOD_COORD:
                continue
            match = primary.get((row["scenario"], row["threshold"]))
            if match is not None:
                result.append({"seed": seed, "scenario": row["scenario"], "threshold": row["threshold"],
                               "comparator": row["method"],
                               "primary_minus_comparator_integrated_w2": match["w2"] - row["w2"],
                               "primary_minus_comparator_integrated_tv": match["label_tv"] - row["label_tv"],
                               "aggregation": "RMS across conditions, then mean across evaluation draws"})
    return result


def diagnostic_summary(rows, scenario, smoke):
    raw = [r for r in rows if r["threshold"] == 0 and r["selected"]
           and r["stage"] == "learned_generator"]
    summary = summarize_rows(raw)
    primary = next((r for r in summary if r["method"] == METHOD_COORD), None)
    ignored = next((r for r in summary if r["method"] == METHOD_IGNORED), None)
    return {
        "scenario": scenario,
        "descriptive_primary_beats_x_ignored_on_both_metrics": bool(
            primary and ignored and primary["w2"] < ignored["w2"]
            and primary["label_tv"] < ignored["label_tv"]),
        "conditioning_established": False if smoke else None,
        "criterion": "descriptive only: raw primary integrated W2 and TV both below X-ignored control",
        "caveat": "A smoke test is not evidence of conditioning; a false flag requires investigation, not a superiority claim.",
    }


def evaluate_unit(args, scenario, master, folder, data, fitted, budget):
    train_x, train_y, _, _, ref_y = data
    encoders, models, decoder_sets, selected_h, empirical_h, extras, fit_records = fitted
    x_grid = (np.arange(args.test_conditions) + .5) / args.test_conditions
    count = max(args.eval_size, args.probability_size)
    rows, runtime, margins = [], [], []

    def evaluate_samples(samples, method, stage, draw, true_ot, h=0., selected=True,
                         sigma=0., raw_extra=None):
        budget.check()
        validate_points(samples.reshape(-1, 3))
        branches, radii = jr_arrays(samples)
        name = f"samples_{artifact_stem(method)}_h{h:g}_s{sigma:g}_d{draw}.npz"
        np.savez_compressed(folder / name, conditions=x_grid, points=samples, J=branches, R=radii,
                            method=method, stage=stage, perturbation_sigma=sigma, h_ratio=h)
        for a in THRESHOLDS:
            budget.check()
            processed = threshold_points(samples.reshape(-1, 3), a).reshape(samples.shape)
            for index, x in enumerate(x_grid):
                subset = processed[index]
                record = {
                    "experiment": EXPERIMENT, "scenario": scenario, "seed": master,
                    "method": method, "stage": stage, "draw": draw, "x": float(x),
                    "h_ratio": h, "selected": selected, "threshold": a,
                    "perturbation_sigma": sigma,
                    "w2": intrinsic_w2(subset[:args.eval_size], true_ot[index]),
                    "ot_sample_count_per_law": args.eval_size,
                    **probability_record(subset[:args.probability_size], x, scenario),
                }
                if raw_extra:
                    record.update(raw_extra)
                rows.append(record)
        write_csv(folder / "per_x_metrics.csv", rows)

    for draw in range(args.evaluation_draws):
        print(f"{scenario}, seed {master}: evaluation draw {draw + 1}", flush=True)
        true_ot = np.asarray([
            sample_pairs(args.eval_size, scenario, stream_seed(master, scenario, 40, draw * 1000 + i),
                         x=[x])[1] for i, x in enumerate(x_grid)])
        true_second = np.asarray([
            sample_pairs(args.eval_size, scenario, stream_seed(master, scenario, 41, draw * 1000 + i),
                         x=[x])[1] for i, x in enumerate(x_grid)])
        first_j, first_r = jr_arrays(true_ot)
        second_j, second_r = jr_arrays(true_second)
        np.savez_compressed(folder / f"untouched_true_ot_draw{draw}.npz",
                            conditions=x_grid, first=true_ot, second=true_second,
                            first_J=first_j, first_R=first_r, second_J=second_j, second_R=second_r)
        for i, x in enumerate(x_grid):
            # Probability reference uses an independent, larger true sample.
            probability_truth = sample_pairs(
                args.probability_size, scenario,
                stream_seed(master, scenario, 42, draw * 1000 + i), x=[x])[1]
            rows.append({
                "experiment": EXPERIMENT, "scenario": scenario, "seed": master,
                "method": "real_real_mc_reference", "stage": "monte_carlo_reference",
                "draw": draw, "x": float(x), "h_ratio": 0., "selected": True,
                "threshold": 0., "perturbation_sigma": 0.,
                "ot_sample_count_per_law": args.eval_size,
                "w2": intrinsic_w2(true_second[i], true_ot[i]),
                **probability_record(probability_truth, x, scenario),
            })
        cached_coordinates = {}
        sample_roles = {"medoid_k3": 50, "x_ignored": 51, "direct": 52,
                        "medoid_k6": 53, "medoid_k12": 54, "frechet_base_k3": 55}
        for tag, model in models.items():
            budget.check()
            start = time.perf_counter()
            coordinates = sample_ddpm(
                model, x_grid, count, args.timesteps,
                stream_seed(master, scenario, sample_roles[tag], draw), budget.check)
            elapsed = time.perf_counter() - start
            runtime.append({"draw": draw, "source": tag, "phase": "sampling", "seconds": elapsed,
                            "sample_count": len(x_grid) * count})
            np.savez_compressed(folder / f"latent_{tag}_draw{draw}.npz",
                                conditions=x_grid, coordinates=coordinates)
            cached_coordinates[tag] = coordinates
            if tag == "direct":
                start = time.perf_counter()
                projected, displacement = project_planar(coordinates.reshape(-1, 2))
                runtime.append({"draw": draw, "source": tag, "phase": "projection",
                                "seconds": time.perf_counter() - start})
                evaluate_samples(projected.reshape(len(x_grid), count, 3), METHOD_DIRECT,
                                 "learned_generator", draw, true_ot,
                                 raw_extra={"ambient_mean_projection_displacement": float(displacement.mean())})
                continue
            encoder_tag = "medoid_k3" if tag == "x_ignored" else tag
            method = (METHOD_IGNORED if tag == "x_ignored" else METHOD_COORD
                      if tag == "medoid_k3" else f"{METHOD_COORD}__{tag}")
            for decoder in decoder_sets[encoder_tag]:
                start = time.perf_counter()
                decoded = decode_array(decoder, coordinates)
                runtime.append({"draw": draw, "source": method, "h_ratio": decoder.h_ratio,
                                "phase": "decoding", "seconds": time.perf_counter() - start})
                margins.append({"method": method, "draw": draw, "h_ratio": decoder.h_ratio,
                                **decoder_diagnostics(decoder, coordinates)})
                evaluate_samples(decoded, method, "learned_generator", draw, true_ot,
                                 h=decoder.h_ratio, selected=decoder.h_ratio == selected_h[tag])
        start = time.perf_counter()
        empirical = conditional_resample(train_x, train_y, x_grid, count, empirical_h,
                                         stream_seed(master, scenario, 60, draw))
        runtime.append({"draw": draw, "source": METHOD_EMPIRICAL, "phase": "sampling",
                        "seconds": time.perf_counter() - start})
        evaluate_samples(empirical, METHOD_EMPIRICAL, "learned_generator", draw, true_ot)
        if args.second_stage:
            for decoder in extras["learned"]:
                start = time.perf_counter()
                decoded = decode_array(decoder, cached_coordinates["medoid_k3"])
                runtime.append({"draw": draw, "source": "conditional_coordinate_ddpm_bounded_residual_frechet",
                                "h_ratio": decoder.h_ratio, "phase": "decoding",
                                "seconds": time.perf_counter() - start})
                margins.append({"method": "conditional_coordinate_ddpm_bounded_residual_frechet",
                                "draw": draw, "h_ratio": decoder.h_ratio,
                                **decoder_diagnostics(decoder, cached_coordinates["medoid_k3"])})
                evaluate_samples(decoded, "conditional_coordinate_ddpm_bounded_residual_frechet",
                                 "learned_generator", draw, true_ot, h=decoder.h_ratio,
                                 selected=decoder.h_ratio == selected_h["learned"])
            start = time.perf_counter()
            mlp = decode_array(extras["mlp"], cached_coordinates["medoid_k3"])
            runtime.append({"draw": draw, "source": "coordinate_mlp_explicit_segment_projection",
                            "phase": "decoding", "seconds": time.perf_counter() - start})
            evaluate_samples(mlp, "coordinate_mlp_explicit_segment_projection",
                             "learned_generator", draw, true_ot)
            start = time.perf_counter()
            categorical = extras["categorical"].sample(x_grid, count, stream_seed(master, scenario, 61, draw))
            runtime.append({"draw": draw, "source": "conditional_categorical_radial_histogram",
                            "phase": "sampling", "seconds": time.perf_counter() - start})
            evaluate_samples(categorical, "conditional_categorical_radial_histogram",
                             "learned_generator", draw, true_ot)
        # These are fresh source draws, never thresholded ground-truth OT samples.
        oracle_y = np.asarray([
            sample_pairs(count, scenario, stream_seed(master, scenario, 70, draw * 1000 + i),
                         x=[x])[1] for i, x in enumerate(x_grid)])
        evaluate_samples(oracle_y, "true_source_threshold_bias_diagnostic",
                         "known_law_threshold_diagnostic", draw, true_ot)
        for encoder_name, encoder in (("primary_sample_medoid", encoders["medoid_k3"]),
                                      ("controlled_center_endpoints", StarEncoder.controlled())):
            with torch.no_grad():
                z = encoder(tensor(oracle_y.reshape(-1, 3))).numpy().reshape(len(x_grid), count, -1)
            rng = np.random.default_rng(stream_seed(master, scenario, 71, draw))
            noise = rng.normal(size=z.shape)
            for sigma in args.perturbations:
                coordinates = z + sigma * noise
                for h in H_RATIOS:
                    decoder = GaussianFrechet(encoder, ref_y, h)
                    start = time.perf_counter()
                    decoded = decode_array(decoder, coordinates)
                    runtime.append({"draw": draw, "source": f"oracle_{encoder_name}", "h_ratio": h,
                                    "perturbation_sigma": sigma, "phase": "decoding",
                                    "seconds": time.perf_counter() - start})
                    margins.append({"method": f"oracle_{encoder_name}", "draw": draw, "h_ratio": h,
                                    "perturbation_sigma": sigma,
                                    **decoder_diagnostics(decoder, coordinates)})
                    stage = "oracle_coordinate" if sigma == 0 else "perturbed_coordinate"
                    evaluate_samples(decoded, f"oracle_{encoder_name}_gaussian_frechet", stage,
                                     draw, true_ot, h=h, selected=h == .10, sigma=sigma)
    for method, record in fit_records.items():
        if "fit_seconds" in record:
            runtime.append({"source": method, "phase": "fitting_including_validation",
                            "seconds": record["fit_seconds"]})
        for phase in ("optimization_seconds", "validation_seconds"):
            if phase in record:
                runtime.append({"source": method, "phase": phase, "seconds": record[phase]})
    write_csv(folder / "runtime.csv", runtime)
    write_csv(folder / "decoder_sign_margins.csv", margins)
    write_csv(folder / "per_x_metrics.csv", rows)
    write_csv(folder / "integrated_metrics.csv", summarize_rows(rows))
    write_csv(folder / "selected_integrated_metrics.csv", summarize_selected(rows))
    write_csv(folder / "paired_comparisons.csv", paired_comparisons(rows))
    write_csv(folder / "integrated_paired_comparisons.csv", integrated_paired_comparisons(rows))
    save_json(folder / "conditioning_diagnostic.json", diagnostic_summary(rows, scenario, args.smoke))
    return rows


def result_plots(folder, rows):
    selected = [r for r in rows if r["selected"] and r["stage"] == "learned_generator"]
    if not selected:
        return
    methods = sorted({r["method"] for r in selected})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fields = ("category_0_probability", "arm_1_probability", "arm_2_probability", "arm_3_probability")
    for axis, field in zip(axes.flat, fields):
        for method in methods:
            subset = [r for r in selected if r["method"] == method and r["threshold"] == 0]
            x = sorted({r["x"] for r in subset})
            means = [np.mean([r[field] for r in subset if r["x"] == value]) for value in x]
            axis.plot(x, means, marker=".", label=METHOD_NAMES.get(method, method))
        subset = [r for r in selected if r["method"] == methods[0] and r["threshold"] == 0]
        x = sorted({r["x"] for r in subset})
        truth_field = field.replace("_probability", "_truth")
        axis.plot(x, [next(r[truth_field] for r in subset if r["x"] == v) for v in x],
                  "k--", label="Analytic truth")
        axis.set(xlabel="X", ylabel=field, ylim=(-.02, 1.02))
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Conditional metric-star pilot: raw conditional probabilities")
    fig.tight_layout()
    fig.savefig(folder / "conditional_probabilities.png", dpi=150)
    plt.close(fig)
    integrated = summarize_rows(selected)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, field in zip(axes, ("w2", "label_tv", "center_absolute_error")):
        for method in methods:
            group = sorted([r for r in integrated if r["method"] == method], key=lambda r: r["threshold"])
            axis.plot([r["threshold"] for r in group], [r[field] for r in group],
                      marker=".", label=METHOD_NAMES.get(method, method))
        axis.set(xlabel="Full-vertex threshold a", ylabel=field)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(folder / "raw_threshold_sensitivity.png", dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 4))
    for stage in ("learned_generator", "oracle_coordinate"):
        subset = [r for r in rows if r["stage"] == stage and r["threshold"] == 0
                  and r["method"] in (METHOD_COORD, "oracle_primary_sample_medoid_gaussian_frechet")]
        for metric in ("w2", "center_absolute_error"):
            summary = summarize_rows(subset)
            summary.sort(key=lambda r: r["h_ratio"])
            axis.plot([r["h_ratio"] for r in summary], [r[metric] for r in summary],
                      marker=".", label=f"{stage}: {metric}")
    axis.set(xlabel="h / h_scale", ylabel="Integrated error")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "bandwidth_sensitivity.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x_values = sorted({r["x"] for r in selected})
    x = x_values[len(x_values) // 2]
    for j, axis in enumerate(axes, start=1):
        for method in methods:
            group = [r for r in selected if r["method"] == method and r["x"] == x and r["threshold"] == 0]
            axis.plot(RADIUS_GRID, [np.mean([r[f"arm_{j}_cdf_{t:.1f}"] for r in group])
                                   for t in RADIUS_GRID], label=METHOD_NAMES.get(method, method))
        truth = next(r for r in selected if r["x"] == x)
        axis.plot(RADIUS_GRID, [truth[f"arm_{j}_cdf_truth_{t:.1f}"] for t in RADIUS_GRID], "k--")
        axis.set(xlabel="Radius", ylabel=f"P(J={j}, 0<R<=t)", title=f"X={x:.3f}")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(folder / "conditional_radius_cdfs.png", dpi=150)
    plt.close(fig)
    pairs = paired_comparisons(selected)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, field in zip(axes, ("primary_minus_comparator_w2", "primary_minus_comparator_tv")):
        for method in sorted({r["comparator"] for r in pairs}):
            group = [r for r in pairs if r["comparator"] == method and r["threshold"] == 0]
            x = sorted({r["x"] for r in group})
            axis.plot(x, [np.mean([r[field] for r in group if r["x"] == v]) for v in x],
                      marker=".", label=METHOD_NAMES.get(method, method))
        axis.axhline(0, color="black", linewidth=.7)
        axis.set(xlabel="X", ylabel=field)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(folder / "paired_w2_tv_comparisons.png", dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 105)))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--second-stage", action="store_true",
                        help="Opt in to k=6/12, Frechet base, learned weights, projected MLP and label-aware baseline.")
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--reference-size", type=int, default=512)
    parser.add_argument("--train-size", type=int, default=4096, choices=(1024, 4096, 8192))
    parser.add_argument("--validation-size", type=int, default=1024)
    parser.add_argument("--validation-conditions", type=int, default=256)
    parser.add_argument("--validation-draws", type=int, default=128)
    parser.add_argument("--test-conditions", type=int, default=21)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--probability-size", type=int, default=4096)
    parser.add_argument("--evaluation-draws", type=int, default=3)
    parser.add_argument("--updates", type=int, default=10000)
    parser.add_argument("--decoder-updates", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--x-bandwidths", type=float, nargs="+", default=list(H_RATIOS))
    parser.add_argument("--perturbations", type=float, nargs="+", default=[0., .02, .10])
    args = parser.parse_args(argv)
    if args.smoke:
        args.seeds = [args.seeds[0]]
        for name, value in {
            "reference_size": 48, "train_size": 128, "validation_size": 48,
            "validation_conditions": 8, "validation_draws": 16, "test_conditions": 3,
            "eval_size": 32, "probability_size": 64, "evaluation_draws": 1,
            "updates": 40, "decoder_updates": 40, "checkpoint_every": 20,
            "timesteps": 12, "batch_size": 32, "width": 32,
        }.items():
            setattr(args, name, value)
    counts = ("reference_size", "train_size", "validation_size", "validation_conditions",
              "validation_draws", "test_conditions", "eval_size", "probability_size",
              "evaluation_draws", "updates", "decoder_updates", "checkpoint_every",
              "timesteps", "batch_size", "width", "threads")
    if any(getattr(args, name) < 1 for name in counts) or args.reference_size < 4:
        parser.error("Counts must be positive; at least four references are required.")
    if (args.validation_conditions > args.validation_size or args.timesteps < 2
            or args.max_seconds <= 0 or not np.isfinite(args.max_seconds)
            or any(not np.isfinite(h) or h <= 0 for h in args.x_bandwidths)
            or any(not np.isfinite(s) or s < 0 for s in args.perturbations)
            or 0. not in args.perturbations or any(seed < 0 for seed in args.seeds)
            or len(set(args.seeds)) != len(args.seeds)
            or len(set(args.scenarios)) != len(args.scenarios)):
        parser.error("Invalid validation, budget, bandwidth, perturbation or seed configuration.")
    if args.second_stage and args.reference_size < 13:
        parser.error("k=12 requires at least 13 reference observations.")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new empty output directory; existing experimental records are never overwritten.")
    args.output.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.max_seconds)
    torch.set_num_threads(args.threads)
    config = {**vars(args), "output": str(args.output), "experiment": EXPERIMENT,
              "status": "smoke_development_validation" if args.smoke else "exploratory_pilot",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": source_hashes(),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "platform": platform.platform()},
              "thresholds": THRESHOLDS, "h_ratios": H_RATIOS,
              "validation_event_radius_grid": RADIUS_GRID.tolist(),
              "validation_score": "sum of Brier losses for center/arm labels and arm-radius events; raw outputs only",
              "validation_selection": "checkpoint and h selected jointly using held-out pairs, no test law",
              "probability_mc_uncertainty": "binomial pointwise standard errors; not training uncertainty",
              "schedule": {"type": "linear_beta", "start": .0001, "end": .08,
                           "reverse_variance": "posterior", "prediction": "unbounded_epsilon",
                           "time_probability": "uniform over all discrete forward steps"},
              "optimizer": {"name": "Adam", "learning_rate": .001,
                            "randomness": "initialization and optimizer draws use separate seeded streams"},
              "appendix_j": {"implemented": False, "reason": "exact cited revised manuscript unavailable",
                             "practical_sampler_not_certified": True},
              "deferred": ["oracle-denoiser quadrature", "Appendix-J aligned sampler"],
              "second_stage_executed": args.second_stage}
    save_json(args.output / "config.json", config)
    all_rows, base_coincidences = [], []
    try:
        for scenario in args.scenarios:
            for master in args.seeds:
                budget.check()
                folder = args.output / scenario / f"seed_{master}"
                folder.mkdir(parents=True)
                save_json(folder / "config.json", {**config, "scenarios": [scenario], "seeds": [master],
                          "streams": {name: stream_seed(master, scenario, role) for name, role in
                                      (("reference", 1), ("training", 2), ("validation", 3))}})
                ref_x, ref_y = sample_pairs(args.reference_size, scenario, stream_seed(master, scenario, 1))
                train_x, train_y = sample_pairs(args.train_size, scenario, stream_seed(master, scenario, 2))
                val_x, val_y = sample_pairs(args.validation_size, scenario, stream_seed(master, scenario, 3))
                ref_j, ref_r = jr_arrays(ref_y)
                train_j, train_r = jr_arrays(train_y)
                val_j, val_r = jr_arrays(val_y)
                np.savez_compressed(folder / "independent_data.npz", reference_x=ref_x, reference_y=ref_y,
                                    reference_J=ref_j, reference_R=ref_r, train_J=train_j, train_R=train_r,
                                    validation_J=val_j, validation_R=val_r,
                                    train_x=train_x, train_y=train_y, validation_x=val_x, validation_y=val_y)
                data = (train_x, train_y, val_x, val_y, ref_y)
                save_json(args.output / "progress.json", {"scenario": scenario, "seed": master,
                                                         "phase": "geometry_and_oracle_diagnostics"})
                primary_encoder = StarEncoder(ref_y)
                reference_plot(folder, ref_y, {"medoid_k3": primary_encoder})
                reconstruction_diagnostics(folder, ref_y, val_y, primary_encoder, budget)
                base_coincidences.append({
                    "scenario": scenario, "seed": master,
                    "medoid_equals_frechet_mean": primary_encoder.metadata["medoid_equals_frechet_mean"]})
                write_csv(args.output / "base_coincidences.csv", base_coincidences)
                save_json(args.output / "progress.json", {"scenario": scenario, "seed": master, "phase": "fitting"})
                fitted = fit_unit(args, scenario, master, folder, data, budget, primary_encoder)
                save_json(args.output / "progress.json", {"scenario": scenario, "seed": master, "phase": "evaluation"})
                rows = evaluate_unit(args, scenario, master, folder, data, fitted, budget)
                all_rows.extend(rows)
                result_plots(folder, rows)
                save_json(folder / "completion.json", {"status": config["status"], "complete": True,
                          "seconds_since_run_start": time.perf_counter() - budget.started,
                          "row_count": len(rows)})
                write_csv(args.output / "per_x_metrics.csv", all_rows)
        write_csv(args.output / "integrated_metrics.csv", summarize_rows(all_rows))
        write_csv(args.output / "selected_integrated_metrics.csv", summarize_selected(all_rows))
        write_csv(args.output / "paired_comparisons.csv", paired_comparisons(all_rows))
        write_csv(args.output / "integrated_paired_comparisons.csv", integrated_paired_comparisons(all_rows))
        save_json(args.output / "completion.json", {
            "status": config["status"], "complete": True, "row_count": len(all_rows),
            "wall_seconds": time.perf_counter() - budget.started,
            "independent_training_seeds": args.seeds, "second_stage_executed": args.second_stage,
            "medoid_frechet_mean_coincidence_frequency": float(np.mean([
                r["medoid_equals_frechet_mean"] for r in base_coincidences])),
            "scientific_claim": "none; smoke completion verifies execution, not statistical accuracy"})
        print(f"Completed {EXPERIMENT}: {args.output}", flush=True)
    except Exception as error:
        save_json(args.output / "last_failure.json", {
            "exception": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(),
            "wall_seconds": time.perf_counter() - budget.started})
        raise


if __name__ == "__main__":
    main()
