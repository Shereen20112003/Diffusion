"""Repeated-seed original-protocol studies; no conditional-generation extension."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import traceback

import numpy as np
from scipy.stats import t as student_t
import torch

from run_experiments import (
    Encoder, LANDMARKS, LOSSES, choose_references, curve_plot, fit_decoder,
    fit_diffusion, reconstruction_loss, sample_diffusion, save_json,
    scatter_plot, write_csv,
)
from metric_spaces import (
    ALL_DATASETS, METRIC_DATASETS, FrechetDecoder, MetricEncoder, ObjectSpace,
    fps_indices, metric_loss, object_diagnostics, object_w2,
)
import matplotlib.pyplot as plt


def stream_seed(seed, dataset, role, draw=0):
    sequence = np.random.SeedSequence([20261101, seed, ALL_DATASETS.index(dataset), role, draw])
    return int(sequence.generate_state(1)[0] & 0x7fffffff)


def progress(output, dataset, seed, phase):
    value = {"dataset": dataset, "seed": seed, "phase": phase,
             "updated_utc": datetime.now(timezone.utc).isoformat()}
    save_json(output / "progress.json", value)
    print(f"{dataset}, seed {seed}: {phase}", flush=True)


def make_encoder(train, dataset, rule, seed, args, space):
    if dataset in METRIC_DATASETS:
        encoder = MetricEncoder(train, space, args.metric_landmarks)
        indices = encoder.indices
    else:
        base, landmarks = choose_references(train, rule, stream_seed(seed, dataset, 10))
        encoder = Encoder(train, base, landmarks)
        indices = [int(base), *landmarks.tolist()]
    return encoder, indices


def fit_reconstruction_models(train, validation, encoder, references, space, args, seed, folder):
    loss_function = metric_loss if space.dataset in METRIC_DATASETS else reconstruction_loss
    models, records, histories = {}, {}, {}
    kinds = [] if space.dataset in METRIC_DATASETS else args.losses
    for kind in kinds:
        best, selected, total = float("inf"), None, 0.0
        candidates = []
        for delta in args.huber_deltas if kind == "huber" else [1.0]:
            model, history, score, elapsed = fit_decoder(
                train, validation, encoder, kind, delta, args, seed, torch.device("cpu")
            )
            total += elapsed
            candidates.append({"delta": delta, "selection_score": score, "history": history,
                               "training_seconds": elapsed})
            if score < best:
                best, selected = score, (model, history, delta)
        model, history, delta = selected
        models[kind], histories[kind] = model, history
        records[kind] = {"selected_delta": delta if kind == "huber" else None,
                         "selection_score": best, "training_seconds_including_tuning": total,
                         "candidate_count": len(candidates), "candidates": candidates}
    val = torch.as_tensor(validation)
    with torch.no_grad():
        val_z = encoder(val)
    for learned, name in ((False, "kernel_frechet"), (True, "learned_frechet")):
        best, selected, total = float("inf"), None, 0.0
        candidates = []
        for bandwidth in args.bandwidths:
            started = time.perf_counter()

            def factory():
                return FrechetDecoder(encoder, references, space, bandwidth, args.width, learned)

            if learned:
                model, history, score, elapsed = fit_decoder(
                    train, validation, encoder, "embedded_mse", 1.0, args, seed,
                    torch.device("cpu"), model_factory=factory, loss_function=loss_function,
                )
            else:
                model, history = factory().eval(), []
                with torch.no_grad():
                    score = loss_function(model(val_z), val, encoder, "embedded_mse").item()
                elapsed = time.perf_counter() - started
            if not math.isfinite(score):
                raise FloatingPointError(f"Nonfinite {name} selection score.")
            total += elapsed
            candidates.append({"bandwidth": bandwidth, "selection_score": score,
                               "training_seconds": elapsed, "history": history})
            if score < best:
                best, selected = score, (model, history, bandwidth)
        model, history, bandwidth = selected
        models[name] = model
        if history:
            histories[name] = history
        records[name] = {"selected_bandwidth": bandwidth, "selection_score": best,
                         "training_seconds_including_tuning": total,
                         "candidate_count": len(candidates), "candidates": candidates}
    for name, model in models.items():
        records[name]["parameters"] = sum(p.numel() for p in model.parameters())
        torch.save({"model": model.state_dict(), "encoder": encoder.state_dict()},
                   folder / f"{name}.pt")
    save_json(folder / "decoder_selection.json", records)
    return models, records, histories


def plot_objects(path, panels, space, title):
    if space.dataset not in METRIC_DATASETS:
        scatter_plot(path, panels, title)
        return
    columns = min(3, len(panels))
    rows = math.ceil(len(panels) / columns)
    figure = plt.figure(figsize=(4.6 * columns, 4.4 * rows))
    lower = min(float(points.min()) for _, points in panels)
    upper = max(float(points.max()) for _, points in panels)
    for index, (label, points) in enumerate(panels):
        axis = figure.add_subplot(rows, columns, index + 1,
                                  projection="3d" if space.dataset == "sphere" else None)
        if space.dataset == "sphere":
            axis.scatter(*points.T, s=6, alpha=0.65)
            axis.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                     xlabel="x", ylabel="y", zlabel="z")
            axis.set_box_aspect((1, 1, 1))
        elif space.dataset == "tree":
            angles = np.arange(3) * 2 * np.pi / 3
            ends = np.column_stack([np.cos(angles), np.sin(angles)])
            for endpoint in ends:
                axis.plot([0, endpoint[0]], [0, endpoint[1]], color="gray")
            xy = points @ ends
            axis.scatter(*xy.T, s=7, alpha=0.35)
            vertex_mass = float((points.sum(axis=1) <= 1e-7).mean())
            axis.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1),
                     xlabel=f"Vertex mass: {vertex_mass:.3f}")
            axis.set_aspect("equal")
        else:
            u = (np.arange(space.quantiles) + 0.5) / space.quantiles
            for quantile in points[:40]:
                axis.plot(u, quantile, alpha=0.25)
            axis.set(xlabel="Quantile probability", ylabel="Atom location",
                     ylim=(lower - 0.1, upper + 0.1))
        axis.set_title(label)
    figure.suptitle(title)
    figure.tight_layout()
    for suffix in (".png", ".pdf"):
        figure.savefig(path.with_suffix(suffix), dpi=160)
    plt.close(figure)


def summarize(rows, output):
    summaries = []
    keys = sorted({(r["dataset"], r["landmarks"], r["method"]) for r in rows})
    direct = {(r["dataset"], r["seed"], r["draw"]): r["w2"]
              for r in rows if r["method"] == "direct_ddpm"}
    for dataset, landmarks, method in keys:
        group = [r for r in rows if (r["dataset"], r["landmarks"], r["method"])
                 == (dataset, landmarks, method)]
        seeds = sorted({r["seed"] for r in group})
        seed_means, seed_deltas = [], []
        for seed in seeds:
            subset = [r for r in group if r["seed"] == seed]
            seed_means.append(np.mean([r["w2"] for r in subset]))
            seed_deltas.append(np.mean([
                r["w2"] - direct[(dataset, seed, r["draw"])] for r in subset
            ]))
        mean = float(np.mean(seed_means))
        sd = float(np.std(seed_means, ddof=1)) if len(seeds) > 1 else None
        half = (float(student_t.ppf(0.975, len(seeds) - 1) * sd / np.sqrt(len(seeds)))
                if sd is not None else None)
        record = {
            "dataset": dataset, "landmarks": landmarks, "method": method,
            "independent_training_seeds": len(seeds), "evaluation_rows": len(group),
            "w2_mean": mean, "w2_between_seed_sd": sd,
            "w2_seed_mean_95pct_t_interval_lower": mean - half if half is not None else None,
            "w2_seed_mean_95pct_t_interval_upper": mean + half if half is not None else None,
            "paired_delta_w2_vs_direct_mean": float(np.mean(seed_deltas)),
            "paired_delta_between_seed_sd": (
                float(np.std(seed_deltas, ddof=1)) if len(seeds) > 1 else None
            ),
        }
        scalar_fields = (
            "object_reconstruction_mse", "embedded_reconstruction_mse",
            "generated_localization_mean", "generated_nearest_reference_mean",
            "sphere_mean_log_residual_max", "vertex_mass", "branch_vertex_total_variation",
            "fraction_outside_data_cap", "mean_distance_to_nearest_center",
            "mean_distance_to_clean_arcs", "fraction_farther_than_0.45_from_centers",
            "fraction_farther_than_0.15_from_arcs",
            "diffusion_train_seconds", "decoder_train_seconds_including_tuning",
            "sampling_seconds", "decode_and_reconstruction_seconds",
            "ambient_postprocess_mean_l2_change",
        )
        for field in scalar_fields:
            if all(field in row for row in group):
                values = [np.mean([row[field] for row in group if row["seed"] == seed])
                          for seed in seeds]
                record[f"{field}_mean"] = float(np.mean(values))
                record[f"{field}_between_seed_sd"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                )
        summaries.append(record)
    write_csv(output / "summary.csv", summaries)
    save_json(output / "summary.json", summaries)
    for dataset in sorted({row["dataset"] for row in summaries}):
        subset = [row for row in summaries if row["dataset"] == dataset]
        figure, axis = plt.subplots(figsize=(10, max(4, len(subset) * 0.3)))
        axis.barh(
            [f'{r["landmarks"]}: {r["method"]}' for r in subset],
            [r["w2_mean"] for r in subset],
            xerr=[r["w2_between_seed_sd"] or 0 for r in subset],
        )
        axis.set(xlabel="Object-metric empirical W2; error bars = between-seed SD",
                 title=f"{dataset}: average evaluation draws within each seed first")
        figure.tight_layout()
        figure.savefig(output / f"{dataset}_w2.png", dpi=160)
        plt.close(figure)
    return summaries


def run_unit(args, dataset, seed, folder):
    device = torch.device("cpu")
    space = ObjectSpace(dataset, args.quantiles, args.sphere_iterations)
    train = space.sample(args.train_size, stream_seed(seed, dataset, 1))
    validation = space.sample(args.validation_size, stream_seed(seed, dataset, 2))
    space.validate(train)
    space.validate(validation)
    np.savez_compressed(folder / "fit_data.npz", train=train, validation=validation)
    mean, scale = train.mean(axis=0), np.clip(train.std(axis=0, ddof=1), 1e-3, 1e3)
    progress(args.output, dataset, seed, "training direct baseline")
    direct, direct_history, direct_time = fit_diffusion(
        (train - mean) / scale, (validation - mean) / scale,
        args, stream_seed(seed, dataset, 11), device,
    )
    torch.save({"model": direct.state_dict(), "mean": torch.as_tensor(mean),
                "scale": torch.as_tensor(scale)}, folder / "direct_ddpm.pt")
    save_json(folder / "direct_history.json", direct_history)
    rules = ["fps"] if dataset in METRIC_DATASETS else args.landmarks
    fitted = {}
    train_tensor = torch.as_tensor(train)
    for rule in rules:
        target = folder / rule
        target.mkdir()
        encoder, reference_indices = make_encoder(train, dataset, rule, seed, args, space)
        reconstruction_indices = fps_indices(train_tensor, space, args.reconstruction_count)
        references = train_tensor[reconstruction_indices]
        with torch.no_grad():
            train_z, val_z = encoder(train_tensor).numpy(), encoder(torch.as_tensor(validation)).numpy()
        reference_record = {
            "base_and_encoding_indices": reference_indices,
            "reconstruction_indices": reconstruction_indices.tolist(),
            "coordinate_mean": encoder.mean.tolist(), "coordinate_scale": encoder.scale.tolist(),
            "encoding_landmark_count": len(encoder.mean),
            "reconstruction_object_count": len(references),
            "base_rule": "first training object" if dataset in METRIC_DATASETS
                         else "training object nearest coordinate-wise median",
        }
        if dataset not in METRIC_DATASETS:
            matrix = encoder.directions.numpy()
            reference_record.update({
                "rank": int(np.linalg.matrix_rank(matrix)),
                "condition_number_raw": float(np.linalg.cond(matrix)),
                "condition_number_standardized": float(np.linalg.cond(matrix / encoder.scale.numpy()[:, None])),
            })
        save_json(target / "references.json", reference_record)
        progress(args.output, dataset, seed, f"training {rule} coordinate DDPM")
        coordinate, coordinate_history, coordinate_time = fit_diffusion(
            train_z, val_z, args, stream_seed(seed, dataset, 12), device,
        )
        torch.save({"model": coordinate.state_dict(), "encoder": encoder.state_dict()},
                   target / "coordinate_ddpm.pt")
        progress(args.output, dataset, seed, f"fitting {rule} reconstruction comparisons")
        models, records, histories = fit_reconstruction_models(
            train, validation, encoder, references, space, args,
            stream_seed(seed, dataset, 13), target,
        )
        curve_plot(target / "learning_curves.png",
                   {"direct": direct_history, "coordinate": coordinate_history}, histories)
        save_json(target / "coordinate_history.json", coordinate_history)
        fitted[rule] = (encoder, coordinate, coordinate_time, models, records, reference_indices)
    # No evaluation observations or generated evaluation samples precede model selection.
    save_json(folder / "selection_complete.json", {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "all checkpoints, Huber thresholds and bandwidths fixed using fit validation",
        "evaluation_draws": args.eval_draws,
    })
    progress(args.output, dataset, seed, "evaluating frozen models on fresh independent draws")
    rows = []
    for draw in range(args.eval_draws):
        test = space.sample(args.eval_size, stream_seed(seed, dataset, 3, draw))
        real_reference = space.sample(args.eval_size, stream_seed(seed, dataset, 4, draw))
        draw_folder = folder / f"draw_{draw}"
        draw_folder.mkdir()
        np.savez_compressed(draw_folder / "evaluation_data.npz", test=test,
                            real_reference=real_reference)
        started = time.perf_counter()
        raw_direct = sample_diffusion(
            direct, args.eval_size, args, stream_seed(seed, dataset, 20, draw), device
        ) * torch.as_tensor(scale) + torch.as_tensor(mean)
        direct_samples = space.repair_ambient(raw_direct).numpy()
        direct_seconds = time.perf_counter() - started
        np.savez_compressed(draw_folder / "direct_samples.npz",
                            raw_ambient=raw_direct.numpy(), generated=direct_samples)
        bootstrap_rng = np.random.default_rng(stream_seed(seed, dataset, 21, draw))
        bootstrap = train[bootstrap_rng.integers(len(train), size=args.eval_size)]
        np.save(draw_folder / "empirical_resampling.npy", bootstrap)

        def evaluate(method, rule, samples, **extras):
            t0 = time.perf_counter()
            score = object_w2(samples, test, space)
            row = {"dataset": dataset, "seed": seed, "draw": draw, "landmarks": rule,
                   "method": method, "w2": score, "evaluation_seconds": time.perf_counter() - t0,
                   "artifact_directory": str(folder.relative_to(args.output)), **extras}
            diagnostic = object_diagnostics(samples, space)
            row.update({key: value for key, value in diagnostic.items()
                        if isinstance(value, (float, int))})
            rows.append(row)
            return score, diagnostic

        direct_score, direct_geometry = evaluate(
            "direct_ddpm", "not_applicable", direct_samples,
            diffusion_train_seconds=direct_time, sampling_seconds=direct_seconds,
            denoiser_parameters=sum(p.numel() for p in direct.parameters()),
            ambient_postprocess_mean_l2_change=float(
                (torch.as_tensor(direct_samples) - raw_direct).norm(dim=1).mean()
            ),
        )
        _, real_geometry = evaluate("real_real", "not_applicable", real_reference)
        evaluate("empirical_resampling", "not_applicable", bootstrap)
        save_json(draw_folder / "baseline_geometry.json", {
            "direct": direct_geometry, "real_reference": real_geometry,
            "test": object_diagnostics(test, space),
        })
        for rule, (encoder, coordinate, coordinate_time, models, records, reference_indices) in fitted.items():
            target = draw_folder / rule
            target.mkdir()
            started = time.perf_counter()
            generated_z = sample_diffusion(
                coordinate, args.eval_size, args, stream_seed(seed, dataset, 22, draw), device
            )
            coordinate_sampling_time = time.perf_counter() - started
            actual = torch.as_tensor(test)
            with torch.no_grad():
                actual_z = encoder(actual)
                empirical_cover = space.pairwise_squared(
                    actual, train_tensor[reference_indices]
                ).min(dim=1).values.sqrt().max().item()
            np.save(target / "generated_coordinates.npy", generated_z.numpy())
            panels = [("Held-out real", test),
                      (f"Direct baseline: W2={direct_score:.3f}", direct_samples)]
            for name, decoder in models.items():
                started = time.perf_counter()
                with torch.no_grad():
                    samples = decoder(generated_z)
                    reconstructed = decoder(actual_z)
                    embedded_error = (encoder.raw(reconstructed) - encoder.raw(actual)).square().sum(dim=1).mean()
                    object_error = space.squared(reconstructed, actual).mean()
                    diagnostics = (decoder.diagnostics(generated_z)
                                   if isinstance(decoder, FrechetDecoder) else {})
                    if isinstance(decoder, FrechetDecoder):
                        diagnostics["empirical_encoded_reference_cover_radius"] = float(
                            torch.cdist(actual_z, decoder.coordinates).min(dim=1).values.max()
                        )
                decode_seconds = time.perf_counter() - started
                samples, reconstructed = samples.numpy(), reconstructed.numpy()
                space.validate(reconstructed)
                if not (math.isfinite(float(object_error)) and math.isfinite(float(embedded_error))):
                    raise FloatingPointError("Nonfinite reconstruction metric.")
                np.savez_compressed(target / f"{name}_samples.npz", generated=samples,
                                    reconstructed=reconstructed)
                record = records[name]
                score, geometry = evaluate(
                    name, rule, samples,
                    diffusion_train_seconds=coordinate_time,
                    decoder_train_seconds_including_tuning=record["training_seconds_including_tuning"],
                    denoiser_parameters=sum(p.numel() for p in coordinate.parameters()),
                    decoder_parameters=record["parameters"],
                    selected_bandwidth=record.get("selected_bandwidth"),
                    selected_huber_delta=record.get("selected_delta"),
                    object_reconstruction_mse=float(object_error),
                    embedded_reconstruction_mse=float(embedded_error),
                    sampling_seconds=coordinate_sampling_time,
                    decode_and_reconstruction_seconds=decode_seconds,
                    empirical_encoding_cover_radius=empirical_cover, **diagnostics,
                )
                save_json(target / f"{name}_diagnostics.json", {**geometry, **diagnostics})
                panels.append((f"{name}: W2={score:.3f}", samples))
            if draw == 0:
                plot_objects(folder / rule / "comparison", panels, space,
                             f"{dataset}, {rule}, seed {seed}; frozen-model evaluation draw 0")
        write_csv(folder / "metrics.csv", rows)
        save_json(folder / "metrics.json", rows)
    expected = args.eval_draws * (
        5 if dataset in METRIC_DATASETS else 3 + len(args.landmarks) * (len(args.losses) + 2)
    )
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} evaluation rows, found {len(rows)}.")
    return rows


def configuration(args):
    sources = ("run_remaining_experiments.py", "metric_spaces.py",
               "run_experiments.py", "diagnose_training.py")
    return {
        "arguments": {key: value for key, value in vars(args).items()
                      if key not in ("output", "resume")},
        "study": "original_protocol_repeated_seed_study_no_conditional_generation",
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in sources
        },
        "spaces": {name: ObjectSpace(name, args.quantiles, args.sphere_iterations).description()
                   for name in args.datasets},
        "seed_scheme": "SeedSequence([20261101, training_seed, dataset_index, role, draw]); 31-bit state",
        "selection": "fit-validation only; common raw embedded MSE for decoder checkpoints/candidates",
        "evaluation": "three independent roles: training, fit-validation, fresh evaluation; equal-count exact assignment",
        "uncertainty": "average evaluation draws within training seed; SD and t interval across seed means",
        "learned_weights": "softmax(-coordinate_distance_squared/(2*h*h) + 2*tanh(MLP)); residual range <= 4",
        "scope": "four MLP losses and three landmark rules on planar data; primary-loss Frechet decoders with FPS on extensions",
        "baseline_caveat": "ambient repaired DDPM is not an official intrinsic-manifold baseline reproduction",
        "reference_coverage_caveat": "reported maxima on evaluation objects are empirical probes, not certified support suprema",
        "status_caveat": "completed computation does not imply a favorable outcome or paper readiness",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "device": "cpu", "threads": args.threads},
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=list(ALL_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104])
    parser.add_argument("--landmarks", nargs="+", choices=LANDMARKS, default=list(LANDMARKS))
    parser.add_argument("--losses", nargs="+", choices=LOSSES, default=list(LOSSES))
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--validation-size", type=int, default=1024)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--eval-draws", type=int, default=3)
    parser.add_argument("--diffusion-steps", type=int, default=10000)
    parser.add_argument("--decoder-steps", type=int, default=1000)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--spatial-frequencies", type=int, default=4)
    parser.add_argument("--huber-deltas", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    parser.add_argument("--bandwidths", nargs="+", type=float, default=[0.1, 0.3, 1.0])
    parser.add_argument("--reconstruction-count", type=int, default=64)
    parser.add_argument("--metric-landmarks", type=int, default=8)
    parser.add_argument("--quantiles", type=int, default=32)
    parser.add_argument("--sphere-iterations", type=int, default=24)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("train_size", "validation_size", "eval_size", "eval_draws", "diffusion_steps",
                 "decoder_steps", "batch_size", "width", "reconstruction_count",
                 "metric_landmarks", "sphere_iterations", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive.")
    if args.train_size < max(4, args.reconstruction_count, args.metric_landmarks + 1):
        parser.error("Training size must cover all distinct references.")
    if args.validation_size < 2 or args.timesteps < 100 or args.quantiles < 4:
        parser.error("Use >=2 validation objects, >=100 timesteps, >=4 quantiles.")
    if not 0 <= args.spatial_frequencies <= 8:
        parser.error("Spatial frequencies must lie between 0 and 8.")
    if any(seed < 0 or seed > 1000000 for seed in args.seeds):
        parser.error("Seeds must lie between 0 and 1000000.")
    for name in ("bandwidths", "huber_deltas"):
        if any(not math.isfinite(v) or v <= 0 for v in getattr(args, name)):
            parser.error(f"{name} must be positive and finite.")
    for name in ("datasets", "seeds", "landmarks", "losses", "bandwidths", "huber_deltas"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f"{name} must not contain duplicates.")
    return args


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config = configuration(args)
    if args.resume:
        previous = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("Resume requires identical arguments, environment and source hashes.")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        save_json(args.output / "config.json", config)
    rows, started = [], time.perf_counter()
    try:
        for dataset in args.datasets:
            for seed in args.seeds:
                root = args.output / dataset / f"seed_{seed}"
                root.mkdir(parents=True, exist_ok=True)
                marker = root / "completion.json"
                if marker.exists():
                    complete = json.loads(marker.read_text(encoding="utf-8"))
                    folder = root / complete["attempt"]
                    unit_rows = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                    if len(unit_rows) != complete["metric_rows"]:
                        raise ValueError("Completed unit metric count does not match its marker.")
                else:
                    attempt = 1
                    while (root / f"attempt_{attempt}").exists():
                        attempt += 1
                    folder = root / f"attempt_{attempt}"
                    folder.mkdir()
                    unit_started = time.perf_counter()
                    unit_rows = run_unit(args, dataset, seed, folder)
                    save_json(marker, {"status": "completed", "attempt": folder.name,
                                       "metric_rows": len(unit_rows),
                                       "wall_seconds": time.perf_counter() - unit_started})
                rows.extend(unit_rows)
                write_csv(args.output / "metrics.csv", rows)
                save_json(args.output / "metrics.json", rows)
                summarize(rows, args.output)
    except BaseException:
        save_json(args.output / "last_failure.json", {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "traceback": traceback.format_exc(),
            "recovery": "explicit --resume preserves completed units and old partial attempts",
        })
        raise
    save_json(args.output / "completion.json", {
        "status": "completed", "units": len(args.datasets) * len(args.seeds),
        "metric_rows": len(rows), "wall_seconds_this_invocation": time.perf_counter() - started,
        "independent_training_seeds": len(args.seeds), "evaluation_draws_per_seed": args.eval_draws,
        "interpretation": "computation complete; inspect effect sizes, geometry diagnostics and solver caveats",
    })
    progress(args.output, "all", "all", "completed")
    print(f"Completed study: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    run(parse_args())
