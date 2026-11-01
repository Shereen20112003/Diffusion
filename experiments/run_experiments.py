"""Reproducible CPU/GPU pilot for the experimental protocol in the manuscript."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch
from torch import nn


DATASETS = ("two_moons", "eight_gaussians")
LANDMARKS = ("manual", "random", "fps")
LOSSES = ("embedded_mse", "huber", "object_mse", "standardized_mse")
LABELS = {
    "embedded_mse": "Embedded MSE (primary)",
    "huber": "Embedded Huber",
    "object_mse": "Object MSE",
    "standardized_mse": "Standardized embedded MSE",
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_data(dataset, size, seed):
    """Independent draws; mixture membership is random, not forced balanced."""
    rng = np.random.default_rng(seed)
    if dataset == "two_moons":
        component = rng.integers(0, 2, size)
        angle = rng.uniform(0, np.pi, size)
        x, y = np.cos(angle), np.sin(angle)
        points = np.column_stack(
            [np.where(component == 0, x, 1 - x),
             np.where(component == 0, y, 0.5 - y)]
        )
        points += rng.normal(0, 0.05, points.shape)
    elif dataset == "eight_gaussians":
        component = rng.integers(0, 8, size)
        angles = component * (2 * np.pi / 8)
        points = 2 * np.column_stack([np.cos(angles), np.sin(angles)])
        points += rng.normal(0, 0.15, points.shape)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return points.astype(np.float32)


def choose_references(points, rule, seed):
    """A fixed central sample base and three distinct training landmarks."""
    center = np.median(points, axis=0)
    base_index = int(np.argmin(np.sum((points - center) ** 2, axis=1)))
    selected = [base_index]
    if rule == "manual":
        # Predeclared peripheral directions, not adjusted after seeing results.
        for angle in (0, 2 * np.pi / 3, 4 * np.pi / 3):
            direction = np.array([np.cos(angle), np.sin(angle)])
            score = (points - points[base_index]) @ direction
            score[selected] = -np.inf
            selected.append(int(np.argmax(score)))
    elif rule == "random":
        candidates = np.delete(np.arange(len(points)), base_index)
        selected.extend(np.random.default_rng(seed).choice(candidates, 3, replace=False))
    elif rule == "fps":
        nearest = np.full(len(points), np.inf)
        for _ in range(3):
            distance = np.sum((points - points[selected[-1]]) ** 2, axis=1)
            nearest = np.minimum(nearest, distance)
            nearest[selected] = -np.inf
            selected.append(int(np.argmax(nearest)))
    else:
        raise ValueError(f"Unknown landmark rule: {rule}")
    return base_index, np.asarray(selected[1:], dtype=int)


class Encoder(nn.Module):
    """Euclidean specialization of the paper's distance-contrast encoder."""

    def __init__(self, points, base_index, landmark_indices):
        super().__init__()
        base = points[base_index]
        displacement = points[landmark_indices] - base
        lengths = np.linalg.norm(displacement, axis=1)
        if np.any(lengths <= 1e-8):
            raise ValueError("A landmark coincides with the base.")
        directions = displacement / lengths[:, None]
        if np.linalg.matrix_rank(directions) != 2:
            raise ValueError("Landmark directions do not span the object space.")
        raw = (points - base) @ directions.T
        self.register_buffer("base", torch.tensor(base))
        self.register_buffer("directions", torch.tensor(directions))
        self.register_buffer("mean", torch.tensor(raw.mean(axis=0)))
        self.register_buffer(
            "scale", torch.tensor(np.clip(raw.std(axis=0, ddof=1), 1e-3, 1e3))
        )

    def raw(self, points):
        return (points - self.base) @ self.directions.T

    def forward(self, points):
        return (self.raw(points) - self.mean) / self.scale


class Denoiser(nn.Module):
    def __init__(self, dimension, width, spatial_frequencies=0):
        super().__init__()
        if spatial_frequencies < 0:
            raise ValueError("Spatial frequency count must be nonnegative.")
        self.spatial_frequencies = spatial_frequencies
        self.register_buffer("frequencies", torch.exp(-math.log(10000) * torch.arange(8) / 8))
        if spatial_frequencies:
            self.register_buffer("spatial_bands", math.pi * 2.0 ** torch.arange(spatial_frequencies))
        self.network = nn.Sequential(
            nn.Linear(dimension * (1 + 2 * spatial_frequencies) + 16, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, dimension),
        )

    def forward(self, coordinates, step):
        phase = step[:, None].float() * self.frequencies[None, :]
        embedding = torch.cat([phase.sin(), phase.cos()], dim=1)
        if self.spatial_frequencies:
            spatial = coordinates[:, :, None] * self.spatial_bands
            coordinates = torch.cat(
                [coordinates, spatial.sin().flatten(1), spatial.cos().flatten(1)], dim=1
            )
        return self.network(torch.cat([coordinates, embedding], dim=1))


def make_decoder(width):
    return nn.Sequential(
        nn.Linear(3, width), nn.SiLU(),
        nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2),
    )


def reconstruction_loss(predicted, target, encoder, kind, delta=1.0):
    residual = encoder.raw(predicted) - encoder.raw(target)
    if kind == "embedded_mse":
        penalty = residual.square()
    elif kind == "huber":
        magnitude = residual.abs()
        penalty = torch.where(
            magnitude <= delta, 0.5 * residual.square(),
            delta * (magnitude - 0.5 * delta),
        )
    elif kind == "object_mse":
        penalty = (predicted - target).square()
    elif kind == "standardized_mse":
        penalty = (residual / encoder.scale).square()
    else:
        raise ValueError(f"Unknown loss: {kind}")
    # Matches the manuscript: sum coordinates, then average observations.
    return penalty.sum(dim=1).mean()


def schedule(timesteps, device):
    beta = torch.linspace(1e-4, 0.08, timesteps, device=device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    previous = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]])
    posterior_variance = beta * (1 - previous) / (1 - alpha_bar)
    return beta, alpha, alpha_bar, posterior_variance


def forward_noise(clean, noise, steps, alpha_bar):
    a = alpha_bar[steps, None]
    return a.sqrt() * clean + (1 - a).sqrt() * noise


def reverse_mean(values, predicted_noise, step, beta, alpha, alpha_bar):
    return (
        values - beta[step] / (1 - alpha_bar[step]).sqrt() * predicted_noise
    ) / alpha[step].sqrt()


def fit_diffusion(train, validation, args, seed, device, checkpoints=(), on_checkpoint=None):
    if checkpoints and on_checkpoint is None:
        raise ValueError("A checkpoint callback is required when checkpoints are requested.")
    if any(step <= 0 or step > args.diffusion_steps for step in checkpoints):
        raise ValueError("Checkpoints must fall within the training budget.")
    seed_everything(seed)
    model = Denoiser(train.shape[1], args.width, getattr(args, "spatial_frequencies", 0)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train = torch.tensor(train, device=device)
    validation = torch.tensor(validation, device=device)
    _, _, alpha_bar, _ = schedule(args.timesteps, device)
    validation_rng = torch.Generator(device=device).manual_seed(seed + 10000)
    validation_steps = torch.randint(
        args.timesteps, (len(validation),), generator=validation_rng, device=device
    )
    validation_noise = torch.randn(
        validation.shape, generator=validation_rng, device=device
    )
    noisy_validation = forward_noise(validation, validation_noise, validation_steps, alpha_bar)
    history, best, best_state = [], float("inf"), None
    started = time.perf_counter()
    for step in range(1, args.diffusion_steps + 1):
        indices = torch.randint(len(train), (args.batch_size,), device=device)
        batch = train[indices]
        t = torch.randint(args.timesteps, (len(batch),), device=device)
        noise = torch.randn_like(batch)
        noisy = forward_noise(batch, noise, t, alpha_bar)
        loss = (model(noisy, t) - noise).square().sum(dim=1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite diffusion loss.")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.diffusion_steps or step in checkpoints:
            with torch.no_grad():
                score = (model(noisy_validation, validation_steps) - validation_noise)
                score = score.square().sum(dim=1).mean().item()
            history.append({"step": step, "train": loss.item(), "validation": score})
            if score < best:
                best, best_state = score, copy.deepcopy(model.state_dict())
        if step in checkpoints:
            snapshot = copy.deepcopy(model)
            snapshot.load_state_dict(best_state)
            on_checkpoint(step, snapshot.eval(), best)
    model.load_state_dict(best_state)
    return model.eval(), history, time.perf_counter() - started


@torch.no_grad()
def sample_diffusion(model, count, args, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    dimension = model.network[-1].out_features
    values = torch.randn((count, dimension), generator=generator, device=device)
    beta, alpha, alpha_bar, variance = schedule(args.timesteps, device)
    for step in range(args.timesteps - 1, -1, -1):
        steps = torch.full((count,), step, device=device, dtype=torch.long)
        prediction = model(values, steps)
        values = reverse_mean(values, prediction, step, beta, alpha, alpha_bar)
        if step > 0:
            values += variance[step].sqrt() * torch.randn(
                values.shape, generator=generator, device=device
            )
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite generated coordinates.")
    return values


def fit_decoder(train, validation, encoder, kind, delta, args, seed, device,
                *, model_factory=None, loss_function=reconstruction_loss):
    seed_everything(seed)
    model = (make_decoder(args.width) if model_factory is None else model_factory()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train, validation = torch.tensor(train, device=device), torch.tensor(validation, device=device)
    with torch.no_grad():
        train_z, validation_z = encoder(train), encoder(validation)
    best, best_state, history = float("inf"), None, []
    started = time.perf_counter()
    for step in range(1, args.decoder_steps + 1):
        indices = torch.randint(len(train), (args.batch_size,), device=device)
        prediction = model(train_z[indices])
        loss = loss_function(prediction, train[indices], encoder, kind, delta)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite reconstruction loss.")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.decoder_steps:
            with torch.no_grad():
                predicted = model(validation_z)
                own_loss = loss_function(predicted, validation, encoder, kind, delta).item()
                # Common validation criterion makes checkpoint selection comparable.
                score = loss_function(
                    predicted, validation, encoder, "embedded_mse"
                ).item()
            history.append({
                "step": step, "train": loss.item(),
                "validation": own_loss, "validation_embedded_mse": score,
            })
            if score < best:
                best, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), history, best, time.perf_counter() - started


def empirical_w2(generated, real):
    """Exact equal-mass empirical W2 for equally sized sample sets."""
    if generated.shape != real.shape or generated.ndim != 2 or len(real) == 0:
        raise ValueError("Exact assignment requires equal, nonempty sample shapes.")
    if not np.isfinite(generated).all() or not np.isfinite(real).all():
        raise ValueError("W2 inputs must be finite.")
    cost = cdist(generated, real, metric="sqeuclidean")
    rows, columns = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, columns].mean()))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scatter_plot(path, panels, title, base=None, landmarks=None):
    count = len(panels)
    columns = min(count, 3)
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 4.2 * rows), squeeze=False)
    all_points = np.concatenate([points for _, points in panels])
    if landmarks is not None:
        all_points = np.concatenate([all_points, landmarks, base[None, :]])
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    margin = np.maximum((upper - lower) * 0.08, 0.1)
    for axis, (label, points) in zip(axes.flat, panels):
        axis.scatter(points[:, 0], points[:, 1], s=5, alpha=0.65)
        if landmarks is not None:
            axis.scatter(landmarks[:, 0], landmarks[:, 1], c="red", marker="X", s=100, label="Landmarks")
            axis.scatter(*base, c="black", marker="*", s=120, label="Base")
            axis.legend()
        axis.set(xlim=(lower[0] - margin[0], upper[0] + margin[0]),
                 ylim=(lower[1] - margin[1], upper[1] + margin[1]), title=label,
                 xlabel="x", ylabel="y")
        axis.set_aspect("equal", adjustable="box")
    for axis in list(axes.flat)[count:]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.tight_layout()
    for suffix in (".png", ".pdf"):
        figure.savefig(path.with_suffix(suffix), dpi=180)
    plt.close(figure)


def curve_plot(path, diffusion_histories, decoder_histories):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, history in diffusion_histories.items():
        axes[0].plot([x["step"] for x in history], [x["validation"] for x in history], label=label)
    for label, history in decoder_histories.items():
        axes[1].plot([x["step"] for x in history],
                     [x["validation_embedded_mse"] for x in history], label=label)
    for axis, title in zip(axes, ("Diffusion validation noise error", "Decoder validation embedded MSE")):
        axis.set(title=title, xlabel="Optimizer updates", ylabel="Squared error")
        axis.set_yscale("log")
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def evaluate_and_save(args, output, device):
    results = []
    for dataset in args.datasets:
        for seed in args.seeds:
            folder = output / dataset / f"seed_{seed}"
            folder.mkdir(parents=True)
            train = make_data(dataset, args.train_size, seed * 100 + 1)
            validation = make_data(dataset, args.validation_size, seed * 100 + 2)
            test = make_data(dataset, args.eval_size, seed * 100 + 3)
            reference = make_data(dataset, args.eval_size, seed * 100 + 4)
            np.savez_compressed(folder / "data.npz", train=train, validation=validation,
                                test=test, real_reference=reference)
            mean, scale = train.mean(axis=0), np.clip(train.std(axis=0, ddof=1), 1e-3, 1e3)
            print(f"{dataset}, seed {seed}: training direct DDPM", flush=True)
            direct, direct_history, direct_time = fit_diffusion(
                (train - mean) / scale, (validation - mean) / scale,
                args, seed + 1000, device,
            )
            started = time.perf_counter()
            direct_samples = sample_diffusion(direct, args.eval_size, args, seed + 2000, device)
            direct_samples = direct_samples.cpu().numpy() * scale + mean
            direct_sampling_time = time.perf_counter() - started
            torch.save({"model": direct.state_dict(), "mean": torch.tensor(mean),
                        "scale": torch.tensor(scale)}, folder / "direct_ddpm.pt")
            np.save(folder / "direct_samples.npy", direct_samples)
            started = time.perf_counter()
            direct_w2, real_w2 = empirical_w2(direct_samples, test), empirical_w2(reference, test)
            direct_evaluation_time = time.perf_counter() - started
            for method, score in (("direct_ddpm", direct_w2), ("real_real", real_w2)):
                results.append({
                    "dataset": dataset, "seed": seed, "landmarks": "not_applicable",
                    "method": method, "w2": score,
                    "diffusion_train_seconds": direct_time if method == "direct_ddpm" else 0,
                    "sampling_seconds": direct_sampling_time if method == "direct_ddpm" else 0,
                })
            print(f"  direct W2={direct_w2:.4f}; real-real W2={real_w2:.4f}", flush=True)
            save_json(folder / "direct_history.json", direct_history)
            for rule in args.landmarks:
                target = folder / rule
                target.mkdir()
                base_index, indices = choose_references(train, rule, seed + 3000)
                encoder = Encoder(train, base_index, indices).to(device)
                with torch.no_grad():
                    train_z = encoder(torch.tensor(train, device=device)).cpu().numpy()
                    val_z = encoder(torch.tensor(validation, device=device)).cpu().numpy()
                matrix = encoder.directions.cpu().numpy()
                scaled_matrix = matrix / encoder.scale.cpu().numpy()[:, None]
                save_json(target / "references.json", {
                    "rule": rule, "base_index": base_index, "landmark_indices": indices.tolist(),
                    "base": train[base_index].tolist(), "landmarks": train[indices].tolist(),
                    "rank": int(np.linalg.matrix_rank(matrix)),
                    "condition_number_raw": float(np.linalg.cond(matrix)),
                    "condition_number_standardized": float(np.linalg.cond(scaled_matrix)),
                    "singular_values_raw": np.linalg.svd(matrix, compute_uv=False).tolist(),
                    "standardization_mean": encoder.mean.cpu().tolist(),
                    "standardization_scale": encoder.scale.cpu().tolist(),
                })
                scatter_plot(target / "references", [("Training data and references", train)],
                             f"{dataset}: {rule}", train[base_index], train[indices])
                print(f"  {rule}: training coordinate DDPM", flush=True)
                coordinate, coordinate_history, coordinate_time = fit_diffusion(
                    train_z, val_z, args, seed + 4000, device,
                )
                started = time.perf_counter()
                generated_z = sample_diffusion(coordinate, args.eval_size, args, seed + 5000, device)
                coordinate_sampling_time = time.perf_counter() - started
                np.save(target / "generated_coordinates.npy", generated_z.cpu().numpy())
                torch.save({"model": coordinate.state_dict(), "encoder": encoder.state_dict()},
                           target / "coordinate_ddpm.pt")
                generated = {}
                decoder_histories, tuning = {}, {}
                for kind in args.losses:
                    deltas = args.huber_deltas if kind == "huber" else [1.0]
                    best, decoder, history, selected_delta = float("inf"), None, None, None
                    total_decoder_time = 0
                    candidates = []
                    for delta in deltas:
                        candidate, candidate_history, score, elapsed = fit_decoder(
                            train, validation, encoder, kind, delta, args, seed + 6000, device,
                        )
                        total_decoder_time += elapsed
                        candidates.append({"delta": delta if kind == "huber" else None,
                                           "validation_embedded_mse": score,
                                           "history": candidate_history})
                        if score < best:
                            best, decoder, history, selected_delta = score, candidate, candidate_history, delta
                    tuning[kind] = candidates
                    decoder_histories[kind] = history
                    started = time.perf_counter()
                    with torch.no_grad():
                        samples = decoder(generated_z).cpu().numpy()
                        actual = torch.tensor(test, device=device)
                        reconstructed = decoder(encoder(actual))
                        object_error = reconstruction_loss(
                            reconstructed, actual, encoder, "object_mse"
                        ).item()
                        embedded_error = reconstruction_loss(
                            reconstructed, actual, encoder, "embedded_mse"
                        ).item()
                    decode_and_reconstruction_time = time.perf_counter() - started
                    started = time.perf_counter()
                    score = empirical_w2(samples, test)
                    evaluation_time = time.perf_counter() - started
                    generated[kind] = samples
                    np.savez_compressed(target / f"{kind}_samples.npz", generated=samples,
                                        reconstructed=reconstructed.cpu().numpy())
                    torch.save({
                        "decoder": decoder.state_dict(), "encoder": encoder.state_dict(),
                        "loss": kind, "huber_delta": selected_delta if kind == "huber" else None,
                    }, target / f"{kind}_decoder.pt")
                    results.append({
                        "dataset": dataset, "seed": seed, "landmarks": rule, "method": kind,
                        "w2": score, "delta_w2_vs_direct": score - direct_w2,
                        "object_reconstruction_mse": object_error,
                        "embedded_reconstruction_mse": embedded_error,
                        "huber_delta": selected_delta if kind == "huber" else None,
                        "diffusion_train_seconds": coordinate_time,
                        "decoder_train_seconds_including_tuning": total_decoder_time,
                        "sampling_seconds": coordinate_sampling_time,
                        "decode_and_reconstruction_seconds": decode_and_reconstruction_time,
                        "evaluation_seconds": evaluation_time,
                    })
                    print(f"    {kind}: W2={score:.4f}, reconstruction MSE={object_error:.6f}", flush=True)
                panels = [("Held-out real", test), (f"Direct DDPM: W2={direct_w2:.3f}", direct_samples)]
                for kind, points in generated.items():
                    score = next(row["w2"] for row in reversed(results) if row["method"] == kind)
                    panels.append((f"{LABELS[kind]}\nW2={score:.3f}", points))
                scatter_plot(target / "comparison_all_losses", panels,
                             f"PILOT: {dataset}, {rule}, seed {seed}; real-real W2={real_w2:.3f}")
                primary = "embedded_mse" if "embedded_mse" in generated else next(iter(generated))
                scatter_plot(target / "comparison_primary",
                             [("Held-out real", test), (LABELS[primary], generated[primary]),
                              ("Direct Euclidean DDPM", direct_samples)],
                             f"PILOT: {dataset}, {rule}, seed {seed}")
                curve_plot(target / "learning_curves.png",
                           {"direct": direct_history, "coordinate": coordinate_history}, decoder_histories)
                save_json(target / "histories.json", {"coordinate": coordinate_history, "decoder_tuning": tuning})
                # Preserve completed measurements even if a later combination fails.
                write_csv(output / "metrics.csv", results)
                save_json(output / "metrics.json", results)
            save_json(folder / "evaluation_timing.json", {
                "direct_and_real_real_w2_seconds": direct_evaluation_time,
            })
    return results


def summarize(rows, output):
    summaries = []
    groups = sorted({(r["dataset"], r["landmarks"], r["method"]) for r in rows})
    for dataset, landmarks, method in groups:
        group = [r for r in rows if (r["dataset"], r["landmarks"], r["method"]) == (dataset, landmarks, method)]
        scores = np.array([r["w2"] for r in group])
        summaries.append({
            "dataset": dataset, "landmarks": landmarks, "method": method,
            "runs": len(group), "w2_mean": float(scores.mean()),
            "w2_sample_sd": float(scores.std(ddof=1)) if len(group) > 1 else None,
        })
    write_csv(output / "summary.csv", summaries)
    for dataset in sorted({r["dataset"] for r in summaries}):
        entries = [r for r in summaries if r["dataset"] == dataset]
        figure, axis = plt.subplots(figsize=(10, 6))
        labels = [f'{r["landmarks"]}: {r["method"]}' for r in entries]
        scores = [r["w2_mean"] for r in entries]
        errors = [r["w2_sample_sd"] or 0 for r in entries]
        axis.barh(labels, scores, xerr=errors)
        axis.set(xlabel="Exact empirical W2 (lower is better)",
                 title=f"PILOT: {dataset}; error bars only with multiple training seeds")
        figure.tight_layout()
        figure.savefig(output / f"{dataset}_w2.png", dpi=180)
        plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--landmarks", nargs="+", choices=LANDMARKS, default=list(LANDMARKS))
    parser.add_argument("--losses", nargs="+", choices=LOSSES, default=list(LOSSES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--validation-size", type=int, default=1024)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--diffusion-steps", type=int, default=2500)
    parser.add_argument("--decoder-steps", type=int, default=1000)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--spatial-frequencies", type=int, default=0)
    parser.add_argument("--huber-deltas", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("train_size", "validation_size", "eval_size", "diffusion_steps",
                 "decoder_steps", "batch_size", "width", "threads"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.train_size < 4 or args.timesteps < 100:
        parser.error("At least 4 training points and 100 diffusion timesteps are required.")
    if not 0 <= args.spatial_frequencies <= 8:
        parser.error("--spatial-frequencies must lie between 0 and 8.")
    if any(seed < 0 or seed > 1_000_000 for seed in args.seeds):
        parser.error("Seeds must lie between 0 and 1000000.")
    if any(not math.isfinite(delta) or delta <= 0 for delta in args.huber_deltas):
        parser.error("Huber thresholds must be finite and positive.")
    for name in ("datasets", "landmarks", "losses", "seeds"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f"--{name} must not contain duplicates")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback.")
    device = torch.device(args.device)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path(__file__).resolve().parent / "results" / f"pilot_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "output": str(output.resolve())}
    config.update({
        "status": "pilot_not_publication_result",
        "diffusion_schedule": {"type": "linear_beta", "start": 0.0001, "end": 0.08},
        "optimizer": "Adam", "learning_rate": 0.001,
        "decoder_selection": "lowest validation raw embedded MSE; checkpoints every 100 updates",
        "huber_selection": "lowest validation raw embedded MSE among predeclared thresholds",
        "manual_landmarks": "training extrema in directions 0, 120, 240 degrees; fixed before results",
        "base": "training observation nearest coordinate-wise training median",
        "moons": {"angle": "uniform [0, pi]", "mixture_weights": [0.5, 0.5],
                  "arcs": ["(cos(t), sin(t))", "(1-cos(t), 0.5-sin(t))"],
                  "isotropic_noise_std": 0.05},
        "gaussians": {"components": 8, "center_radius": 2.0, "isotropic_std": 0.15,
                      "mixture_weights": [0.125] * 8},
        "evaluation": "exact equal-weight assignment W2; original 2D units",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__,
                        "device": str(device), "cpu_threads": args.threads},
    })
    save_json(output / "config.json", config)
    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True, check=True,
    )
    (output / "environment.txt").write_text(packages.stdout, encoding="utf-8")
    started = time.perf_counter()
    rows = evaluate_and_save(args, output, device)
    summarize(rows, output)
    save_json(output / "completion.json", {
        "status": "completed", "wall_seconds": time.perf_counter() - started,
        "metric_rows": len(rows), "independent_training_seeds": len(args.seeds),
        "interpretation": "A single seed is a pilot, not evidence of superiority or equivalence.",
    })
    print(f"Completed. Results: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
