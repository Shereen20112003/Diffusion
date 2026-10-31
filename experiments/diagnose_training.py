"""Validation-only training-duration study; never loads pilot test observations."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from run_experiments import (
    DATASETS, Encoder, choose_references, empirical_w2, fit_decoder,
    fit_diffusion, forward_noise, make_data, sample_diffusion, save_json,
    scatter_plot, schedule, write_csv,
)
import matplotlib.pyplot as plt


def geometry_metrics(dataset, points):
    """Diagnostics supplement W2; neither is used to choose a checkpoint."""
    if dataset == "two_moons":
        angles = np.linspace(0, np.pi, 2048)
        arcs = np.concatenate([
            np.column_stack([np.cos(angles), np.sin(angles)]),
            np.column_stack([1 - np.cos(angles), 0.5 - np.sin(angles)]),
        ])
        from scipy.spatial import cKDTree
        distances, _ = cKDTree(arcs).query(points)
        return {"mean_distance_to_clean_arcs": float(distances.mean()),
                "fraction_farther_than_0.15_from_arcs": float((distances > 0.15).mean())}
    if dataset == "eight_gaussians":
        angles = np.arange(8) * 2 * np.pi / 8
        centers = 2 * np.column_stack([np.cos(angles), np.sin(angles)])
        squared = ((points[:, None] - centers) ** 2).sum(axis=2)
        labels = squared.argmin(axis=1)
        distances = np.sqrt(squared.min(axis=1))
        return {"mean_distance_to_nearest_center": float(distances.mean()),
                "fraction_farther_than_0.45_from_centers": float((distances > 0.45).mean()),
                "nearest_center_frequencies": (np.bincount(labels, minlength=8) / len(points)).tolist()}
    raise ValueError(f"Unknown dataset: {dataset}")


@torch.no_grad()
def noise_profile(model, validation, args, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    clean = torch.tensor(validation, device=device)
    _, _, alpha_bar, _ = schedule(args.timesteps, device)
    noise = torch.randn(clean.shape, generator=generator, device=device)
    profile = []
    for t in range(args.timesteps):
        steps = torch.full((len(clean),), t, device=device, dtype=torch.long)
        noisy = forward_noise(clean, noise, steps, alpha_bar)
        residual = model(noisy, steps) - noise
        profile.append({
            "timestep": t + 1, "alpha_bar": alpha_bar[t].item(),
            "noise_mse_per_coordinate": residual.square().mean().item(),
        })
    return profile


def render_dataset(folder, dataset, checkpoints, seed):
    import csv

    with np.load(folder / "development_data.npz") as data:
        validation = data["distribution_validation"]
    for step in checkpoints:
        scatter_plot(
            folder / f"comparison_updates_{step}",
            [("Development validation data", validation),
             ("Coordinate DDPM + MLP", np.load(folder / "coordinate" / f"updates_{step}" / "generated.npy")),
             ("Direct DDPM", np.load(folder / "direct" / f"updates_{step}" / "generated.npy"))],
            f"DEVELOPMENT ONLY: {dataset}, {step} updates, seed {seed}",
        )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, method in zip(axes, ("direct", "coordinate")):
        for step in checkpoints:
            with (folder / method / f"updates_{step}" / "noise_profile.csv").open(newline="") as file:
                profile = list(csv.DictReader(file))
            axis.plot([int(r["timestep"]) for r in profile],
                      [float(r["noise_mse_per_coordinate"]) for r in profile], label=f"{step} updates")
        axis.set(title=method, xlabel="Diffusion timestep", ylabel="Noise MSE per coordinate")
        axis.legend()
    figure.tight_layout()
    figure.savefig(folder / "noise_profiles.png", dpi=180)
    plt.close(figure)


def render_saved_run(output):
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    for dataset in config["datasets"]:
        render_dataset(output / dataset, dataset, config["checkpoints"], config["seed"])
    save_json(output / "rendering.json", {
        "status": "rendered_from_saved_samples",
        "note": "Rendering does not certify that the original training process completed normally.",
    })


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    args.output.mkdir(parents=True, exist_ok=False)
    settings = SimpleNamespace(
        width=128, diffusion_steps=max(args.checkpoints), decoder_steps=1000,
        timesteps=200, batch_size=128, spatial_frequencies=args.spatial_frequencies,
    )
    source_files = [Path(__file__), Path(__file__).with_name("run_experiments.py")]
    save_json(args.output / "config.json", {
        "stage": "validation_only_development_not_final_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": args.datasets, "seed": args.seed, "checkpoints": args.checkpoints,
        "model_settings": vars(settings), "threads": args.threads,
        "train_size": 4096, "decoder_validation_size": 1024,
        "distribution_validation_size": 512,
        "training_seed_rule": "seed*100+1; same training observations as pilot",
        "fit_validation_seed_rule": "seed*100+2; same fitting-validation observations as pilot",
        "distribution_validation_seeds": [90000 + args.seed, 91000 + args.seed],
        "landmarks": "manual peripheral directions",
        "loss": "embedded_mse",
        "selection": "best noise-validation checkpoint up to each stated update budget",
        "controlled_factors": "architecture, schedule, initial weights, training stream, sampling seed",
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "torch": torch.__version__, "numpy": np.__version__,
        "reference": "https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils.py",
    })
    metrics, started = [], time.perf_counter()
    for dataset in args.datasets:
        folder = args.output / dataset
        folder.mkdir()
        train = make_data(dataset, 4096, args.seed * 100 + 1)
        validation = make_data(dataset, 1024, args.seed * 100 + 2)
        distribution_validation = make_data(dataset, 512, 90000 + args.seed)
        reference = make_data(dataset, 512, 91000 + args.seed)
        np.savez_compressed(folder / "development_data.npz", train=train,
                            fitting_validation=validation,
                            distribution_validation=distribution_validation,
                            real_reference=reference)
        real_w2 = empirical_w2(reference, distribution_validation)
        save_json(folder / "real_reference.json", {
            "w2": real_w2, "validation_geometry": geometry_metrics(dataset, distribution_validation),
            "independent_real_geometry": geometry_metrics(dataset, reference),
        })
        base_index, indices = choose_references(train, "manual", args.seed + 3000)
        encoder = Encoder(train, base_index, indices).to(device)
        decoder, decoder_history, decoder_score, decoder_time = fit_decoder(
            train, validation, encoder, "embedded_mse", 1, settings, args.seed + 6000, device,
        )
        torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
                   folder / "reconstruction.pt")
        save_json(folder / "reconstruction_history.json", {
            "history": decoder_history, "best_embedded_mse": decoder_score,
            "train_seconds": decoder_time,
        })
        mean = train.mean(axis=0)
        scale = np.clip(train.std(axis=0, ddof=1), 1e-3, 1e3)
        with torch.no_grad():
            encoded_train = encoder(torch.tensor(train)).numpy()
            encoded_validation = encoder(torch.tensor(validation)).numpy()
        histories = {}
        for method in ("direct", "coordinate"):
            if method == "direct":
                train_input, validation_input = (train - mean) / scale, (validation - mean) / scale
                training_seed, sampling_seed = args.seed + 1000, args.seed + 2000
            else:
                train_input, validation_input = encoded_train, encoded_validation
                training_seed, sampling_seed = args.seed + 4000, args.seed + 5000
            def checkpoint(step, model, validation_score):
                checkpoint_folder = folder / method / f"updates_{step}"
                checkpoint_folder.mkdir(parents=True)
                t0 = time.perf_counter()
                values = sample_diffusion(model, 512, settings, sampling_seed, device)
                with torch.no_grad():
                    if method == "direct":
                        generated = values.numpy() * scale + mean
                    else:
                        generated = decoder(values).numpy()
                score = empirical_w2(generated, distribution_validation)
                np.save(checkpoint_folder / "generated.npy", generated)
                torch.save({"model": model.state_dict(), "mean": torch.tensor(mean),
                            "scale": torch.tensor(scale),
                            "spatial_frequencies": settings.spatial_frequencies},
                           checkpoint_folder / "denoiser.pt")
                profile = noise_profile(model, validation_input, settings, 92000 + args.seed, device)
                write_csv(checkpoint_folder / "noise_profile.csv", profile)
                geometry = geometry_metrics(dataset, generated)
                save_json(checkpoint_folder / "geometry.json", geometry)
                metrics.append({
                    "dataset": dataset, "method": method, "updates": step,
                    "validation_w2": score, "real_real_w2": real_w2,
                    "checkpoint_noise_validation_sum_mse": validation_score,
                    "low_noise_mse_per_coordinate": float(np.mean(
                        [row["noise_mse_per_coordinate"] for row in profile[:20]])),
                    "high_noise_mse_per_coordinate": float(np.mean(
                        [row["noise_mse_per_coordinate"] for row in profile[-20:]])),
                    "validation_generation_seconds": time.perf_counter() - t0,
                    "denoiser_parameters": sum(p.numel() for p in model.parameters()),
                })
                write_csv(args.output / "validation_metrics.csv", metrics)
                print(f"{dataset} {method} {step} updates: validation W2={score:.4f}", flush=True)

            print(f"Training {dataset} {method} to {max(args.checkpoints)} updates", flush=True)
            _, history, elapsed = fit_diffusion(
                train_input, validation_input, settings, training_seed, device,
                checkpoints=args.checkpoints, on_checkpoint=checkpoint,
            )
            histories[method] = {"history": history, "seconds_including_diagnostics": elapsed}
            save_json(folder / "training_histories.json", histories)
        render_dataset(folder, dataset, args.checkpoints, args.seed)
    save_json(args.output / "completion.json", {
        "status": "completed", "seconds": time.perf_counter() - started,
        "rows": len(metrics), "stage": "validation_only_development",
    })
    print(f"Development results: {args.output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[2500, 10000])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spatial-frequencies", type=int, default=0)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.seed < 0 or args.seed > 1_000_000 or args.threads <= 0:
        parser.error("Use a seed from 0 to 1000000 and a positive thread count.")
    if (any(step <= 0 for step in args.checkpoints)
            or len(args.checkpoints) != len(set(args.checkpoints))):
        parser.error("Checkpoints must be positive and distinct.")
    if len(args.datasets) != len(set(args.datasets)):
        parser.error("Datasets must be distinct.")
    if not 0 <= args.spatial_frequencies <= 8:
        parser.error("--spatial-frequencies must lie between 0 and 8.")
    args.checkpoints.sort()
    if args.render_only:
        render_saved_run(args.output)
    else:
        run(args)


if __name__ == "__main__":
    main()
