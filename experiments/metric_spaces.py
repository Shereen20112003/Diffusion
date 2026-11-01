"""Explicit synthetic object spaces and geometry-compatible reconstruction."""

import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import ndtri
import torch
from torch import nn

from run_experiments import make_data


METRIC_DATASETS = ("sphere", "tree", "distributions")
ALL_DATASETS = ("two_moons", "eight_gaussians", *METRIC_DATASETS)


def angular_squared(dot):
    """Squared angle with a finite derivative at coincident unit vectors."""
    dot = dot.clamp(-1, 1)
    epsilon = torch.finfo(dot.dtype).eps
    regular = torch.acos(dot.clamp(-1 + epsilon, 1 - epsilon)).square()
    gap = 1 - dot
    return torch.where(dot > 0.9999, 2 * gap + gap.square() / 3, regular)


class ObjectSpace:
    def __init__(self, dataset, quantiles=32, sphere_iterations=24):
        if dataset not in ALL_DATASETS:
            raise ValueError(f"Unknown object dataset: {dataset}")
        if quantiles < 4 or sphere_iterations < 1:
            raise ValueError("At least four quantiles and one sphere iteration are required.")
        self.dataset = dataset
        self.quantiles = quantiles
        self.sphere_iterations = sphere_iterations

    def sample(self, size, seed):
        if size < 1:
            raise ValueError("Sample size must be positive.")
        if self.dataset not in METRIC_DATASETS:
            return make_data(self.dataset, size, seed)
        rng = np.random.default_rng(seed)
        if self.dataset == "sphere":
            component = rng.choice(3, size, p=[0.2, 0.3, 0.5])
            angle = 2 * np.pi * component / 3
            centers = 0.45 * np.column_stack([np.cos(angle), np.sin(angle)])
            noise_angle = rng.uniform(0, 2 * np.pi, size)
            noise_radius = 0.08 * np.sqrt(rng.uniform(size=size))
            tangent = centers + noise_radius[:, None] * np.column_stack(
                [np.cos(noise_angle), np.sin(noise_angle)]
            )
            radius = np.linalg.norm(tangent, axis=1)
            points = np.column_stack([
                tangent * np.sinc(radius[:, None] / np.pi), np.cos(radius)
            ])
        elif self.dataset == "tree":
            # A compact three-edge metric star; label 3 denotes the vertex atom.
            label = rng.choice(4, size, p=[0.2, 0.3, 0.4, 0.1])
            radius = 0.2 + 0.8 * rng.beta(2, 2, size)
            points = np.zeros((size, 3))
            nonvertex = label < 3
            points[np.flatnonzero(nonvertex), label[nonvertex]] = radius[nonvertex]
        else:
            component = rng.choice(3, size, p=[0.2, 0.3, 0.5])
            location = np.asarray([-1.0, 0.0, 1.0])[component] + rng.uniform(-0.15, 0.15, size)
            scale = np.asarray([0.25, 0.4, 0.55])[component] + rng.uniform(-0.03, 0.03, size)
            skew = rng.uniform(0, 0.65, size)
            u = (np.arange(self.quantiles) + 0.5) / self.quantiles
            shape = ((1 - skew[:, None]) * ndtri(u)
                     + skew[:, None] * (-np.log1p(-u) - 1))
            points = location[:, None] + scale[:, None] * shape
        return points.astype(np.float32)

    def pairwise_squared(self, x, y):
        if self.dataset == "sphere":
            return angular_squared(
                torch.nn.functional.normalize(x, dim=1) @ torch.nn.functional.normalize(y, dim=1).T
            )
        if self.dataset == "tree":
            return torch.cdist(x, y, p=1).square()
        cost = torch.cdist(x, y).square()
        return cost / x.shape[1] if self.dataset == "distributions" else cost

    def squared(self, x, y):
        if self.dataset == "sphere":
            return angular_squared(
                (torch.nn.functional.normalize(x, dim=-1)
                 * torch.nn.functional.normalize(y, dim=-1)).sum(dim=-1)
            )
        if self.dataset == "tree":
            return (x - y).abs().sum(dim=-1).square()
        cost = (x - y).square().sum(dim=-1)
        return cost / x.shape[-1] if self.dataset == "distributions" else cost

    def validate(self, points):
        values = np.asarray(points)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Objects must form a nonempty finite matrix.")
        expected = (self.quantiles if self.dataset == "distributions"
                    else 3 if self.dataset in ("sphere", "tree") else 2)
        if values.shape[1] != expected:
            raise ValueError(f"{self.dataset} expects {expected} representation coordinates.")
        if self.dataset == "sphere" and not np.allclose(
            np.linalg.norm(values, axis=1), 1, atol=2e-5
        ):
            raise ValueError("Sphere objects must be unit vectors.")
        if self.dataset == "tree":
            if (np.any(values < -1e-7) or np.any(values > 1 + 1e-7)
                    or np.any((values > 1e-7).sum(axis=1) > 1)):
                raise ValueError("A tree object must lie on one of the three unit edges.")
        if self.dataset == "distributions" and np.any(np.diff(values, axis=1) < -1e-6):
            raise ValueError("Distribution quantiles must be nondecreasing.")

    def repair_ambient(self, values):
        """Explicit baseline postprocessing, never applied to coordinate samples."""
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite ambient samples.")
        if self.dataset == "sphere":
            length = values.norm(dim=1, keepdim=True)
            if torch.any(length < 1e-10):
                raise ValueError("A zero vector has no radial sphere projection.")
            return values / length
        if self.dataset == "tree":
            radius, branch = values.max(dim=1)
            return torch.nn.functional.one_hot(branch, 3) * radius.clamp(0, 1)[:, None]
        if self.dataset == "distributions":
            return values.sort(dim=1).values
        return values

    def frechet(self, weights, references):
        if self.dataset == "tree":
            moments = weights @ references
            return (2 * moments - moments.sum(dim=1, keepdim=True)).clamp_min(0)
        if self.dataset != "sphere":
            return weights @ references
        mean = torch.nn.functional.normalize(weights @ references, dim=1)
        for _ in range(self.sphere_iterations):
            tangent = self.sphere_log_average(mean, weights, references)
            length = tangent.norm(dim=1, keepdim=True)
            mean = (length.cos() * mean + torch.sinc(length / math.pi) * tangent)
            mean = torch.nn.functional.normalize(mean, dim=1)
        return mean

    @staticmethod
    def sphere_log_average(mean, weights, references):
        dot = (mean @ references.T).clamp(-1, 1)
        squared = angular_squared(dot)
        angle = squared.clamp_min(1e-12).sqrt()
        factor = torch.where(squared < 1e-6, 1 + squared / 6,
                             angle / angle.sin().clamp_min(1e-8))
        scaled = weights * factor
        return scaled @ references - (scaled * dot).sum(dim=1, keepdim=True) * mean

    def description(self):
        return {
            "sphere": {
                "metric": "unit-sphere geodesic angle (radians)",
                "data": "three tangent-disk components at radius .45, disk radius .08; north exponential map",
                "mixture_weights": [0.2, 0.3, 0.5],
                "support": "north-pole geodesic cap of radius <= .53 < pi/4",
                "decoder": "unrolled Karcher iteration from normalized weighted Euclidean mean",
                "solver_caveat": "gradient residual reported; not a certified global objective-gap estimate",
                "baseline": "ambient 3D DDPM with radial normalization",
            },
            "tree": {
                "metric": "shortest path on fixed compact three-edge unit metric star",
                "data": "branch masses .2/.3/.4 and vertex atom .1; radius .2+.8*Beta(2,2)",
                "decoder": "exact weighted Frechet mean using positive folded branch moment",
                "baseline": "ambient one-hot/radius DDPM with nearest Euclidean star projection",
            },
            "distributions": {
                "metric": "exact 1D W2 between equally weighted sorted-atom measures",
                "data": "bounded location/scale/skew family of normal/exponential quantile mixtures",
                "objects": f"{self.quantiles} equal-mass atoms per distribution, not exact continuous laws",
                "decoder": "exact Frechet barycenter: weighted average of sorted atoms",
                "baseline": "ambient quantile DDPM with increasing rearrangement",
            },
        }.get(self.dataset, {
            "metric": "Euclidean",
            "data": "unchanged original planar generator",
            "support_caveat": "unbounded noise; does not satisfy compact-support theorem",
            "decoder": "MLP comparison and exact weighted Euclidean Frechet means",
            "baseline": "direct Euclidean DDPM, no projection",
        })


def fps_indices(points, space, count, start=0):
    if not 1 <= count <= len(points):
        raise ValueError("Reference count must lie within the training sample size.")
    selected, nearest = [int(start)], torch.full(
        (len(points),), float("inf"), dtype=points.dtype, device=points.device
    )
    while len(selected) < count:
        nearest = torch.minimum(
            nearest, space.pairwise_squared(points, points[selected[-1]:selected[-1] + 1])[:, 0]
        )
        nearest[selected] = -1
        index = int(nearest.argmax())
        if nearest[index] <= 1e-12:
            raise ValueError("Not enough distinct training objects for reference selection.")
        selected.append(index)
    return np.asarray(selected)


class MetricEncoder(nn.Module):
    def __init__(self, points, space, count):
        super().__init__()
        self.space = space
        tensor = torch.as_tensor(points)
        indices = fps_indices(tensor, space, count + 1)
        self.indices = indices.tolist()
        self.register_buffer("base", tensor[indices[:1]].clone())
        self.register_buffer("landmarks", tensor[indices[1:]].clone())
        self.register_buffer(
            "lengths", space.pairwise_squared(self.base, self.landmarks)[0].sqrt()
        )
        raw = self.raw(tensor)
        self.register_buffer("mean", raw.mean(dim=0))
        self.register_buffer("scale", raw.std(dim=0).clamp(1e-3, 1e3))

    def raw(self, points):
        base_distance = self.space.pairwise_squared(points, self.base)
        landmark_distance = self.space.pairwise_squared(points, self.landmarks)
        return ((base_distance + self.lengths.square() - landmark_distance)
                / (2 * self.lengths))

    def forward(self, points):
        return (self.raw(points) - self.mean) / self.scale


class FrechetDecoder(nn.Module):
    """E2M-style learned weights; a bounded residual to a localized kernel."""

    def __init__(self, encoder, references, space, bandwidth, width, learned):
        super().__init__()
        if not math.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("Bandwidth must be positive and finite.")
        space.validate(references.detach().cpu().numpy())
        if space.dataset == "sphere" and torch.any(references[:, 2] <= math.cos(math.pi / 4)):
            raise ValueError("This spherical solver study requires references in the specified north cap.")
        self.space, self.bandwidth, self.learned = space, bandwidth, learned
        self.register_buffer("references", references.detach().clone())
        with torch.no_grad():
            self.register_buffer("coordinates", encoder(references).detach().clone())
        self.network = nn.Sequential(
            nn.Linear(len(encoder.mean), width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, len(references)),
        ) if learned else None
        if learned:
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def weights(self, z):
        squared = (z[:, None] - self.coordinates[None]).square().sum(dim=2)
        logits = -squared / (2 * self.bandwidth ** 2)
        if self.learned:
            logits = logits + 2 * torch.tanh(self.network(z))
        return logits.softmax(dim=1)

    def forward(self, z):
        return self.space.frechet(self.weights(z), self.references)

    @torch.no_grad()
    def diagnostics(self, z):
        weights = self.weights(z)
        distances = torch.cdist(z, self.coordinates)
        nearest = distances.min(dim=1).values
        localization = (weights * distances).sum(dim=1) - nearest
        result = {
            "generated_localization_mean": float(localization.mean()),
            "generated_nearest_reference_mean": float(nearest.mean()),
            "kernel_localization_bound": self.bandwidth * math.sqrt(
                2 * (math.log(len(self.references)) + (4 if self.learned else 0))
            ),
            "weight_sum_max_error": float((weights.sum(dim=1) - 1).abs().max()),
        }
        if self.space.dataset == "sphere":
            predicted = self(z)
            residual = self.space.sphere_log_average(predicted, weights, self.references).norm(dim=1)
            result["sphere_mean_log_residual_max"] = float(residual.max())
            result["sphere_mean_log_residual_mean"] = float(residual.mean())
            if not torch.isfinite(residual).all() or residual.max() > 1e-4:
                raise RuntimeError("Spherical Frechet iteration failed the 1e-4 log-residual tolerance.")
        else:
            result["global_objective_gap"] = 0.0
        return result


def metric_loss(predicted, target, encoder, kind, delta=1.0):
    if kind == "object_mse":
        return encoder.space.squared(predicted, target).mean()
    residual = encoder.raw(predicted) - encoder.raw(target)
    if kind == "embedded_mse":
        penalty = residual.square()
    elif kind == "standardized_mse":
        penalty = (residual / encoder.scale).square()
    elif kind == "huber":
        penalty = torch.where(residual.abs() <= delta, 0.5 * residual.square(),
                              delta * (residual.abs() - 0.5 * delta))
    else:
        raise ValueError(f"Unknown reconstruction loss: {kind}")
    return penalty.sum(dim=1).mean()


def object_w2(generated, real, space):
    space.validate(generated)
    space.validate(real)
    if generated.shape != real.shape:
        raise ValueError("Exact assignment requires equally sized sample matrices.")
    with torch.no_grad():
        cost = space.pairwise_squared(
            torch.as_tensor(generated, dtype=torch.float64),
            torch.as_tensor(real, dtype=torch.float64),
        ).numpy()
    if not np.isfinite(cost).all():
        raise FloatingPointError("Nonfinite transport cost.")
    rows, columns = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, columns].mean()))


def object_diagnostics(points, space):
    space.validate(points)
    if space.dataset not in METRIC_DATASETS:
        from diagnose_training import geometry_metrics
        return geometry_metrics(space.dataset, points)
    if space.dataset == "tree":
        radius = points.sum(axis=1)
        vertex = radius <= 1e-7
        labels = np.where(vertex, 3, points.argmax(axis=1))
        frequencies = np.bincount(labels, minlength=4) / len(points)
        return {
            "vertex_mass": float(vertex.mean()),
            "branch_0_mass": float(frequencies[0]),
            "branch_1_mass": float(frequencies[1]),
            "branch_2_mass": float(frequencies[2]),
            "branch_vertex_total_variation": float(
                np.abs(frequencies - [0.2, 0.3, 0.4, 0.1]).sum() / 2
            ),
        }
    if space.dataset == "sphere":
        return {
            "unit_norm_max_error": float(np.abs(np.linalg.norm(points, axis=1) - 1).max()),
            "fraction_outside_data_cap": float((points[:, 2] < np.cos(0.53) - 1e-6).mean()),
        }
    return {
        "object_mean_average": float(points.mean(axis=1).mean()),
        "object_std_average": float(points.std(axis=1).mean()),
        "minimum_quantile_increment": float(np.diff(points, axis=1).min()),
    }
