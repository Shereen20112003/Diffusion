"""Conditional unit-tripod laws, reference geometry, and intrinsic diagnostics."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from scipy.special import betainc, softmax
import torch
from torch import nn

from metric_spaces import ObjectSpace


SCENARIOS = ("S0_no_atom", "S1_varying_atom", "S2_near_vertex_no_atom")
THRESHOLDS = (0., .0025, .005, .01, .02, .05)
H_RATIOS = (.05, .10, .20, .40)
RADIUS_GRID = np.arange(1, 10) / 10
SPACE = ObjectSpace("tree")
DIRECTIONS = np.column_stack((
    np.cos(2 * np.pi * np.arange(3) / 3),
    np.sin(2 * np.pi * np.arange(3) / 3),
))


def stream_seed(master, scenario, role, draw=0):
    if scenario not in SCENARIOS or min(master, role, draw) < 0:
        raise ValueError("Invalid scenario or random-stream identifier.")
    return int(np.random.SeedSequence(
        [20261102, master, SCENARIOS.index(scenario), role, draw]
    ).generate_state(1)[0] & 0x7fffffff)


def conditions(x):
    values = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Conditions must be a finite vector in [0,1].")
    return values


def law_probabilities(x, scenario):
    x = conditions(x)
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown conditional-star scenario: {scenario}")
    branches = softmax(1.5 * np.cos(
        2 * np.pi * (x[:, None] - np.arange(3)[None] / 3)
    ), axis=1)
    p0 = .05 + .30 * np.sin(np.pi * x) ** 2 if scenario == SCENARIOS[1] else np.zeros(len(x))
    return p0, branches


def radial_cdf(t, x, scenario):
    x = conditions(x)
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown conditional-star scenario: {scenario}")
    t = np.clip(np.asarray(t, dtype=np.float64), 0, 1)
    a, b = ((np.full_like(x, .5), np.full_like(x, 2.)) if scenario == SCENARIOS[2]
            else (2 + 2 * x, 4 - 2 * x))
    return .25 * t + .75 * betainc(a, b, t)


def sample_pairs(size, scenario, seed, x=None):
    if size < 1:
        raise ValueError("Sample size must be positive.")
    rng = np.random.default_rng(seed)
    covariate = rng.uniform(size=size) if x is None else conditions(x)
    if len(covariate) == 1 and size != 1:
        covariate = np.repeat(covariate, size)
    if len(covariate) != size:
        raise ValueError("One condition per observation is required.")
    p0, branch_prob = law_probabilities(covariate, scenario)
    branch = (rng.uniform(size=(size, 1)) > np.cumsum(branch_prob, axis=1)).sum(axis=1)
    if scenario == SCENARIOS[2]:
        radius = rng.beta(.5, 2, size)
    else:
        radius = rng.beta(2 + 2 * covariate, 4 - 2 * covariate)
    uniform = rng.uniform(size=size)
    radius = np.where(rng.uniform(size=size) < .25, uniform, radius)
    # Continuous laws have no vertex/end-point atoms, even at floating-point edges.
    radius = np.clip(radius, np.nextafter(0., 1.), np.nextafter(1., 0.))
    radius[rng.uniform(size=size) < p0] = 0
    points = np.eye(3)[branch] * radius[:, None]
    return covariate, points


def tensor(values):
    return torch.as_tensor(values, dtype=torch.float64)


def distances(x, y):
    return np.abs(np.asarray(x)[:, None] - np.asarray(y)[None]).sum(axis=2)


def validate_points(points):
    values = np.asarray(points)
    if (values.ndim != 2 or values.shape[1] != 3 or not len(values)
            or not np.isfinite(values).all() or np.any(values < 0)
            or np.any(values > 1) or np.any((values > 0).sum(axis=1) > 1)):
        raise ValueError("Tripod points must have at most one positive radius in [0,1].")


def jr_arrays(points):
    values = np.asarray(points)
    validate_points(values.reshape(-1, 3))
    radius = values.sum(axis=-1)
    branch = np.where(radius == 0, 0, values.argmax(axis=-1) + 1)
    return branch, radius


def weighted_mean(weights, references):
    return SPACE.frechet(weights, references)


class StarEncoder(nn.Module):
    def __init__(self, references, k=3, base_kind="medoid", landmarks=None):
        super().__init__()
        validate_points(references)
        if base_kind not in ("medoid", "frechet"):
            raise ValueError(f"Unknown base rule: {base_kind}")
        references = np.asarray(references, dtype=np.float64)
        medoid_index = int(np.argmin(np.mean(distances(references, references) ** 2, axis=1)))
        medoid = references[medoid_index]
        frechet_base = weighted_mean(
            torch.full((1, len(references)), 1 / len(references), dtype=torch.float64),
            tensor(references))[0].numpy()
        base = medoid if base_kind == "medoid" else frechet_base
        chosen = []
        forced = []
        if landmarks is not None:
            for index in landmarks:
                location = references[index]
                if np.sum(np.abs(location - base)) > 0 and not any(
                    np.array_equal(location, references[previous]) for previous in chosen
                ):
                    chosen.append(int(index))
                else:
                    forced.append(int(index))
        nearest = distances(references, base[None])[:, 0]
        for index in chosen:
            nearest = np.minimum(nearest, distances(references, references[index:index + 1])[:, 0])
        while len(chosen) < k:
            index = int(np.argmax(nearest))
            if nearest[index] <= 0:
                raise ValueError(f"Reference sample has fewer than {k} admissible distinct landmarks.")
            chosen.append(index)
            nearest = np.minimum(nearest, distances(references, references[index:index + 1])[:, 0])
        self.indices = chosen[:k]
        self.metadata = {
            "base_kind": base_kind, "base": base.tolist(), "medoid_index": medoid_index,
            "medoid": medoid.tolist(), "base_equals_medoid": bool(np.array_equal(base, medoid)),
            "empirical_frechet_mean": frechet_base.tolist(),
            "medoid_equals_frechet_mean": bool(np.array_equal(frechet_base, medoid)),
            "forced_landmark_changes": forced, "landmark_indices": self.indices,
            "landmarks": references[self.indices].tolist(), "reference_count": len(references),
            "center_multiplicity": int(np.sum(references.sum(axis=1) == 0)),
        }
        self.register_buffer("base", tensor(base[None]))
        self.register_buffer("landmarks", tensor(references[self.indices]))
        self.register_buffer("lengths", SPACE.pairwise_squared(self.base, self.landmarks)[0].sqrt())
        raw = self.raw(tensor(references))
        self.register_buffer("mean", raw.mean(dim=0))
        self.register_buffer("scale", raw.std(dim=0, correction=1).clamp(.25, 4))

    @classmethod
    def controlled(cls):
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.indices = [0, 1, 2]
        instance.metadata = {"base_kind": "controlled_center_endpoints", "diagnostic_only": True}
        instance.register_buffer("base", torch.zeros((1, 3), dtype=torch.float64))
        instance.register_buffer("landmarks", torch.eye(3, dtype=torch.float64))
        instance.register_buffer("lengths", torch.ones(3, dtype=torch.float64))
        instance.register_buffer("mean", torch.zeros(3, dtype=torch.float64))
        instance.register_buffer("scale", torch.full((3,), np.sqrt(3), dtype=torch.float64))
        return instance

    def raw(self, points):
        return (SPACE.pairwise_squared(points, self.base)
                + self.lengths.square()
                - SPACE.pairwise_squared(points, self.landmarks)) / (2 * self.lengths)

    def forward(self, points):
        return (self.raw(points) - self.mean) / self.scale

    def describe(self):
        support = torch.cat((torch.zeros((1, 3), dtype=torch.float64),
                             torch.eye(3, dtype=torch.float64)))
        with torch.no_grad():
            radius = float(self(support).norm(dim=1).max())
        return {
            **self.metadata, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "scale_clipping": [.25, 4.], "minimum_base_landmark_distance": float(self.lengths.min()),
            "certified_coordinate_radius": radius,
            "radius_basis": "maximum encoded norm at center and three endpoints; affine on each arm",
        }


class GaussianFrechet(nn.Module):
    def __init__(self, encoder, references, h_ratio, width=64, learned=False):
        super().__init__()
        if h_ratio <= 0:
            raise ValueError("Bandwidth ratio must be positive.")
        self.encoder = encoder
        self.register_buffer("references", tensor(references))
        with torch.no_grad():
            encoded = encoder(self.references)
        pairwise = pdist(encoded.numpy())
        positive = pairwise[pairwise > 0]
        if not len(positive):
            raise ValueError("Gaussian decoder requires distinct encoded references.")
        self.h_scale = float(np.median(positive))
        self.h_ratio = h_ratio
        self.h = h_ratio * self.h_scale
        self.register_buffer("encoded", encoded)
        self.residual = (nn.Sequential(nn.Linear(encoded.shape[1], width), nn.SiLU(),
                                       nn.Linear(width, len(references))).double()
                         if learned else None)
        if self.residual is not None:
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def weights(self, z):
        logits = -torch.cdist(z.double(), self.encoded).square() / (2 * self.h ** 2)
        if self.residual is not None:
            logits = logits + 2 * self.residual(z.double()).tanh()
        return logits.softmax(dim=1)

    def forward(self, z):
        return weighted_mean(self.weights(z), self.references)

    def margins(self, z):
        moments = self.weights(z) @ self.references
        return (2 * moments - moments.sum(dim=1, keepdim=True)).max(dim=1).values


@torch.no_grad()
def decoder_diagnostics(decoder, coordinates, chunk=512):
    z = tensor(coordinates.reshape(-1, coordinates.shape[-1]))
    margins, localization, nearest_all = [], [], []
    for start in range(0, len(z), chunk):
        batch = z[start:start + chunk]
        weights = decoder.weights(batch)
        moments = weights @ decoder.references
        margins.append((2 * moments - moments.sum(dim=1, keepdim=True)).max(dim=1).values.numpy())
        distance = torch.cdist(batch, decoder.encoded)
        nearest = distance.min(dim=1).values
        localization.append(((weights * distance).sum(dim=1) - nearest).numpy())
        nearest_all.append(nearest.numpy())
    margin = np.concatenate(margins)
    return {
        "minimum_absolute_sign_margin": float(np.abs(margin).min()),
        "positive_margin_le_1e_12_count": int(((margin > 0) & (margin <= 1e-12)).sum()),
        "exact_zero_margin_count": int((margin == 0).sum()),
        "mean_localization_excess": float(np.concatenate(localization).mean()),
        "mean_nearest_encoded_reference_distance": float(np.concatenate(nearest_all).mean()),
        "gaussian_localization_bound": float(decoder.h * np.sqrt(
            2 * (np.log(len(decoder.references)) + (4 if decoder.residual is not None else 0)))),
    }


def to_planar(points):
    return np.asarray(points) @ DIRECTIONS


def project_planar(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Ambient samples must be a finite N by 2 matrix.")
    radii = np.clip(values @ DIRECTIONS.T, 0, 1)
    candidates = radii[:, :, None] * DIRECTIONS[None]
    squared = ((values[:, None] - candidates) ** 2).sum(axis=2)
    branch = squared.argmin(axis=1)
    projected = np.eye(3)[branch] * radii[np.arange(len(values)), branch, None]
    displacement = np.linalg.norm(values - to_planar(projected), axis=1)
    return projected, displacement


def threshold_points(points, a):
    validate_points(points)
    if not 0 <= a < .5:
        raise ValueError("Full-vertex threshold must be in [0,.5).")
    result = np.array(points, copy=True)
    if a == 0:
        return result
    radius = result.sum(axis=1)
    result[radius <= a] = 0
    endpoint = radius >= 1 - a
    result[endpoint] = np.eye(3)[result[endpoint].argmax(axis=1)]
    return result


def category_labels(points):
    validate_points(points)
    radius = points.sum(axis=1)
    branch = points.argmax(axis=1)
    return np.where(radius == 0, 0, np.where(radius == 1, branch + 4, branch + 1))


def event_features(points):
    labels = category_labels(points)
    radius = points.sum(axis=1)
    branch = points.argmax(axis=1)
    coarse = np.where(labels >= 4, labels - 3, labels)
    events = [np.eye(4)[coarse]]
    for j in range(3):
        events.append(((radius[:, None] <= RADIUS_GRID)
                       & (radius[:, None] > 0) & (branch[:, None] == j)).astype(float))
    return np.concatenate(events, axis=1)


def brier_score(predicted_probabilities, observed):
    truth = event_features(observed)
    if predicted_probabilities.shape != truth.shape:
        raise ValueError("Predicted event probabilities must match held-out pairs.")
    return float(np.mean(np.sum((predicted_probabilities - truth) ** 2, axis=1)))


def intrinsic_w2(x, y):
    validate_points(x)
    validate_points(y)
    if len(x) != len(y):
        raise ValueError("Exact equal-weight assignment requires equal sample counts.")
    cost = distances(x, y) ** 2
    row, col = linear_sum_assignment(cost)
    return float(np.sqrt(cost[row, col].mean()))


def probability_record(points, x, scenario):
    labels = category_labels(points)
    estimated = np.bincount(labels, minlength=7) / len(points)
    p0, pi = law_probabilities([x], scenario)
    truth = np.r_[p0[0], (1 - p0[0]) * pi[0], np.zeros(3)]
    combined = estimated[1:4] + estimated[4:]
    se = np.sqrt(estimated * (1 - estimated) / len(points))
    result = {
        "label_tv": float(.5 * np.abs(estimated - truth).sum()),
        "center_absolute_error": float(abs(estimated[0] - truth[0])),
        "center_signed_error": float(estimated[0] - truth[0]),
        "false_endpoint_mass": float(estimated[4:].sum()), "validity": 1.,
        "probability_sample_count": len(points),
    }
    for index in range(7):
        result[f"category_{index}_probability"] = float(estimated[index])
        result[f"category_{index}_truth"] = float(truth[index])
        result[f"category_{index}_mc_se"] = float(se[index])
    for j in range(3):
        result[f"arm_{j + 1}_probability"] = float(combined[j])
        result[f"arm_{j + 1}_truth"] = float(truth[j + 1])
        for t in RADIUS_GRID:
            event = (points[:, j] > 0) & (points[:, j] <= t)
            result[f"arm_{j + 1}_cdf_{t:.1f}"] = float(event.mean())
            result[f"arm_{j + 1}_cdf_truth_{t:.1f}"] = float(
                truth[j + 1] * radial_cdf(t, [x], scenario)[0]
            )
    return result


def conditional_resample(train_x, train_y, x, count, bandwidth, seed):
    if count < 1 or bandwidth <= 0:
        raise ValueError("Positive resampling count and bandwidth are required.")
    rng = np.random.default_rng(seed)
    result = []
    for condition in conditions(x):
        logits = -.5 * ((train_x - condition) / bandwidth) ** 2
        probability = softmax(logits)
        result.append(train_y[rng.choice(len(train_y), count, p=probability)])
    return np.asarray(result)
