"""Practical conditional DDPM and opt-in second-stage tripod comparators."""

import time

import numpy as np
import torch
from torch import nn

from conditional_star import (
    category_labels, event_features, project_planar, tensor, to_planar,
)
from run_experiments import Denoiser, forward_noise, reverse_mean, schedule, seed_everything


def optimization_generator(seed):
    child = int(np.random.SeedSequence([seed, 20261103, 1]).generate_state(1)[0])
    return torch.Generator().manual_seed(child)


class ConditionalDenoiser(nn.Module):
    def __init__(self, dimension, width, ignore_x=False):
        super().__init__()
        self.dimension = dimension
        self.ignore_x = ignore_x
        self.network = Denoiser(dimension + 1, width, spatial_frequencies=4)

    def forward(self, z, t, x):
        condition = torch.zeros_like(x) if self.ignore_x else 2 * x - 1
        return self.network(torch.cat((z, condition[:, None]), dim=1), t)[:, :self.dimension]


@torch.no_grad()
def sample_ddpm(model, x, count, timesteps, seed, check_budget):
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(np.repeat(x, count), dtype=torch.float32)
    z = torch.randn((len(x), model.dimension), generator=rng)
    beta, alpha, alpha_bar, posterior = schedule(timesteps, torch.device("cpu"))
    for step in reversed(range(timesteps)):
        if step % 20 == 0:
            check_budget()
        t = torch.full((len(x),), step, dtype=torch.long)
        z = reverse_mean(z, model(z, t, x), step, beta, alpha, alpha_bar)
        if step:
            z = z + posterior[step].sqrt() * torch.randn(z.shape, generator=rng)
    if not torch.isfinite(z).all():
        raise FloatingPointError("Nonfinite conditional DDPM samples.")
    return z.reshape(-1, count, model.dimension).numpy()


def fit_ddpm(train_z, train_x, args, seed, select, folder, check_budget, ignore_x=False):
    seed_everything(seed)
    model = ConditionalDenoiser(train_z.shape[1], args.width, ignore_x)
    rng = optimization_generator(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    z = torch.as_tensor(train_z, dtype=torch.float32)
    x = torch.as_tensor(train_x, dtype=torch.float32)
    _, _, alpha_bar, _ = schedule(args.timesteps, torch.device("cpu"))
    best_score, best_state, selected, history = float("inf"), None, None, []
    started = time.perf_counter()
    optimization_seconds, validation_seconds = 0., 0.
    for update in range(1, args.updates + 1):
        check_budget()
        before = time.perf_counter()
        model.train()
        index = torch.randint(len(z), (args.batch_size,), generator=rng)
        t = torch.randint(args.timesteps, (len(index),), generator=rng)
        noise = torch.randn(z[index].shape, generator=rng)
        prediction = model(forward_noise(z[index], noise, t, alpha_bar), t, x[index])
        loss = (prediction - noise).square().sum(dim=1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite conditional training loss.")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimization_seconds += time.perf_counter() - before
        if update % args.checkpoint_every == 0 or update == args.updates:
            before = time.perf_counter()
            score, candidate, records = select(model)
            validation_seconds += time.perf_counter() - before
            history.append({"update": update, "training_loss": float(loss.detach()),
                            "validation_brier": score, "candidates": records})
            if score < best_score:
                best_score = score
                selected = candidate
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No conditional checkpoint selected.")
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "dimension": model.dimension, "width": args.width,
                "ignore_x": ignore_x, "timesteps": args.timesteps,
                "selected": selected, "validation_brier": best_score}, folder)
    return model.eval(), selected, {
        "history": history, "validation_brier": best_score, "selected": selected,
        "fit_seconds": time.perf_counter() - started,
        "optimization_seconds": optimization_seconds, "validation_seconds": validation_seconds,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def decode_array(decoder, coordinates, chunk=512):
    original = coordinates.shape[:-1]
    values = tensor(coordinates.reshape(-1, coordinates.shape[-1]))
    with torch.no_grad():
        parts = [decoder(values[start:start + chunk]).numpy()
                 for start in range(0, len(values), chunk)]
    return np.concatenate(parts).reshape(*original, 3)


def predicted_events(samples):
    return np.asarray([event_features(group).mean(axis=0) for group in samples])


class ProjectedMLP(nn.Module):
    def __init__(self, dimension, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimension, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2)).double()

    def forward(self, z):
        planar = self.network(z)
        directions = tensor(to_planar(np.eye(3))).to(planar.device)
        radii = (planar @ directions.T).clamp(0, 1)
        error = (planar[:, None] - radii[:, :, None] * directions).square().sum(dim=2)
        branch = error.argmin(dim=1)
        radius = radii.gather(1, branch[:, None])
        return nn.functional.one_hot(branch, 3) * radius


def fit_decoder(model, train_z, train_y, val_z, val_y, args, seed, check_budget):
    seed_everything(seed)
    model.double()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    rng = optimization_generator(seed)
    train_z, train_y, val_z, val_y = map(tensor, (train_z, train_y, val_z, val_y))
    best, state, history = float("inf"), None, []
    started = time.perf_counter()
    for update in range(1, args.decoder_updates + 1):
        check_budget()
        index = torch.randint(len(train_z), (args.batch_size,), generator=rng)
        predicted = model(train_z[index])
        # Intrinsic squared reconstruction error, not ambient display error.
        loss = (predicted - train_y[index]).abs().sum(dim=1).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite second-stage decoder loss.")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update % args.checkpoint_every == 0 or update == args.decoder_updates:
            with torch.no_grad():
                score = float((model(val_z) - val_y).abs().sum(dim=1).square().mean())
            history.append({"update": update, "paired_validation_intrinsic_mse": score})
            if score < best:
                best = score
                state = {key: value.clone() for key, value in model.state_dict().items()}
    if state is None:
        raise RuntimeError("No second-stage decoder checkpoint selected.")
    model.load_state_dict(state)
    return {"fit_seconds": time.perf_counter() - started, "history": history,
            "paired_validation_intrinsic_mse": best}


class CategoricalRadial(nn.Module):
    """Explicit center category plus uniform radial histogram cells on each arm."""

    def __init__(self, width, bins=20):
        super().__init__()
        self.bins = bins
        self.network = nn.Sequential(nn.Linear(1, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU(),
                                     nn.Linear(width, 1 + 3 * bins))

    def forward(self, x):
        return self.network((2 * x - 1)[:, None])

    @torch.no_grad()
    def sample(self, x, count, seed):
        rng = np.random.default_rng(seed)
        probabilities = self(torch.as_tensor(x, dtype=torch.float32)).softmax(dim=1).numpy()
        groups = []
        for p in probabilities:
            category = rng.choice(len(p), count, p=p)
            arm = np.maximum(category - 1, 0) // self.bins
            cell = np.maximum(category - 1, 0) % self.bins
            u = np.clip(rng.uniform(size=count), np.nextafter(0., 1.), np.nextafter(1., 0.))
            radius = (cell + u) / self.bins
            radius = np.clip(radius, np.nextafter(0., 1.), np.nextafter(1., 0.))
            radius[category == 0] = 0
            groups.append(np.eye(3)[arm] * radius[:, None])
        return np.asarray(groups)


def fit_categorical(train_x, train_y, val_x, val_y, args, seed, check_budget):
    seed_everything(seed)
    model = CategoricalRadial(args.width)
    rng = optimization_generator(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    radius = train_y.sum(axis=1)
    labels = 1 + train_y.argmax(axis=1) * model.bins + np.minimum(
        (radius * model.bins).astype(int), model.bins - 1)
    labels[category_labels(train_y) == 0] = 0
    x = torch.as_tensor(train_x, dtype=torch.float32)
    target = torch.as_tensor(labels, dtype=torch.long)
    best, state, history = float("inf"), None, []
    started = time.perf_counter()
    from conditional_star import brier_score
    for update in range(1, args.updates + 1):
        check_budget()
        index = torch.randint(len(x), (args.batch_size,), generator=rng)
        loss = nn.functional.cross_entropy(model(x[index]), target[index])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite categorical/radial training loss.")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update % args.checkpoint_every == 0 or update == args.updates:
            samples = model.sample(val_x, args.validation_draws, seed + 1)
            score = brier_score(predicted_events(samples), val_y)
            history.append({"update": update, "validation_brier": score})
            if score < best:
                best = score
                state = {k: v.clone() for k, v in model.state_dict().items()}
    if state is None:
        raise RuntimeError("No categorical/radial checkpoint selected.")
    model.load_state_dict(state)
    return model.eval(), {"fit_seconds": time.perf_counter() - started,
                          "history": history, "radial_bins_per_arm": model.bins}
