import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.optimize import minimize
import torch

from metric_spaces import (
    ALL_DATASETS, METRIC_DATASETS, FrechetDecoder, MetricEncoder, ObjectSpace,
    fps_indices, metric_loss, object_diagnostics, object_w2,
)
from run_experiments import Encoder, fit_decoder
from run_remaining_experiments import stream_seed, summarize


class MetricExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_synthetic_objects_and_support(self):
        for dataset in ALL_DATASETS:
            space = ObjectSpace(dataset)
            points = space.sample(512, 19)
            space.validate(points)
            np.testing.assert_array_equal(points, space.sample(512, 19))
            self.assertFalse(np.array_equal(points, space.sample(512, 20)))
            self.assertLess(object_w2(points[:16], points[:16][::-1].copy(), space), 1e-6)
            if dataset == "sphere":
                self.assertGreaterEqual(float(points[:, 2].min()), np.cos(0.53) - 1e-6)
            if dataset == "tree":
                self.assertGreater((points.sum(axis=1) == 0).sum(), 0)

    def test_known_geodesic_and_star_distances(self):
        sphere = ObjectSpace("sphere")
        points = torch.tensor([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]], dtype=torch.float64)
        distances = sphere.pairwise_squared(points, points).sqrt()
        self.assertAlmostEqual(distances[0, 1].item(), np.pi / 2)
        self.assertAlmostEqual(distances[0, 2].item(), np.pi, places=6)
        star = ObjectSpace("tree")
        x = torch.tensor([[.3, 0., 0.], [0., .4, 0.]], dtype=torch.float64)
        self.assertAlmostEqual(star.pairwise_squared(x, x)[0, 1].sqrt().item(), .7)

    def test_distribution_w2_and_barycenter(self):
        space = ObjectSpace("distributions", quantiles=4)
        a = np.array([[0., 1., 2., 3.]])
        b = np.array([[1., 3., 5., 7.]])
        self.assertAlmostEqual(object_w2(a, b, space), np.sqrt(7.5))
        references = torch.tensor(np.concatenate([a, b]))
        weights = torch.tensor([[.25, .75]], dtype=torch.float64)
        actual = space.frechet(weights, references)
        torch.testing.assert_close(actual, .25 * references[:1] + .75 * references[1:])
        space.validate(actual.numpy())

    def test_star_mean_matches_dense_global_search(self):
        space = ObjectSpace("tree")
        references = torch.tensor([[.8, 0., 0.], [0., .7, 0.], [0., 0., .9]])
        weights = torch.tensor([[.8, .1, .1], [.3, .3, .4], [0., 1., 0.]])
        mean = space.frechet(weights, references)
        torch.testing.assert_close(mean[0], torch.tensor([.48, 0., 0.]))
        torch.testing.assert_close(mean[1], torch.zeros(3))
        torch.testing.assert_close(mean[2], references[1])
        grid = torch.cat([torch.eye(3)[i:i + 1] * torch.linspace(0, 1, 1001)[:, None]
                          for i in range(3)])
        cost = space.pairwise_squared(grid, references)
        actual = (space.pairwise_squared(mean, references) * weights).sum(dim=1)
        optimum_grid = (weights @ cost.T).min(dim=1).values
        self.assertTrue(torch.all(actual <= optimum_grid + 1e-6))

    def test_spherical_frechet_mean_and_solver_residual(self):
        space = ObjectSpace("sphere", sphere_iterations=24)
        references = torch.tensor(space.sample(12, 21), dtype=torch.float64)
        weights = torch.softmax(torch.arange(12, dtype=torch.float64)[None] / 5, dim=1)
        mean = space.frechet(weights, references)
        residual = space.sphere_log_average(mean, weights, references).norm().item()
        self.assertLess(residual, 1e-8)
        ref, w = references.numpy(), weights.numpy()[0]

        def objective(v):
            m = np.r_[v, 1.]
            m /= np.linalg.norm(m)
            return np.sum(w * np.arccos(np.clip(ref @ m, -1, 1)) ** 2)

        optimized = minimize(objective, [0., 0.], method="BFGS", tol=1e-9)
        actual = (weights * space.pairwise_squared(mean, references)).sum().item()
        self.assertAlmostEqual(actual, optimized.fun, places=7)
        opposite = torch.tensor([[np.sin(.4), 0., np.cos(.4)],
                                 [-np.sin(.4), 0., np.cos(.4)]], dtype=torch.float64)
        center = space.frechet(torch.tensor([[.5, .5]], dtype=torch.float64), opposite)
        torch.testing.assert_close(center, torch.tensor([[0., 0., 1.]], dtype=torch.float64))

    def test_generic_encoder_matches_planar_specialization(self):
        space = ObjectSpace("two_moons")
        points = space.sample(100, 7)
        generic = MetricEncoder(points, space, 3)
        specialized = Encoder(points, generic.indices[0], np.array(generic.indices[1:]))
        torch.testing.assert_close(generic.raw(torch.tensor(points)),
                                   specialized.raw(torch.tensor(points)), atol=2e-6, rtol=2e-5)

    def test_distinct_metric_references(self):
        space = ObjectSpace("tree")
        points = torch.tensor(space.sample(100, 7))
        selected = fps_indices(points, space, 12)
        self.assertEqual(len(set(selected)), 12)
        self.assertTrue(torch.all(
            space.pairwise_squared(points[selected], points[selected])
            + torch.eye(12) > 0
        ))
        with self.assertRaisesRegex(ValueError, "distinct"):
            fps_indices(torch.zeros(10, 3), space, 2)

    def test_frechet_decoders_valid_gradients_and_localization(self):
        for dataset in METRIC_DATASETS:
            space = ObjectSpace(dataset)
            train = space.sample(80, 33)
            encoder = MetricEncoder(train, space, 5)
            references = torch.tensor(train[fps_indices(torch.tensor(train), space, 16)])
            decoder = FrechetDecoder(encoder, references, space, .3, 16, True)
            target = torch.tensor(train[:24])
            z = encoder(target)
            predicted = decoder(z)
            space.validate(predicted.detach().numpy())
            loss = metric_loss(predicted, target, encoder, "embedded_mse")
            loss.backward()
            grads = [p.grad for p in decoder.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
            diagnostics = decoder.diagnostics(z)
            self.assertLessEqual(diagnostics["generated_localization_mean"],
                                 diagnostics["kernel_localization_bound"] + 1e-5)
            self.assertLess(diagnostics["weight_sum_max_error"], 1e-5)

    def test_generalized_decoder_training(self):
        space = ObjectSpace("tree")
        train, validation = space.sample(64, 12), space.sample(24, 13)
        encoder = MetricEncoder(train, space, 5)
        references = torch.tensor(train[:16])
        args = SimpleNamespace(width=16, decoder_steps=3, batch_size=16)
        model, history, score, elapsed = fit_decoder(
            train, validation, encoder, "embedded_mse", 1., args, 34, torch.device("cpu"),
            model_factory=lambda: FrechetDecoder(encoder, references, space, .3, 16, True),
            loss_function=metric_loss,
        )
        self.assertTrue(np.isfinite(score))
        self.assertGreater(elapsed, 0)
        self.assertEqual(history[-1]["step"], 3)
        space.validate(model(encoder(torch.tensor(validation))).detach().numpy())

    def test_ambient_repairs_and_invalid_objects(self):
        sphere = ObjectSpace("sphere")
        sphere.validate(sphere.repair_ambient(torch.tensor([[2., 0., 1.]])).numpy())
        with self.assertRaisesRegex(ValueError, "zero vector"):
            sphere.repair_ambient(torch.zeros(1, 3))
        with self.assertRaises(ValueError):
            sphere.validate(np.array([[2., 0., 0.]]))
        tree = ObjectSpace("tree")
        values = torch.tensor([[-1., -2., -3.], [.4, .8, -.2], [2., .3, .1]])
        actual = tree.repair_ambient(values)
        torch.testing.assert_close(actual, torch.tensor([[0., 0., 0.], [0., .8, 0.], [1., 0., 0.]]))
        tree.validate(actual.numpy())
        with self.assertRaises(ValueError):
            tree.validate(np.array([[.5, .5, 0.]]))
        distributions = ObjectSpace("distributions", quantiles=4)
        distributions.validate(distributions.repair_ambient(torch.tensor([[3., 1., 2., 0.]])).numpy())

    def test_tree_probability_diagnostics(self):
        points = np.array([[1., 0., 0.]] * 2 + [[0., 1., 0.]] * 3
                          + [[0., 0., 1.]] * 4 + [[0., 0., 0.]])
        result = object_diagnostics(points, ObjectSpace("tree"))
        self.assertEqual(result["vertex_mass"], .1)
        self.assertEqual(result["branch_vertex_total_variation"], 0.)

    def test_independent_streams(self):
        seeds = {stream_seed(seed, dataset, role, draw)
                 for seed in range(100, 105) for dataset in ALL_DATASETS
                 for role in (1, 2, 3, 4, 10, 11, 12, 13, 20, 21, 22)
                 for draw in range(3)}
        self.assertEqual(len(seeds), 5 * 5 * 11 * 3)
        self.assertEqual(stream_seed(100, "sphere", 1), stream_seed(100, "sphere", 1))

    def test_summary_averages_draws_before_seed_variability(self):
        rows = []
        for seed, scores in ((100, [1., 3.]), (101, [5., 7.])):
            for draw, score in enumerate(scores):
                for method, value in (("direct_ddpm", score), ("kernel_frechet", score + 1)):
                    rows.append({"dataset": "sphere", "seed": seed, "draw": draw,
                                 "landmarks": "fps", "method": method, "w2": value})
        with tempfile.TemporaryDirectory() as folder:
            summarize(rows, Path(folder))
            output = json.loads((Path(folder) / "summary.json").read_text())
        direct = next(row for row in output if row["method"] == "direct_ddpm")
        self.assertEqual(direct["independent_training_seeds"], 2)
        self.assertEqual(direct["evaluation_rows"], 4)
        self.assertEqual(direct["w2_mean"], 4.)
        self.assertAlmostEqual(direct["w2_between_seed_sd"], np.sqrt(8))
        learned = next(row for row in output if row["method"] == "kernel_frechet")
        self.assertEqual(learned["paired_delta_w2_vs_direct_mean"], 1.)


if __name__ == "__main__":
    unittest.main()
