import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from conditional_star import (
    H_RATIOS, SCENARIOS, GaussianFrechet, StarEncoder, brier_score, category_labels,
    conditional_resample, decoder_diagnostics, distances, event_features, intrinsic_w2,
    jr_arrays, law_probabilities,
    probability_record, project_planar, radial_cdf, sample_pairs, stream_seed, tensor,
    threshold_points, to_planar, validate_points, weighted_mean,
)
from conditional_star_models import (
    CategoricalRadial, ConditionalDenoiser, ProjectedMLP, decode_array, fit_categorical,
    fit_ddpm, fit_decoder, predicted_events, sample_ddpm,
)
from run_conditional_star import (
    Budget, artifact_stem, fit_unit, parse_args, summarize_rows, summarize_selected,
)


class ConditionalStarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_scenarios_support_and_role_streams(self):
        seeds = {stream_seed(100, scenario, role, draw)
                 for scenario in SCENARIOS for role in range(1, 8) for draw in range(3)}
        self.assertEqual(len(seeds), len(SCENARIOS) * 7 * 3)
        for scenario in SCENARIOS:
            seed = stream_seed(100, scenario, 1)
            x, y = sample_pairs(1000, scenario, seed)
            validate_points(y)
            xx, yy = sample_pairs(1000, scenario, seed)
            np.testing.assert_array_equal(x, xx)
            np.testing.assert_array_equal(y, yy)
            j, r = jr_arrays(y)
            restored = np.eye(3)[np.maximum(j - 1, 0)] * r[:, None]
            np.testing.assert_array_equal(y, restored)
            self.assertTrue(np.all(j[r == 0] == 0))
            self.assertTrue(np.all(y.sum(axis=1) < 1))
            if scenario != SCENARIOS[1]:
                self.assertTrue(np.all(y.sum(axis=1) > 0))
            else:
                self.assertGreater(np.mean(y.sum(axis=1) == 0), .1)

    def test_known_laws_and_hard_negative_threshold(self):
        p0, pi = law_probabilities([0, .5, 1], SCENARIOS[1])
        np.testing.assert_allclose(p0, [.05, .35, .05])
        np.testing.assert_allclose(pi.sum(axis=1), 1)
        self.assertGreater(pi[0, 0], pi[0, 1])
        self.assertAlmostEqual(radial_cdf(.01, [.3], SCENARIOS[2])[0], .114625, places=14)
        _, y = sample_pairs(100000, SCENARIOS[2], 123, x=[.3])
        empirical = np.mean(threshold_points(y, .01).sum(axis=1) == 0)
        self.assertLess(abs(empirical - .114625), .004)
        self.assertTrue(np.all(y.sum(axis=1) > 0))

    def test_intrinsic_w2_not_ambient_or_mean_distance(self):
        x = np.array([[.4, 0, 0], [1., 0, 0]])
        y = np.array([[0, .4, 0], [0, 1., 0]])
        self.assertAlmostEqual(intrinsic_w2(x, y), 1.4)
        x = np.zeros((2, 3))
        y = np.array([[1., 0, 0], [0, 0, 0]])
        self.assertAlmostEqual(intrinsic_w2(x, y), np.sqrt(.5))
        self.assertEqual(intrinsic_w2(y, y[::-1]), 0)
        with self.assertRaises(ValueError):
            intrinsic_w2(x, y[:1])

    def test_exact_frechet_and_strict_small_positive_radius(self):
        references = tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [0, 0, 0]])
        w = tensor([[.5 + 1e-10, .5 - 1e-10, 0, 0], [.2, .2, .2, .4], [1, 0, 0, 0]])
        actual = weighted_mean(w, references).numpy()
        self.assertGreater(actual[0, 0], 0)
        self.assertLess(actual[0, 0], 1e-8)
        self.assertEqual(category_labels(actual)[0], 1)
        np.testing.assert_array_equal(actual[1], 0)
        np.testing.assert_array_equal(actual[2], [1, 0, 0])
        grid = np.concatenate([np.eye(3)[i] * np.linspace(0, 1, 1001)[:, None] for i in range(3)])
        objective = (distances(actual, references.numpy()) ** 2 * w.numpy()).sum(axis=1)
        dense_best = (w.numpy() @ (distances(grid, references.numpy()) ** 2).T).min(axis=1)
        self.assertTrue(np.all(objective <= dense_best + 1e-12))

    def test_sample_references_scales_multiplicity_and_base_sensitivity(self):
        ref = np.concatenate((np.zeros((20, 3)), np.eye(3),
                              np.eye(3) * .5, np.eye(3) * .25, np.eye(3) * .75))
        encoder = StarEncoder(ref)
        self.assertEqual(encoder.metadata["center_multiplicity"], 20)
        self.assertEqual(len(set(encoder.indices)), 3)
        self.assertTrue(torch.all(encoder.lengths > 0))
        self.assertTrue(torch.all((encoder.scale >= .25) & (encoder.scale <= 4)))
        decoder = GaussianFrechet(encoder, ref, .1)
        self.assertEqual(len(decoder.references), len(ref))
        self.assertEqual(int((decoder.references.sum(dim=1) == 0).sum()), 20)
        alternate = StarEncoder(ref, base_kind="frechet", landmarks=encoder.indices)
        self.assertEqual(alternate.indices, encoder.indices)
        self.assertTrue(alternate.metadata["base_equals_medoid"])
        self.assertEqual(alternate.metadata["forced_landmark_changes"], [])
        self.assertEqual(len(StarEncoder(ref, k=12).indices), 12)
        asymmetric = np.array([[.1, 0, 0], [.9, 0, 0], [1., 0, 0], [0, .1, 0], [0, 0, .1]])
        medoid = StarEncoder(asymmetric)
        mean = StarEncoder(asymmetric, base_kind="frechet", landmarks=medoid.indices)
        self.assertFalse(mean.metadata["base_equals_medoid"])
        with self.assertRaises(ValueError):
            StarEncoder(np.zeros((10, 3)))

    def test_controlled_encoder_and_certified_radius(self):
        _, y = sample_pairs(256, SCENARIOS[0], 2)
        controlled = StarEncoder.controlled()
        z = controlled(tensor(y)).numpy()
        expected = (2 * y - y.sum(axis=1, keepdims=True)) / np.sqrt(3)
        np.testing.assert_allclose(z, expected, atol=1e-14)
        metric = distances(y, y)
        euclidean = np.linalg.norm(z[:, None] - z[None], axis=2)
        self.assertTrue(np.all(euclidean <= metric + 1e-14))
        self.assertTrue(np.all(euclidean >= np.sqrt(2 / 3) * metric - 1e-14))
        self.assertAlmostEqual(controlled.describe()["certified_coordinate_radius"], 1.)
        encoder = StarEncoder(y[:48])
        actual = encoder(tensor(y)).norm(dim=1).max().item()
        self.assertLessEqual(actual, encoder.describe()["certified_coordinate_radius"] + 1e-14)

    def test_full_vertex_threshold_categories_and_no_mutation(self):
        points = np.array([[0, 0, 0], [.001, 0, 0], [0, .999, 0],
                           [0, 0, 1], [.2, 0, 0]], dtype=float)
        before = points.copy()
        processed = threshold_points(points, .01)
        np.testing.assert_array_equal(category_labels(processed), [0, 0, 5, 6, 1])
        np.testing.assert_array_equal(points, before)
        np.testing.assert_array_equal(threshold_points(points, 0), points)
        with self.assertRaises(ValueError):
            threshold_points(points, .5)

    def test_projection_is_nearest_segment_and_records_displacement(self):
        values = np.array([[2., 0], [0, 0], [.1, -.4], [-1, -1]])
        points, repair = project_planar(values)
        validate_points(points)
        np.testing.assert_array_equal(points[0], [1, 0, 0])
        self.assertEqual(repair[0], 1)
        self.assertEqual(category_labels(points)[1], 0)
        grid = np.concatenate([np.eye(3)[j] * np.linspace(0, 1, 10001)[:, None] for j in range(3)])
        minimum = np.linalg.norm(values[:, None] - to_planar(grid)[None], axis=2).min(axis=1)
        self.assertTrue(np.all(repair <= minimum + 1e-12))

    def test_probability_scores_and_empirical_baseline(self):
        points = np.concatenate((np.zeros((1, 3)), np.eye(3) * .5))
        events = event_features(points)
        self.assertEqual(events.shape, (4, 31))
        self.assertEqual(brier_score(events, points), 0)
        record = probability_record(points, .5, SCENARIOS[1])
        self.assertAlmostEqual(record["category_0_probability"], .25)
        self.assertAlmostEqual(record["center_signed_error"], -.1)
        self.assertEqual(record["false_endpoint_mass"], 0)
        x = np.array([0., .3, .7, 1.])
        samples = conditional_resample(x, points, [0., 1.], 500, .01, 4)
        np.testing.assert_array_equal(samples[0], np.zeros((500, 3)))
        np.testing.assert_array_equal(samples[1], np.tile(points[-1], (500, 1)))

    def test_bounded_residual_and_projected_mlp_gradients(self):
        _, ref = sample_pairs(48, SCENARIOS[1], 20)
        encoder = StarEncoder(ref)
        z = encoder(tensor(ref)).detach()
        fixed = GaussianFrechet(encoder, ref, .2)
        learned = GaussianFrechet(encoder, ref, .2, learned=True)
        self.assertTrue(all(parameter.dtype == torch.float64 for parameter in learned.parameters()))
        for parameter in learned.residual.parameters():
            parameter.data.zero_()
        torch.testing.assert_close(fixed(z), learned(z))
        diagnostic = decoder_diagnostics(learned, z.numpy(), chunk=8)
        self.assertLessEqual(diagnostic["mean_localization_excess"],
                             diagnostic["gaussian_localization_bound"] + 1e-12)
        for model in (learned, ProjectedMLP(3, 8)):
            args = SimpleNamespace(decoder_updates=2, batch_size=8, checkpoint_every=1)
            record = fit_decoder(model, z.numpy(), ref, z.numpy(), ref, args, 4, lambda: None)
            self.assertTrue(np.isfinite(record["paired_validation_intrinsic_mse"]))
            validate_points(decode_array(model, z.numpy()))

    def test_conditional_and_x_ignored_networks_sampler_and_checkpoint(self):
        args = SimpleNamespace(width=8, updates=2, batch_size=8, timesteps=4, checkpoint_every=1)
        _, ref = sample_pairs(24, SCENARIOS[0], 8)
        x = np.linspace(0, 1, len(ref))
        encoder = StarEncoder(ref)
        z = encoder(tensor(ref)).detach().numpy()
        ignored = ConditionalDenoiser(3, 8, ignore_x=True)
        values = torch.randn((4, 3))
        t = torch.zeros(4, dtype=torch.long)
        torch.testing.assert_close(ignored(values, t, torch.zeros(4)),
                                   ignored(values, t, torch.ones(4)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model, selected, record = fit_ddpm(
                z, x, args, 9, lambda m: (float(sum(p.square().sum() for p in m.parameters()).detach()),
                                          .1, []), path, lambda: None)
            self.assertTrue(path.exists())
            self.assertEqual(selected, .1)
            self.assertEqual(len(record["history"]), 2)
            a = sample_ddpm(model, [.2, .8], 8, 4, 99, lambda: None)
            b = sample_ddpm(model, [.2, .8], 8, 4, 99, lambda: None)
            self.assertEqual(a.shape, (2, 8, 3))
            np.testing.assert_array_equal(a, b)

    def test_label_aware_baseline_and_parsed_budgets(self):
        x, y = sample_pairs(32, SCENARIOS[1], 19)
        args = SimpleNamespace(width=8, updates=2, batch_size=8, checkpoint_every=1, validation_draws=8)
        model, record = fit_categorical(x, y, x[:8], y[:8], args, 10, lambda: None)
        self.assertEqual(record["radial_bins_per_arm"], 20)
        samples = model.sample([0., .5, 1.], 256, 12)
        validate_points(samples.reshape(-1, 3))
        self.assertFalse(np.any(samples.sum(axis=2) == 1))
        parsed = parse_args(["--output", "unused"])
        self.assertEqual(parsed.seeds, [100, 101, 102, 103, 104])
        self.assertEqual(parsed.timesteps, 1000)
        self.assertEqual(parsed.updates, 10000)
        self.assertEqual(parsed.validation_draws, 128)
        self.assertFalse(parsed.second_stage)
        long_name = "oracle_controlled_center_endpoints_gaussian_frechet"
        self.assertLessEqual(len(artifact_stem(long_name)), 33)
        self.assertNotEqual(artifact_stem(long_name), artifact_stem(long_name + "_other"))
        smoke = parse_args(["--output", "unused", "--smoke"])
        self.assertEqual(smoke.seeds, [100])
        self.assertEqual(smoke.test_conditions, 3)
        budget = Budget(.001)
        budget.started -= 1
        with self.assertRaises(TimeoutError):
            budget.check()

    def test_integrated_w2_rms_and_between_seed_uncertainty(self):
        rows = []
        for seed in (100, 101):
            for draw in (0, 1, 2):
                for x, w2 in ((.25, 0.), (.75, 2.)):
                    rows.append({
                        "scenario": SCENARIOS[0], "method": "test", "stage": "test",
                        "h_ratio": .1, "threshold": 0., "perturbation_sigma": 0.,
                        "selected": True, "seed": seed, "draw": draw, "x": x, "w2": w2,
                        "label_tv": .1, "center_absolute_error": .2,
                        "center_signed_error": -.2, "false_endpoint_mass": 0., "validity": 1.,
                    })
        summary = summarize_rows(rows)[0]
        self.assertAlmostEqual(summary["w2"], np.sqrt(2))
        self.assertEqual(summary["independent_training_seed_count"], 2)
        self.assertEqual(summary["evaluation_draw_count_per_seed"], 3)
        self.assertEqual(summary["w2_between_seed_sd"], 0)
        one_seed = summarize_rows([r for r in rows if r["seed"] == 100])[0]
        self.assertIsNone(one_seed["w2_between_seed_sd"])
        different_h = [{**r, "h_ratio": .1 if r["seed"] == 100 else .2} for r in rows]
        selected = summarize_selected(different_h)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["independent_training_seed_count"], 2)

    def test_second_stage_wiring_without_launching_ablation_training(self):
        args = parse_args(["--output", "unused", "--smoke", "--second-stage"])
        x, y = sample_pairs(48, SCENARIOS[1], 19)
        encoder = StarEncoder(y)
        data = (x, y, x, y, y)

        def fake_fit(train_z, train_x, args, seed, select, folder, check, ignore_x=False):
            return ConditionalDenoiser(train_z.shape[1], 8, ignore_x), .1, {"fit_seconds": 0.}

        with tempfile.TemporaryDirectory() as tmp, \
                patch("run_conditional_star.fit_ddpm", side_effect=fake_fit) as diffusion_fit, \
                patch("run_conditional_star.fit_decoder", return_value={"fit_seconds": 0.}) as decoder_fit, \
                patch("run_conditional_star.choose_coordinate", return_value=(1., .1, [])), \
                patch("run_conditional_star.fit_categorical",
                      return_value=(CategoricalRadial(8), {"fit_seconds": 0.})):
            fitted = fit_unit(args, SCENARIOS[1], 100, Path(tmp), data, Budget(30), encoder)
            encoders, models, decoder_sets, selected_h, _, extras, _ = fitted
            self.assertEqual(set(encoders), {"medoid_k3", "medoid_k6", "medoid_k12", "frechet_base_k3"})
            self.assertEqual(diffusion_fit.call_count, 6)
            self.assertEqual(decoder_fit.call_count, 5)
            self.assertEqual(len(decoder_sets["medoid_k12"][0].encoder.mean), 12)
            self.assertEqual(set(extras), {"learned", "mlp", "categorical"})
            self.assertEqual(selected_h["learned"], .1)
            self.assertEqual(models["medoid_k12"].dimension, 12)


if __name__ == "__main__":
    unittest.main()
