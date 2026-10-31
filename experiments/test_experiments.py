import unittest

import numpy as np
import torch

from run_experiments import (
    Encoder, choose_references, empirical_w2, make_data,
    reconstruction_loss, schedule,
)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1]], dtype=np.float32)
        self.encoder = Encoder(self.points, 0, np.array([1, 2, 3]))

    def test_encoder_matches_distance_contrasts(self):
        landmarks, base = self.points[[1, 2, 3]], self.points[0]
        lengths = np.linalg.norm(landmarks - base, axis=1)
        distances_to_base = np.sum((self.points - base) ** 2, axis=1)
        distances_to_landmarks = np.sum((self.points[:, None] - landmarks) ** 2, axis=2)
        expected = (distances_to_base[:, None] + lengths ** 2 - distances_to_landmarks) / (2 * lengths)
        actual = self.encoder.raw(torch.tensor(self.points)).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        np.testing.assert_allclose(actual[0], 0)
        np.testing.assert_allclose(np.diag(actual[[1, 2, 3]]), lengths, atol=1e-6)

    def test_standardization_and_linear_recovery(self):
        z = self.encoder(torch.tensor(self.points))
        np.testing.assert_allclose(z.mean(dim=0), 0, atol=1e-6)
        np.testing.assert_allclose(z.std(dim=0), 1, atol=1e-6)
        raw = z * self.encoder.scale + self.encoder.mean
        recovered = raw @ torch.linalg.pinv(self.encoder.directions).T + self.encoder.base
        np.testing.assert_allclose(recovered, self.points, atol=1e-6)

    def test_degenerate_references_rejected(self):
        with self.assertRaisesRegex(ValueError, "coincides"):
            Encoder(self.points, 0, np.array([0, 1, 2]))
        collinear = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "span"):
            Encoder(collinear, 0, np.array([1, 2, 3]))

    def test_reconstruction_losses_and_gradients(self):
        target = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        predicted = (target + torch.tensor([1.0, 2.0])).requires_grad_()
        residual = self.encoder.raw(predicted) - self.encoder.raw(target)
        expected_huber = torch.where(
            residual.abs() <= 1, 0.5 * residual.square(), residual.abs() - 0.5
        ).sum(dim=1).mean()
        actual = reconstruction_loss(predicted, target, self.encoder, "huber", 1)
        self.assertAlmostEqual(actual.item(), expected_huber.item(), places=6)
        self.assertAlmostEqual(
            reconstruction_loss(predicted, target, self.encoder, "object_mse").item(), 5.0
        )
        for loss in ("embedded_mse", "object_mse", "huber", "standardized_mse"):
            self.assertEqual(reconstruction_loss(target, target, self.encoder, loss).item(), 0)
            value = reconstruction_loss(predicted, target, self.encoder, loss)
            gradient = torch.autograd.grad(value, predicted)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_embedded_mse_uses_sum_not_coordinate_mean(self):
        target = torch.tensor([[0.0, 0.0]])
        predicted = torch.tensor([[1.0, 0.0]])
        self.assertAlmostEqual(
            reconstruction_loss(predicted, target, self.encoder, "embedded_mse").item(), 1.5
        )

    def test_reference_selection_is_reproducible_and_training_only(self):
        points = make_data("two_moons", 100, 13)
        for rule in ("manual", "random", "fps"):
            base, landmarks = choose_references(points, rule, 99)
            base_again, landmarks_again = choose_references(points, rule, 99)
            self.assertEqual(base, base_again)
            np.testing.assert_array_equal(landmarks, landmarks_again)
            self.assertEqual(len(set([base, *landmarks])), 4)
            self.assertTrue(np.all((landmarks >= 0) & (landmarks < len(points))))
            if rule == "fps":
                selected = [base]
                for index in landmarks:
                    distance = ((points[:, None] - points[selected]) ** 2).sum(axis=2).min(axis=1)
                    distance[selected] = -np.inf
                    self.assertEqual(index, int(np.argmax(distance)))
                    selected.append(index)

    def test_synthetic_draws(self):
        for dataset in ("two_moons", "eight_gaussians"):
            first = make_data(dataset, 1024, 1)
            self.assertEqual(first.shape, (1024, 2))
            self.assertTrue(np.isfinite(first).all())
            np.testing.assert_array_equal(first, make_data(dataset, 1024, 1))
            self.assertFalse(np.array_equal(first, make_data(dataset, 1024, 2)))

    def test_w2_is_distance_not_squared_cost(self):
        points = np.array([[0, 0], [1, 0], [2, 1]], dtype=float)
        self.assertEqual(empirical_w2(points, points[::-1]), 0)
        self.assertAlmostEqual(empirical_w2(points + [3, 4], points), 5.0)
        with self.assertRaises(ValueError):
            empirical_w2(points[:-1], points)
        invalid = points.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            empirical_w2(invalid, points)

    def test_diffusion_schedule(self):
        beta, alpha, alpha_bar, variance = schedule(200, torch.device("cpu"))
        self.assertTrue(torch.all((beta > 0) & (beta < 1)))
        torch.testing.assert_close(alpha, 1 - beta)
        self.assertTrue(torch.all(alpha_bar[1:] < alpha_bar[:-1]))
        self.assertLess(alpha_bar[-1].item(), 0.001)
        self.assertEqual(variance[0].item(), 0)
        self.assertTrue(torch.all(variance >= 0))


if __name__ == "__main__":
    unittest.main()
