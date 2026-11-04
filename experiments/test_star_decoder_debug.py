import unittest

import numpy as np
import torch

from conditional_star import (
    GaussianFrechet, StarEncoder, tensor, threshold_points,
)
from conditional_star_models import decode_array
from diagnose_star_decoder import (
    DiagnosticDecoder, balanced_dictionary, center_balance, confusion_record,
    identical_center_audit, reference_records,
)


class StarDecoderDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_confusion_separates_lost_atoms_from_false_positives(self):
        truth = np.array([[0, 0, 0], [0, 0, 0], [.2, 0, 0], [0, .4, 0]], dtype=float)
        predicted = np.array([[0, 0, 0], [.01, 0, 0], [0, 0, 0], [0, .3, 0]], dtype=float)
        record = confusion_record(truth, predicted)
        self.assertEqual([record[k] for k in ("TP", "FN", "FP", "TN")], [1, 1, 1, 1])
        self.assertEqual(record["paired_signed_center_mass_error"], 0)
        self.assertGreater(record["intrinsic_reconstruction_mse"], 0)
        no_atoms = confusion_record(truth[2:], predicted[2:])
        self.assertIsNone(no_atoms["center_retention_rate"])

    def test_symmetrization_preserves_center_prior_and_radial_marginal(self):
        original = np.array([[0, 0, 0], [0, 0, 0], [.1, 0, 0], [.3, 0, 0], [0, .6, 0]])
        balanced, prior = balanced_dictionary(original)
        self.assertEqual(len(balanced), 11)
        self.assertAlmostEqual(prior.sum(), len(original))
        records = reference_records(balanced, prior)
        self.assertEqual(records[0]["count"], 2)
        self.assertEqual(records[0]["prior_mass"], 2)
        for arm in (1, 2, 3):
            self.assertEqual(records[arm]["count"], 3)
            self.assertAlmostEqual(records[arm]["prior_mass"], 1.)
            self.assertAlmostEqual(records[arm]["prior_radial_moment"], 1 / 3)
        original_radius = original.sum(axis=1)
        balanced_radius = balanced.sum(axis=1)
        for radius in np.unique(original_radius):
            self.assertAlmostEqual(float(prior[balanced_radius == radius].sum()),
                                   int((original_radius == radius).sum()))

    def test_diagnostic_original_is_existing_decoder_without_kernel_change(self):
        encoder = StarEncoder.controlled()
        references = np.r_[np.zeros((2, 3)), np.eye(3) * .5]
        baseline = GaussianFrechet(encoder, references, .1)
        diagnostic = DiagnosticDecoder(encoder, references, .1, baseline.h_scale)
        points = tensor(np.r_[np.zeros((5, 3)), np.eye(3) * .1])
        with torch.no_grad():
            z = encoder(points)
            torch.testing.assert_close(baseline(z), diagnostic(z), rtol=0, atol=0)

    def test_center_folded_moments_and_balanced_oracle(self):
        encoder = StarEncoder.controlled()
        references = np.r_[np.zeros((1, 3)), np.array([[.2, 0, 0]] * 6),
                           np.array([[0, .2, 0], [0, 0, .2]])]
        common_scale = GaussianFrechet(encoder, references, .2).h_scale
        original = DiagnosticDecoder(encoder, references, .2, common_scale)
        result = center_balance(original)
        self.assertEqual(result["center_decoded_J"], 1)
        self.assertGreater(result["center_decoded_R"], 0)
        self.assertAlmostEqual(result["g_1"], result["center_decoded_R"])
        balanced, prior = balanced_dictionary(references)
        fixed = DiagnosticDecoder(encoder, balanced, .2, common_scale, prior)
        center = center_balance(fixed)
        self.assertEqual(center["center_decoded_R"], 0)
        self.assertEqual(fixed.h, original.h)
        self.assertAlmostEqual(center["T_1"], center["T_2"])
        self.assertAlmostEqual(center["T_2"], center["T_3"])
        self.assertTrue(center["strict_center_rule_satisfied"])

    def test_fixed_centers_all_displaced_and_cross_batch_audit(self):
        encoder = StarEncoder.controlled()
        references = np.array([[.2, 0, 0], [.3, 0, 0], [0, .9, 0], [0, 0, .9]])
        decoder = GaussianFrechet(encoder, references, .4)
        truth = np.r_[np.zeros((50, 3)), np.eye(3) * .1]
        with torch.no_grad():
            z = encoder(tensor(truth)).numpy()
        decoded = decode_array(decoder, z)
        result = identical_center_audit(encoder, decoder, truth, z, decoded)
        self.assertTrue(result["passed"])
        self.assertEqual(result["center_encoded_unique_rows"], 1)
        centers = decoded[:50].sum(axis=1) == 0
        self.assertTrue(np.all(centers) or np.all(~centers))
        decoded[0] = [.5, 0, 0]
        with self.assertRaises(AssertionError):
            identical_center_audit(encoder, decoder, truth, z, decoded)

    def test_threshold_cannot_remove_false_raw_centers(self):
        truth = np.array([[.1, 0, 0], [0, .2, 0], [0, 0, .3]])
        decoded = np.array([[0, 0, 0], [0, .001, 0], [0, 0, .02]])
        for a in (0, .0025, .01, .05):
            result = confusion_record(truth, threshold_points(decoded, a))
            self.assertGreaterEqual(result["FP"], 1)
        self.assertEqual(confusion_record(truth, threshold_points(decoded, .05))["FP"], 3)

    def test_controlled_representation_injective_but_not_euclidean_isometry(self):
        encoder = StarEncoder.controlled()
        points = np.r_[np.zeros((1, 3)), np.eye(3) * .4]
        with torch.no_grad():
            z = encoder(tensor(points)).numpy()
        radius = np.linalg.norm(z, axis=1)
        recovered = np.eye(3)[z.argmax(axis=1)] * radius[:, None]
        np.testing.assert_allclose(recovered, points, atol=1e-14)
        intrinsic = .8
        self.assertAlmostEqual(np.linalg.norm(z[1] - z[2]), intrinsic * np.sqrt(2 / 3))
        self.assertNotAlmostEqual(np.linalg.norm(z[1] - z[2]), intrinsic)
        self.assertAlmostEqual(np.max(np.abs(np.sqrt(3) * (z[1] - z[2]))), intrinsic)


if __name__ == "__main__":
    unittest.main()
