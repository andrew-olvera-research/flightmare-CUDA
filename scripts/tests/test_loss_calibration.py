import importlib.util
from pathlib import Path
import unittest

# Exercise the portable calibration logic without CUDA on the hosted runner.
path = Path(__file__).resolve().parents[2] / "flightlib-cuda/python/flightlib_cuda/loss_calibration.py"
spec = importlib.util.spec_from_file_location("loss_calibration", path)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


class TripletCalibrationTest(unittest.TestCase):
    def test_partial_receptions_reduce_borrowing_despite_shared_misses(self):
        model = calibration.TripletLossCalibrator([[0, 0, 3] * 16], 2)
        # Both sessions have excess complete misses, but only the pilots hear
        # the third copy alone. Repeating that pattern weakens the reference.
        pilot = [4, 4, 7, 7, 7, 7, 0, 7]
        model.observe([pilot])
        initial_weight = model.reference_weight
        for _ in range(3):
            model.observe([pilot])
        self.assertGreater(model.reference_weight, 0)
        self.assertLess(model.reference_weight, initial_weight)

    def test_finite_run_projection_matches_exhaustive_packet_draws(self):
        # Alternating between hit and miss changes the probabilities mid-triplet.
        table = [[.1, .6, .75], [.25, .3, .5], [.8, .1, .9]]
        expected = [0.0]*8
        for length in (1, 2):
            for trace in range(1 << (3*length)):
                p = 1.0
                last = None
                for bit in range(3*length):
                    heard = (trace >> bit) & 1
                    node = 0 if last is None else 2-last
                    loss = table[bit % 3][node]
                    p *= 1-loss if heard else loss
                    last = heard
                for i in range(length):
                    expected[(trace >> (3*i)) & 7] += p/3
        actual = calibration._expected_masks(table, calibration._triplet_paths(1), [1, 2])
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b)

    def test_shared_misses_independent_copies_and_session_reset(self):
        reference = [[0, 7]*8]
        model = calibration.TripletLossCalibrator(reference, 3)
        pilots = [0, 7]*4
        table = model.observe([iter(pilots)])
        self.assertEqual(model.reference_weight, 1)
        reference_table = calibration.fit_context(reference, 3)
        self.assertEqual(table, calibration.fit_context([pilots], 3, reference_table))

        model.reset_session()
        independent = list(range(8))
        self.assertEqual(model.observe([independent]), calibration.fit_context([independent], 3))
        self.assertEqual(model.reference_weight, 0)
        self.assertEqual(table, calibration.TripletLossCalibrator(reference, 3).observe([pilots]))

    def test_invalid_block_is_atomic_and_only_recent_pilots_fit_the_table(self):
        model = calibration.TripletLossCalibrator([range(8)], 2)
        model.observe([[0, 0]])
        with self.assertRaises(ValueError):
            model.observe([[7, 8]])
        with self.assertRaises(ValueError):
            model.observe([[]])
        for _ in range(8):
            table = model.observe([[5, 7]])
        self.assertEqual(model.reference_weight, 0)
        self.assertEqual(table, calibration.fit_context([[5, 7]]*8, 2))
