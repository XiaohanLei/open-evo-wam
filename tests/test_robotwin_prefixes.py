import unittest
import numpy as np
from evo_wam.verification.robotwin import score_prefixes, window_seed


class PrefixTests(unittest.TestCase):
    def check_interface(self, interface, ends, size, stride):
        n = max(ends)
        calls = []
        stats = {"action": {"std": [2.0] * 14}}

        def predict(**kw):
            length = len(kw["video"]) - 1 if stride == 1 else kw["action_length"]
            calls.append(
                (int(kw["frame_indices"][0]), length, kw["seed"], kw["video"].copy())
            )
            return np.full((length, 14), length, np.float32)

        video = np.arange(n // stride + 1)[:, None]
        actions = np.zeros((n, 14))
        states = np.zeros(((n + size - 1) // size, 14))
        result = score_prefixes(
            "candidate",
            ends,
            video,
            actions,
            states,
            "task",
            predict,
            stats,
            interface,
            "native",
        )
        for end in ends:
            expected = (
                sum(min(size, end - a) ** 3 / 4 for a in range(0, end, size)) / end
            )
            self.assertAlmostEqual(result["prefixes"][str(end)]["mse"], expected)
            self.assertEqual(result["prefixes"][str(end)]["scored_actions"], end)
        unique = {(a, min(size, e - a)) for e in ends for a in range(0, e, size)}
        self.assertEqual({c[:2] for c in calls}, unique)
        self.assertEqual(len(calls), len(unique))
        for start, length, seed, frames in calls:
            self.assertEqual(frames[-1, 0], (start + length) // stride)
            self.assertEqual(
                seed, window_seed("candidate", start, length, interface, "native")
            )

    def test_dense_all_boundaries(self):
        self.check_interface("dense", [1, 3, 4, 5, 63, 64, 65, 127, 128, 150], 64, 1)

    def test_unsupported_interface_rejected(self):
        with self.assertRaises(ValueError):
            window_seed("candidate", 0, 1, "unsupported", "native")
        with self.assertRaises(ValueError):
            self.check_interface("unsupported", [1], 64, 1)

    def test_candidate_does_not_enter_predictor(self):
        calls = []

        def predict(**kw):
            calls.append(kw)
            return np.zeros((5, 14))

        args = (
            "id",
            [5],
            np.zeros((6, 1)),
            np.zeros((5, 14)),
            np.zeros((1, 14)),
            "task",
            predict,
            {"action": {"std": [1] * 14}},
            "dense",
            "ns",
        )
        a = score_prefixes(*args)
        args = list(args)
        args[3] = np.ones((5, 14))
        b = score_prefixes(*args)
        self.assertEqual(a["prefixes"]["5"]["mse"], 0)
        self.assertEqual(b["prefixes"]["5"]["mse"], 1)
        for key in calls[0]:
            if isinstance(calls[0][key], np.ndarray):
                np.testing.assert_array_equal(calls[0][key], calls[1][key])
            else:
                self.assertEqual(calls[0][key], calls[1][key])

    def test_nonfinite_fails(self):
        with self.assertRaises(ValueError):
            score_prefixes(
                "id",
                [1],
                np.zeros((2, 1)),
                np.zeros((1, 14)),
                np.zeros((1, 14)),
                "x",
                lambda **kw: np.full((1, 14), np.nan),
                {"action": {"std": [1] * 14}},
                "dense",
                "ns",
            )
