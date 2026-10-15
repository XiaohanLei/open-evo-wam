import unittest
from evo_wam.evaluation import summarize


class EvaluationTests(unittest.TestCase):
    def test_denominator(self):
        out = summarize(
            [dict(id=x) for x in "abc"],
            [
                dict(id=x, status=s)
                for x, s in zip("abc", ["success", "failure", "infrastructure_error"])
            ],
        )
        self.assertEqual(out["success_rate"], 1 / 3)
        self.assertFalse(out["complete_without_infrastructure_errors"])

    def test_missing_and_duplicate(self):
        for rows in ([], [dict(id="a", status="success")] * 2):
            with self.assertRaises(ValueError):
                summarize([dict(id="a")], rows)

    def test_frozen_seed_and_instruction_cannot_change(self):
        registry = [
            dict(
                id="a",
                fixed_seed=17,
                instruction="move red",
                episode_spec_sha256="pinned",
            )
        ]
        good = dict(registry[0], status="success")
        self.assertEqual(summarize(registry, [good])["success"], 1)
        for key, value in [
            ("fixed_seed", 18),
            ("instruction", "move blue"),
            ("episode_spec_sha256", "changed"),
        ]:
            with self.assertRaisesRegex(ValueError, "registry mismatch"):
                summarize(registry, [dict(good, **{key: value})])

    def test_typed_failures_and_split_denominators(self):
        registry = [
            dict(id="a", task_config="demo_clean"),
            dict(id="b", task_config="demo_clean"),
            dict(id="c", task_config="demo_randomized"),
        ]
        results = [
            dict(row, status=status)
            for row, status in zip(
                registry, ["success", "timeout", "model_task_failure"]
            )
        ]
        result = summarize(registry, results)
        self.assertEqual(result["valid_success"], 0.5)
        self.assertEqual(result["conservative_success"], 1 / 3)
        self.assertEqual(result["outcome_counts"]["timeout"], 1)
        self.assertEqual(
            result["by_task_config"]["demo_clean"]["conservative_success"], 0.5
        )
