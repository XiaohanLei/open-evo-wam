import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml
from evo_wam.configuration import THRESHOLD, doctor, validate_profile
from evo_wam.adapters.cosmos_robotwin_train import select
from evo_wam.cli import main

ROOT = Path(__file__).resolve().parents[1]

def numerical_records(value, path=()):
    if isinstance(value, (bool, int, float)):
        return [[list(path), value]]
    if isinstance(value, dict):
        return [row for key in sorted(value) for row in numerical_records(value[key], (*path, key))]
    if isinstance(value, list):
        return [row for i, child in enumerate(value) for row in numerical_records(child, (*path, i))]
    return []

class PrivateProfileTests(unittest.TestCase):
    def config(self):
        return yaml.safe_load((ROOT / 'configs/cosmos3-robotwin-improve.yaml').read_text())

    def test_profile_rejects_algorithm_drift(self):
        config = self.config()
        validate_profile(config)
        for name, field, value in [('verify', 'threshold', THRESHOLD + .001), ('verify', 'interface', 'unsupported'), ('train', 'recipe', 'unsupported')]:
            changed = copy.deepcopy(config)
            changed['stages'][name][field] = value
            with self.assertRaises(ValueError): validate_profile(changed)

    def test_all_rounds_select_standalone_recipes(self):
        recipes = {i: str(ROOT / f'recipes/robotwin/r{i}.yaml') for i in range(1, 5)}
        for round_number in range(1, 5):
            context, kind = select({'round': round_number, 'spec': {'recipe': 'robotwin', 'recipe_configs': recipes, 'source_root': '/native/cosmos', 'python': '/python', 'master_addr': 'localhost', 'master_port': 29500}})
            self.assertEqual(kind, 'native')
            self.assertEqual(context['spec']['train_config'], recipes[round_number])
            self.assertEqual(context['spec']['source_root'], '/native/cosmos')
            self.assertIn('--nnodes=16', context['spec']['launcher'])

    def test_numerical_recipes_match_frozen_reference(self):
        expected = json.loads((ROOT / 'tests/fixtures/recipe-numerical-signatures.json').read_text())
        for round_number in range(1, 5):
            recipe = yaml.safe_load((ROOT / f'recipes/robotwin/r{round_number}.yaml').read_text())
            digest = hashlib.sha256(json.dumps(numerical_records(recipe), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            self.assertEqual(digest, expected[str(round_number)])
            self.assertEqual(recipe['trainer']['max_iter'], 1000)
            self.assertFalse(recipe['checkpoint']['load_training_state'])
            self.assertNotIn('/pfs/', json.dumps(recipe))

    def test_doctor_does_not_claim_native_execution(self):
        with patch.dict(os.environ, {}, clear=True):
            report = doctor(ROOT / 'configs/cosmos3-robotwin-improve.yaml')
        self.assertTrue(report['configuration_valid'])
        self.assertIn('UMT5_TOKENIZER', report['missing_environment_variables'])
        self.assertFalse(report['ready_for_native_run'])
        self.assertFalse(report['gpu_execution_verified'])

    def test_improve_defaults_to_all_four_rounds(self):
        with patch('evo_wam.cli.load_profile', return_value={}), patch('evo_wam.cli.improve', return_value=[]) as run:
            self.assertEqual(main(['improve', '--config', 'unused']), 0)
        self.assertEqual(run.call_args.args[1:3], (None, 4))
