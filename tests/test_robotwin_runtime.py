"""Vanilla simulator sources must not require a historical custom policy folder."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from evo_wam.adapters.robotwin_evaluation import stage_runtime, runtime_dependencies, execution_mode


class RobotWinRuntimeTests(unittest.TestCase):
    def test_replay_and_lossless_modes_are_explicit(self):
        self.assertEqual(execution_mode({}),'closed_loop')
        self.assertEqual(execution_mode({'dense_mode':'self_imagine','save_lossless_rgb':True}),'self_imagine')
        with self.assertRaises(ValueError):execution_mode({'dense_mode':'unknown'})
        with self.assertRaises(ValueError):execution_mode({'save_lossless_rgb':'false'})

    def test_vanilla_source_policy_import_and_evaluator_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source,target,assets = [root/name for name in ['source','runtime','assets']]
            (source/'script').mkdir(parents=True)
            original = source/'script/eval_policy.py'
            original.write_text('UPSTREAM_SOURCE = True\n')
            assets.mkdir()
            policy = root/'policy.py'
            policy.write_text('def get_model(*args): return "fixture"\ndef eval(*args): pass\ndef reset_model(*args): pass\n')
            runner = root/'runner.py'
            runner.write_text('FIXED_EPISODE_SOURCE = True\n')
            records = stage_runtime(source,target,assets,policy,runner)
            dependencies = runtime_dependencies(records, assets)
            self.assertEqual(set(dependencies), {str(assets),str(original),str(policy),str(runner)})
            self.assertNotIn(str(target/'policy/WAMDenseGlobalSinkHTTP/deploy_policy.yml'),dependencies)
            self.assertEqual((target/'script/_upstream_eval_policy.py').read_bytes(),original.read_bytes())
            self.assertEqual((target/'script/eval_policy.py').read_bytes(),runner.read_bytes())
            self.assertEqual((target/'assets').resolve(),assets.resolve())
            for record in records:
                self.assertEqual(hashlib.sha256((target/record['path']).read_bytes()).hexdigest(),record['sha256'])
            env = dict(os.environ,PYTHONPATH=str(target/'policy'))
            subprocess.run([sys.executable,'-c',
                'import WAMDenseGlobalSinkHTTP as p; assert p.get_model({}) == "fixture"; p.reset_model(None)'],
                env=env,cwd=root,check=True,capture_output=True)


if __name__ == '__main__':
    unittest.main()
