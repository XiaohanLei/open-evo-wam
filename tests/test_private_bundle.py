import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

BUILDER = Path(__file__).resolve().parents[1] / 'scripts/build_private_bundle.py'


class PrivateBundleTests(unittest.TestCase):
    def prepare(self, root):
        (root / 'scripts').mkdir()
        script = root / 'scripts/build_private_bundle.py'
        script.write_bytes(BUILDER.read_bytes())
        (root / 'source.txt').write_text('neutral configuration\n')
        (root / '.private').mkdir()
        (root / '.private/untracked-secret.txt').write_text('fixture-only dummy value')
        for args in (['init', '-q'], ['add', 'source.txt', 'scripts/build_private_bundle.py'],
                     ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'Fixture']):
            subprocess.run(['git', *args], cwd=root, check=True, capture_output=True)
        spec = importlib.util.spec_from_file_location('fixture_builder', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_bundle_excludes_private_files_and_git_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = self.prepare(root)
            with patch.object(sys, 'argv', ['builder', str(root / '.private/bundle.tar.gz')]):
                module.main()
            with tarfile.open(root / '.private/bundle.tar.gz') as bundle:
                self.assertEqual(set(bundle.getnames()), {'source.txt', 'scripts/build_private_bundle.py', 'DEPLOYMENT.json'})
                manifest = json.load(bundle.extractfile('DEPLOYMENT.json'))
                self.assertTrue(manifest['private'])
                for member in bundle.getmembers():
                    self.assertEqual(member.mode, 0o600)
                    self.assertGreater(member.mtime, 0)

    def test_private_policy_blocks_identity_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = self.prepare(root)
            policy = root / '.private/policy.json'
            policy.write_text(json.dumps([{'category': 'fixture_identity', 'pattern': 'neutral configuration'}]))
            output = root / '.private/blocked.tar.gz'
            with patch.object(sys, 'argv', ['builder', str(output), '--privacy-rules', str(policy)]):
                with self.assertRaisesRegex(SystemExit, 'Privacy policy failed'):
                    module.main()
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
