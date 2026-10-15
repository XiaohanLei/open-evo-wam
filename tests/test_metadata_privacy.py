import pickle
import pickletools
import unittest
from evo_wam.metadata_privacy import rewrite_pickle_strings


class MetadataPrivacyTests(unittest.TestCase):
    def test_multiframe_rewrite_preserves_tensor_descriptions_and_memos(self):
        source_path = '/private/account/checkpoint'
        original = {'checkpoint_id': source_path, 'storage': {'path': source_path},
                    'tensors': [(f'layer.{i}.weight', (64, i + 1), 'BF16', i * 128) for i in range(12000)]}
        raw = pickle.dumps(original, protocol=4)
        self.assertGreater(sum(op.name == 'FRAME' for op, _, _ in pickletools.genops(raw)), 1)
        updated = rewrite_pickle_strings(raw, {source_path: 'assets/checkpoint'})
        restored = pickle.loads(updated)  # A locally generated, trusted fixture only.
        self.assertEqual(restored['tensors'], original['tensors'])
        self.assertEqual(restored['checkpoint_id'], 'assets/checkpoint')
        self.assertIs(restored['checkpoint_id'], restored['storage']['path'])
        self.assertNotIn(source_path.encode(), updated)

    def test_unchanged_metadata_is_byte_identical(self):
        raw = pickle.dumps({'file': 'chunk.distcp', 'shape': (16, 32), 'dtype': 'BF16'}, protocol=4)
        self.assertEqual(rewrite_pickle_strings(raw, {}), raw)


if __name__ == '__main__':
    unittest.main()
