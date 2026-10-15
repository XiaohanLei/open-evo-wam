import unittest
from evo_wam.data.robotwin_dense import training_universe


class NativeExportMetadataTests(unittest.TestCase):
    def test_sparse_pilot_preserves_full_native_task_universe(self):
        tasks=['place_empty_cup','place_bread_basket','stamp_seal','place_object_scale',
               'place_a2b_left','place_cans_plasticbox','stack_blocks_three']
        self.assertEqual(training_universe({'authorized_tasks':tasks},[{'task':tasks[0]}]),sorted(tasks))
        with self.assertRaisesRegex(ValueError,'seven-task'):
            training_universe({'authorized_tasks':[tasks[0]]},[{'task':tasks[0]}])
        with self.assertRaisesRegex(ValueError,'outside'):
            training_universe({'authorized_tasks':tasks},[{'task':'unknown_task'}])

class SceneIdentityTests(unittest.TestCase):
    def test_registry_binds_original_generation_and_cannot_change_scene(self):
        from pathlib import Path
        import tempfile
        from evo_wam.io import sha
        from evo_wam.data.robotwin_dense_online import training_scene_identity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'generation_manifest.json').write_text('{}')
            initial=root/'initial.npz';initial.write_bytes(b'fixture-observation')
            row=dict(id='candidate',case=str(root),initial_observation=str(initial))
            entry=dict(id='candidate',episode_key='scene0',scene_set='A',repeat_index=0,
                       generation_sha256=sha(root/'generation_manifest.json'),initial_observation_sha256=sha(initial))
            registry=dict(schema='evo-wam.export-scene-registry.v1',episodes=[entry])
            self.assertEqual(training_scene_identity(row,registry)['episode_key'],'scene0')
            with self.assertRaisesRegex(ValueError,'conflicts'):
                training_scene_identity(dict(row,episode_key='different'),registry)
            initial.write_bytes(b'changed-observation')
            with self.assertRaisesRegex(ValueError,'different generated'):
                training_scene_identity(row,registry)
