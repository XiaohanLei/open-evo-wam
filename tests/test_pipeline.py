import copy, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from evo_wam.io import read, write, identity
from evo_wam.pipeline import run_stage, improve, load_config


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.parent = self.root / "r0.bin"
        self.parent.write_bytes(b"initial")
        self.worker = self.root / "worker.py"
        self.worker.write_text("""import json,os,pathlib
c=json.load(open(os.environ['EVO_WAM_CONTEXT']));p=pathlib.Path(c['output']);a=p/'artifact.bin';a.write_text(c['parent_checkpoint']+c['stage']);r=dict(status='complete',artifacts=[str(a)])
if c['stage']=='train':r['checkpoint']=str(a)
(p/'result.json').write_text(json.dumps(r))
""")
        self.config = dict(
            schema="evo-wam.pipeline.v1",
            run_dir=str(self.root / "run"),
            initial_checkpoint=str(self.parent),
            stages={
                stage: dict(
                    executor="command",
                    argv=[sys.executable, str(self.worker)],
                    code_inputs=[str(self.worker)],
                )
                for stage in (
                    "generate",
                    "verify",
                    "export-prefix",
                    "train",
                    "evaluate",
                )
            },
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_nested_config_variables_resolve_before_stage(self):
        path = self.root / 'config.json'
        write(path, dict(schema='evo-wam.pipeline.v1',
                        variables={'PACKAGE_ROOT': '${CONFIG_DIR}/..'},
                        stages={'evaluate': {'episode_runner': '${PACKAGE_ROOT}/runner.py',
                                             'output': '${STAGE_DIR}/result.json'}}))
        config = load_config(path)
        self.assertEqual(config['stages']['evaluate']['episode_runner'], str(self.root)+'/../runner.py')
        self.assertEqual(config['stages']['evaluate']['output'], '${STAGE_DIR}/result.json')

    def test_two_rounds_reuse_and_parent(self):
        first = list(improve(self.config, rounds=2))
        self.assertEqual(len(first), 10)
        expected = str(self.root / "run/R01/train/artifact.bin")
        self.assertEqual(
            read(self.root / "run/R02/generate/context.json")["parent_checkpoint"],
            expected,
        )
        self.assertTrue(
            all(x["status"] == "reused" for x in improve(self.config, rounds=2))
        )

    def test_changed_inputs_reject(self):
        run_stage(self.config, "generate")
        self.parent.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            run_stage(self.config, "generate")

    def test_changed_output_reject(self):
        run_stage(self.config, "generate")
        (self.root / "run/R01/generate/artifact.bin").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "output changed"):
            run_stage(self.config, "generate")

    def test_failure_stops_then_resume(self):
        flag = self.root / "fail"
        flag.touch()
        s = self.worker.read_text()
        self.worker.write_text(
            "import pathlib,sys\nif pathlib.Path("
            + repr(str(flag))
            + ").exists():sys.exit(7)\n"
            + s
        )
        with self.assertRaises(subprocess.CalledProcessError):
            list(improve(self.config))
        self.assertFalse((self.root / "run/R01/verify").exists())
        self.assertEqual(
            read(self.root / "run/R01/generate/stage.json")["status"], "failed"
        )
        flag.unlink()
        self.assertEqual(len(list(improve(self.config))), 5)

    def test_dry_run_no_mutations(self):
        plans = list(improve(self.config, rounds=2, dry_run=True))
        self.assertEqual(len(plans), 10)
        self.assertFalse((self.root / "run").exists())
        self.assertTrue(plans[-1]["missing_dependencies"])

    def test_invalid_rounds(self):
        with self.assertRaises(ValueError):
            list(improve(self.config, rounds=0))

    def test_partial_resume(self):
        list(improve(self.config, end="export-prefix"))
        self.assertEqual(
            [x["stage"] for x in improve(self.config, round_number=1, start="train")],
            ["train", "evaluate"],
        )

    def test_independent_matches_aggregate(self):
        list(improve(self.config))
        self.assertEqual(run_stage(self.config, "verify")["status"], "reused")

    def test_evaluation_uses_current_round_checkpoint(self):
        list(improve(self.config))
        c = read(self.root / "run/R01/evaluate/context.json")
        self.assertEqual(
            c["current_checkpoint"], str(self.root / "run/R01/train/artifact.bin")
        )
        self.assertEqual(c["parent_checkpoint"], str(self.parent))

    def test_cumulative_pool_requires_previous_manifest(self):
        self.config["stages"]["export-prefix"]["cumulative"] = True
        list(improve(self.config))
        list(improve(self.config, round_number=2, end="verify"))
        with self.assertRaisesRegex(ValueError, "pool manifest"):
            run_stage(self.config, "export-prefix", 2)
        receipt = self.root / "run/R01/export-prefix/result.json"
        data = read(receipt)
        pool = self.root / "pool.json"
        write(pool, {"episodes": []})
        data["manifest"] = str(pool)
        write(receipt, data)
        # Simulate a producer that committed this manifest as an output.
        producer = receipt.parent / "stage.json"
        state = read(producer)
        state["outputs"][str(receipt)] = identity(receipt)
        state["outputs"][str(pool)] = identity(pool)
        write(producer, state)
        run_stage(self.config, "export-prefix", 2)
        c = read(self.root / "run/R02/export-prefix/context.json")
        self.assertEqual(c["spec"]["previous_pool"], str(pool))
        write(pool, {"episodes": [{"id": "changed"}]})
        with self.assertRaisesRegex(ValueError, "changed"):
            run_stage(self.config, "export-prefix", 2)

    def test_declared_payload_dependency_invalidates_reuse(self):
        payload = self.root / "native.npy"
        payload.write_bytes(b"first")
        self.worker.write_text(
            self.worker.read_text()
            + "r['dependency_paths']="
            + repr([str(payload)])
            + "\n(p/'result.json').write_text(json.dumps(r))\n"
        )
        run_stage(self.config, "generate")
        payload.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "output changed"):
            run_stage(self.config, "generate")

    def test_independent_export_round_two_needs_no_trained_parent(self):
        self.config["stages"]["export-prefix"]["requires_previous"] = False
        run_stage(self.config, "export-prefix", round_number=2)
        self.assertEqual(
            read(self.root / "run/R02/export-prefix/context.json")["parent_checkpoint"],
            "",
        )

    def test_explicit_round_parent(self):
        self.config["round_parent_checkpoints"] = {"2": str(self.parent)}
        run_stage(self.config, "generate", round_number=2)
        self.assertEqual(
            read(self.root / "run/R02/generate/context.json")["parent_checkpoint"],
            str(self.parent),
        )

    def test_standalone_evaluation_checkpoint(self):
        self.config["stages"]["evaluate"].update(
            requires_previous=False, checkpoint=str(self.parent)
        )
        run_stage(self.config, "evaluate")
        self.assertEqual(
            read(self.root / "run/R01/evaluate/context.json")["current_checkpoint"],
            str(self.parent),
        )

    def test_standalone_evaluation_later_round(self):
        self.config["stages"]["evaluate"].update(
            requires_previous=False, checkpoint=str(self.parent)
        )
        run_stage(self.config, "evaluate", round_number=4)

    def test_failed_producer_cannot_feed_next_stage(self):
        run_stage(self.config, "generate")
        receipt = self.root / "run/R01/generate/stage.json"
        state = read(receipt)
        state["status"] = "failed"
        write(receipt, state)
        with self.assertRaisesRegex(ValueError, "not complete"):
            run_stage(self.config, "verify")
        self.assertFalse((self.root / "run/R01/verify").exists())

    def test_changed_producer_cannot_feed_next_stage(self):
        run_stage(self.config, "generate")
        (self.root / "run/R01/generate/artifact.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            run_stage(self.config, "verify")

    def test_failed_train_cannot_become_round_parent(self):
        list(improve(self.config, end="train"))
        receipt = self.root / "run/R01/train/stage.json"
        state = read(receipt)
        state["status"] = "failed"
        write(receipt, state)
        with self.assertRaisesRegex(ValueError, "not complete"):
            run_stage(self.config, "generate", round_number=2)

    def test_retry_requires_fresh_result(self):
        marker = self.root / "already_attempted"
        self.worker.write_text(
            "import pathlib,sys\nm=pathlib.Path(" + repr(str(marker)) + ")\n"
            "if m.exists():sys.exit(0)\nm.touch()\n"
            + self.worker.read_text()
            + "\nsys.exit(7)\n"
        )
        with self.assertRaises(subprocess.CalledProcessError):
            run_stage(self.config, "generate")
        with self.assertRaises(FileNotFoundError):
            run_stage(self.config, "generate")
        directory = self.root / "run/R01/generate"
        self.assertEqual(read(directory / "stage.json")["status"], "failed")
        self.assertEqual(len(list(directory.glob("result.previous-*.json"))), 1)

    def test_selected_data_payload_invalidates_cache(self):
        payload = self.root / "rgb.bin"
        payload.write_bytes(b"original")
        manifest = self.root / "data.json"
        write(
            manifest, {"episodes": [{"id": "episode", "files": {"rgb": str(payload)}}]}
        )
        self.config["stages"]["prepare-data"] = dict(
            manifest=str(manifest), interface="test", split="train"
        )
        run_stage(self.config, "prepare-data")
        payload.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "output changed"):
            run_stage(self.config, "prepare-data")

    def test_training_bridge_can_consume_independent_manifest(self):
        trainer = self.root / "trainer.py"
        checkpoint = self.root / "trained.bin"
        trainer.write_text(
            "from pathlib import Path\nPath("
            + repr(str(checkpoint))
            + ").write_bytes(b'updated')\n"
        )
        self.config["stages"]["train"] = dict(
            requires_previous=False,
            adapter="evo_wam.adapters.training:run",
            source_root=str(self.root / "source"),
            checkpoint_output=str(checkpoint),
            train_argv=[sys.executable, str(trainer)],
            code_inputs=[str(trainer)],
        )
        (self.root / "source").mkdir()
        run_stage(self.config, "train")
        self.assertTrue(checkpoint.exists())

    def test_native_initial_and_asset_bindings_invalidate_cache(self):
        initial = self.root / "initial.npz"
        initial.write_bytes(b"one")
        jobs = self.root / "jobs.json"
        write(jobs, {"episodes": [{"scene": {"initial": str(initial)}}]})
        self.config["stages"]["generate"]["jobs"] = str(jobs)
        self.config["stages"]["generate"]["asset_paths"] = {
            "base_checkpoint": str(self.parent)
        }
        run_stage(self.config, "generate")
        initial.write_bytes(b"two")
        with self.assertRaisesRegex(ValueError, "changed"):
            run_stage(self.config, "generate")

    def test_idm_model_content_invalidates_independent_verification(self):
        checkpoint = self.root / "idm.bin"
        checkpoint.write_bytes(b"original")
        spec = self.config["stages"]["verify"]
        spec.update(requires_previous=False, idm_checkpoint=str(checkpoint))
        run_stage(self.config, "verify")
        checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            run_stage(self.config, "verify")
