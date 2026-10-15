# RoboTwin evaluation

The WAM generates the policy actions. The dense IDM scores consistency; it is
not the simulator controller. Evaluate an existing checkpoint with:

```bash
evo-wam doctor --config configs/cosmos3-robotwin-evaluate.yaml
evo-wam evaluate --config configs/cosmos3-robotwin-evaluate.yaml --round 1
```

Supply the checkpoint, native sources, model configuration/statistics, episode
registry, simulator assets and Vulkan configuration before executing the second
command. The example specifies a model GPU index and a render GPU environment
binding; inspect both for your machine.

## Reference environment

Use a separate Python 3.10 environment with Torch 2.4.1 + CUDA 12.1 and
torchvision 0.19.1. The component requirements are `requirements-robotwin.txt`
and `requirements-robotwin.lock.txt`. The recorded source revisions are:

| Component | Revision |
| --- | --- |
| RoboTwin | `958a6d2910a0262f5531fcdeb7fffae4184bb586` |
| CuRobo v0.7.6 | `2fbffc35225398cf9d5f382804faa9de2608753b` |

Follow the [RoboTwin installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)
and the [versioned CuRobo guide](https://nvlabs.github.io/curobo/v0.7.6/get_started/1_install_instructions.html).
Simulator assets are available separately from
[TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0).
Keep their license notices with the downloaded assets.

CuRobo requires a suitable CUDA toolkit and device architecture. The reference
environment pins `warp-lang==1.6.2` and `setuptools==75.8.0`; unconstrained upgrades
can break the selected simulator stack. Check the official RoboTwin guidance
for the MPLib screw-planning compatibility fix. The WAM HTTP client also requires
the public-action-API compatibility patch described in [native assets](native-assets.md).

## Episode protocol

Register the task, task configuration, instruction, fixed simulator seed,
inference noise seed and upstream action limit before a run. The wrapper runs
the upstream expert as a solvability preflight, resets the same seed and then
executes the WAM policy using rendered observations and feedback.

Use the upstream task-success condition for the policy outcome. Keep expert
planning failure, failed expert task conditions, invalid physics and model or
network infrastructure errors separate from a valid policy task failure. A
failed preflight does not justify replacing the seed. The batch guard rejects
unfinished or invalid cases instead of silently treating a partial batch as a
complete benchmark.

Retain the complete registry and all outcomes, including failures. VLM votes and
IDM threshold decisions do not define simulator task success. Limited diagnostics
have shown both successful policy episodes and failed or unstable preflights;
they do not prove simulator determinism or a general success rate.
