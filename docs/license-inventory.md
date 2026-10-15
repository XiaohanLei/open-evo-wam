# Source and asset license inventory

The maintainer's contributions are licensed under the
[PolyForm Noncommercial License 1.0.0](../LICENSE). Third-party code, patch
context and separately obtained assets retain their respective terms. Source
attribution and redistribution rights for historical materials still need to
be confirmed; choosing this license does not replace those upstream terms.

## Source dependencies

| Material | Available information | Outstanding work |
| --- | --- | --- |
| Orchestration, adapters, recipes and tests in this tree | PolyForm Noncommercial 1.0.0 applies to the maintainer's contributions; third-party material retains its original terms | Complete file-level source attribution, confirm historical source rights and preserve third-party notices |
| Native WAM source | Earlier inventory records OpenMDW-1.1 with component exceptions, including Apache-2.0 and GPL-3.0 material | Obtain the exact distributable revision and inspect its complete LICENSE/NOTICE/component files; the native tree is absent here |
| Native IDM source | Earlier inventory records Apache-2.0 and NVIDIA copyright notices | Confirm the exact revision and notices, including any local additions |
| Compatibility patches | Include context from Cosmos, DreamZero and RoboTwin framework files | Identify corresponding upstream revisions and preserve applicable notices |
| RoboTwin source | [Pinned revision's MIT license](https://github.com/RoboTwin-Platform/RoboTwin/blob/958a6d2910a0262f5531fcdeb7fffae4184bb586/LICENSE) | Preserve that revision's notice and review separately acquired assets |
| CuRobo v0.7.6 | [Versioned NVIDIA License](https://github.com/NVlabs/curobo/blob/v0.7.6/LICENSE), including a research/evaluation use restriction | Use this version's terms; do not substitute a newer release's license label |

The native WAM/IDM entries above describe prior inventory records. Their actual
source and license files are not included in this checkout, so they cannot be
certified from this tree alone. Do not label a combined source distribution
under one permissive license without resolving component terms.

## Separately acquired assets

The following publisher declarations were checked on 2026-10-15. Obtain the
actual license text and pin the revision of each asset used.

| Asset | Publisher declaration | Primary source |
| --- | --- | --- |
| Wan2.2 TI2V VAE source model | Apache-2.0 | [Wan model card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/blob/main/README.md) |
| UMT5 XXL tokenizer/model | Apache-2.0 | [UMT5 model card](https://huggingface.co/google/umt5-xxl/blob/main/README.md) |
| Qwen3-VL-8B-Instruct tokenizer/model | Apache-2.0 | [Qwen3-VL model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/README.md) |
| Optional Qwen2.5-VL-3B-Instruct API model | qwen-research | [Qwen2.5-VL model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/README.md) |
| RoboTwin2.0 simulator asset dataset | MIT declaration on the dataset card | [RoboTwin dataset card](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/blob/main/README.md) |
| Selected fine-tuned WAM/IDM checkpoints | Public redistribution rights not established | Separate checkpoint provenance and permissions are required |
| Complete historical/custom training data | Redistribution rights not established | Separate dataset provenance and permissions are required |

These assets are not bundled in the source tree. Publisher metadata does not
certify every underlying asset or extend to unrelated fine-tuned checkpoints or
custom datasets. A source-code license does not automatically license model
weights, datasets or simulator objects.

Preserve required upstream copyright and attribution notices. Naming an upstream
company as the rights holder is attribution, and should not be removed as a
privacy cleanup. See the root [NOTICE](../NOTICE.md) for this reproduction's
identity and current licensing status.
