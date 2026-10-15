# VLM API setup

The verifier sends image-based description and judgment requests to a
chat-completions-compatible API. It parses strict structured responses and
applies fixed-endpoint voting. The local and external examples are verify-stage
configuration fragments, rather than standalone pipeline configurations:

- `configs/vlm-api-local.example.yaml`
- `configs/vlm-api-external.example.yaml`

Set an explicit model revision so cached requests remain tied to the intended
model. Standard payloads omit provider-specific thinking fields. Only configure
`thinking_token_budget` for a provider that supports it. `max_tokens` overrides
the output budget; the local small-model example uses 2048.

## Local Qwen service

Obtain `Qwen/Qwen2.5-VL-3B-Instruct` under its own model terms. In one terminal,
start the server with a compatible WAM Python environment and an available GPU:

```bash
python scripts/serve_qwen_vlm.py --model-path "$VLM_MODEL_PATH" --port 18881
```

Keep that process running. In the terminal that launches the pipeline, set:

```bash
export VLM_ENDPOINT=http://127.0.0.1:18881/v1/chat/completions
export VLM_MODEL=qwen-small
export VLM_MODEL_REVISION=Qwen2.5-VL-3B-Instruct-local
```

The server binds to loopback and supports non-streaming requests with embedded
images. Set the verify stage's `format_instructions` to
`configs/vlm-qwen-small-format.json` for the small model. These reminders specify
the output shape while retaining the task rubric and strict evidence checks.

This service is useful for integration diagnostics. Its deterministic repeated
votes are correlated and do not establish verifier accuracy or equivalence to
the original research's VLM.

## External service

Set `VLM_ENDPOINT` to the provider's complete chat-completions route,
`VLM_MODEL` to the model name and `VLM_MODEL_REVISION` to the selected version.
Configure `api_key_env: EVO_VLM_API_KEY` in the verify stage and supply the key
through that environment variable. The client reads it for the bearer header;
do not put a key or secret query parameter in the endpoint URL.

The provider must support image messages and the requested output budget.
External-provider execution is not part of the recorded local integration
evidence. Runtime request/response records can contain images and experiment
content; inspect them before sharing or committing generated outputs.
