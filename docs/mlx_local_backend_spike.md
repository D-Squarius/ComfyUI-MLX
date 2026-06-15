# Local MLX Backend Spike Notes

This checkout contains a local-only Apple MLX backend experiment. It is not an
upstream PR plan and does not add custom/user-facing MLX nodes.

## Core Node Strategy

- `CLIPLoader` accepts MLX loader types and still returns `CLIP`.
- `TextGenerate` remains the single generation node for LLM, VLM, ASR, TTS,
  embedding, and reranking experiments.
- MLX models are selected by `.mlx` or `.mlx.json` manifests placed in
  `models/text_encoders`.
- Internally, MLX objects are runtime adapters, not real diffusion CLIP models.
  They only support the `TextGenerate` protocol and fail clearly in conditioning
  paths.
- TTS uses `TextGenerate` output 1 as optional `AUDIO`, so existing
  `PreviewAudio` and `SaveAudio` nodes can consume it.
- ASR uses existing `LoadAudio` into `TextGenerate(audio=...)` and returns text.

## Supported Loader Types

- `mlx_lm`: `CLIPLoader -> TextGenerate -> STRING`
- `mlx_vlm`: `CLIPLoader -> TextGenerate(image/video) -> STRING`
- `mlx_audio_asr`: `LoadAudio -> TextGenerate(audio) -> STRING`
- `mlx_audio_tts`: `CLIPLoader -> TextGenerate -> STRING + AUDIO`
- `mlx_embeddings`: `CLIPLoader -> TextGenerate(prompt JSON/text, image optional) -> JSON STRING`

## Manifest Shape

Example:

```json
{
  "backend": "mlx_lm",
  "repo_id": "mlx-community/Qwen3-0.6B-4bit",
  "revision": "main",
  "local_files_only": false,
  "use_hf_token": false,
  "allow_patterns": ["*.json", "*.safetensors", "*.py"],
  "ignore_patterns": ["*.md"],
  "generation": {
    "max_kv_size": 2048
  }
}
```

Do not put token values in manifests. Use Hugging Face CLI login or environment
configuration and set `use_hf_token` when authenticated downloads are needed.
Use either `repo_id`/`model` for Hugging Face snapshots or `local_path` for a
local checkout, not both.

## Optional Dependencies

Base ComfyUI imports must work without MLX packages installed. Install only the
extras needed for a test:

- `pip install -e .[mlx-llm]`
- `pip install -e .[mlx-vlm]`
- `pip install -e .[mlx-audio]`
- `pip install -e .[mlx-embeddings]`
- `pip install -e .[mlx-all]`

For LLM benchmarking, use `mlx>=0.31.2` / `mlx-lm>=0.31.3` or newer. Local
testing on an M1 Max showed the older `mlx==0.17.3` / `mlx-lm==0.18.2` stack
was slower than llama.cpp GGUF Q4_K_M on Qwen2.5 0.5B 4-bit, while the same
MLX model on `mlx==0.31.2` / `mlx-lm==0.31.3` was substantially faster in a
direct backend benchmark. The adapter keeps compatibility with both old
`temp`/`top_p` generation APIs and newer sampler-based `mlx-lm` APIs.

## Current Validation

- Syntax: `python -m py_compile nodes.py comfy/text_encoders/mlx.py comfy/backends/*.py comfy_extras/nodes_textgen.py`
- Focused tests: `python -m pytest tests-unit/comfy_test/mlx_backend_test.py tests-unit/comfy_test/folder_path_test.py -q`
- Full unit tests: `python -m pytest tests-unit -q`
- API e2e: `python tests/e2e/run_mlx_api_e2e.py`

The API e2e harness starts ComfyUI with `--disable-all-custom-nodes` and uses
only default/core nodes:

- LLM: `CLIPLoader -> TextGenerate -> PreviewAny`
- VLM: `LoadImage -> TextGenerate(image) -> PreviewAny`
- TTS: `TextGenerate -> SaveAudio`
- ASR: `LoadAudio -> TextGenerate(audio) -> PreviewAny`
- Embeddings: `TextGenerate -> PreviewAny`

Validated local model set:

- `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
- `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- `mlx-community/Qwen2.5-3B-Instruct-4bit`
- `mlx-community/Qwen2.5-7B-Instruct-4bit`
- `mlx-community/Llama-3.2-3B-Instruct-4bit`
- `mlx-community/gemma-2-2b-it-4bit`
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`
- `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`
- `mlx-community/gemma-3-4b-it-4bit`
- `mlx-community/Soprano-1.1-80M-bf16`
- `mlx-community/pocket-tts-4bit`
- `mlx-community/whisper-tiny-asr-4bit`
- `mlx-community/whisper-base-asr-4bit`
- `mlx-community/all-MiniLM-L6-v2-4bit`

## Open Local Risks

- Adding a second `AUDIO` output to `TextGenerate` must be verified against
  existing workflows and frontend output indexing.
- Embedding/reranking JSON output can get huge; local adapters cap JSON output
  via `max_output_json_bytes`.
- `mlx-vlm`, `mlx-audio`, and `mlx-embeddings` APIs move quickly; manifests
  should carry task-specific generation settings instead of hard-coding every
  option in core.
- Some MLX model repos can be incompatible with current package loaders even
  when they download successfully. For example,
  `mlx-community/llava-interleave-qwen-0.5b-4bit` currently fails local
  `mlx_vlm.load` with extra projector quantization parameters.
- Some `mlx-audio` TTS models need non-Python system audio/text dependencies.
  `mlx-community/kitten-tts-nano-0.8-4bit` loaded locally, but generation
  requires `phonemizer-fork` plus a system `espeak` binary.
- MLX generation is serialized by default until local stress testing proves safe
  concurrent execution on Metal.
- Large model unload and cache behavior needs repeated load/generate/unload
  memory measurements on Apple Silicon.
