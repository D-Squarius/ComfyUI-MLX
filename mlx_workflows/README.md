# MLX Core Node Workflows

These are API-prompt workflows for the local MLX backend spike. They use
default/core nodes only; no custom nodes are required.

## Active Manifests

The matching manifests have already been placed in `models/text_encoders`:

- `qwen2_5_0_5b_mlx_lm.mlx.json`
- `qwen2_vl_2b_mlx_vlm.mlx.json`
- `soprano_80m_mlx_audio_tts.mlx.json`
- `whisper_tiny_4bit_mlx_audio_asr.mlx.json`
- `minilm_mlx_embeddings.mlx.json`

## Fixture Inputs

The VLM and ASR inputs have already been copied into `input`:

- `mlx_vlm_red_square.png`
- `mlx_tts_for_asr.flac`

Copies are also kept under `mlx_workflows/fixtures`.

## API Workflows

- `api/01_lm_text_generate.json`
- `api/02_vlm_image_text_generate.json`
- `api/03_tts_save_audio.json`
- `api/04_asr_load_audio_text_generate.json`
- `api/05_embeddings_text_generate.json`

Submit them to `/prompt` or rebuild the same graph in the UI with `CLIPLoader`,
`TextGenerate`, `LoadImage`, `LoadAudio`, `SaveAudio`, and `PreviewAny`.

Start Comfy normally for manual testing:

```bash
source .venv-mlx-e2e/bin/activate
python main.py
```

Do not use `--cpu` for manual MLX validation. The API e2e harness has an
optional `--torch-cpu` flag only for isolating Torch-side Comfy behavior; MLX
uses its own Metal runtime.
