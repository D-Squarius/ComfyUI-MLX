# MLX LTX

ComfyUI can run LTX 2.3 through an optional MLX backend on Apple Silicon. The
backend is selected from the regular `CheckpointLoaderSimple` checkpoint
dropdown. Put the full dgrauet MLX model folder in `models/checkpoints`, then
select that folder in the workflow.

MLX dependencies are loaded lazily. ComfyUI still starts normally without MLX or
the LTX MLX packages installed until an MLX LTX model folder is selected.

## Requirements

- macOS on Apple Silicon
- the normal ComfyUI requirements from `requirements.txt`

## Models

The validated LTX 2.3 MLX model sources are:

- BF16: `dgrauet/ltx-2.3-mlx`
- Q8: `dgrauet/ltx-2.3-mlx-q8`
- Q4: `dgrauet/ltx-2.3-mlx-q4`

Example Q4 download:

```text
hf download dgrauet/ltx-2.3-mlx-q4 \
  --local-dir models/checkpoints/ltx-2.3-mlx-q4 \
  --include config.json split_model.json transformer-distilled-1.1.safetensors vae_decoder.safetensors vae_encoder.safetensors audio_vae.safetensors vocoder.safetensors connector.safetensors spatial_upscaler_x2_v1_1.safetensors spatial_upscaler_x2_v1_1_config.json
```

Q8:

```text
hf download dgrauet/ltx-2.3-mlx-q8 \
  --local-dir models/checkpoints/ltx-2.3-mlx-q8 \
  --include config.json split_model.json transformer-distilled-1.1.safetensors vae_decoder.safetensors vae_encoder.safetensors audio_vae.safetensors vocoder.safetensors connector.safetensors spatial_upscaler_x2_v1_1.safetensors spatial_upscaler_x2_v1_1_config.json
```

BF16:

```text
hf download dgrauet/ltx-2.3-mlx \
  --local-dir models/checkpoints/ltx-2.3-mlx \
  --include config.json split_model.json transformer-distilled-1.1.safetensors vae_decoder.safetensors vae_encoder.safetensors audio_vae.safetensors vocoder.safetensors connector.safetensors spatial_upscaler_x2_v1_1.safetensors spatial_upscaler_x2_v1_1_config.json
```

If you already downloaded an older folder, fetch the newer 1.1 transformer.
This is the Q4 repair command; use the matching repo and folder for Q8 or BF16:

```text
hf download dgrauet/ltx-2.3-mlx-q4 \
  --local-dir models/checkpoints/ltx-2.3-mlx-q4 \
  --include transformer-distilled-1.1.safetensors
```

Select `ltx-2.3-mlx-q4` in `CheckpointLoaderSimple`. The model is a folder with
multiple MLX files, not a single `.safetensors` checkpoint.

Advanced: `.mlx_ltx.json` checkpoint pointer files are still supported for
custom local paths or Hugging Face repo-pointer setups.


## Workflow

An example Q4 workflow is included at:

- `docs/workflows/ltx23_mlx_q4_workflow.json`

It uses the regular LTX graph:

`CheckpointLoaderSimple -> CLIPTextEncode -> ModelSamplingLTXV -> SamplerCustomAdvanced -> VAEDecode -> LTXVAudioVAEDecode -> CreateVideo -> SaveVideo`

Valid output should be an MP4 with visible frames and generated audio.
