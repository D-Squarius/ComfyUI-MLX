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
huggingface-cli download dgrauet/ltx-2.3-mlx-q4 --local-dir models/checkpoints/ltx-2.3-mlx-q4
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
