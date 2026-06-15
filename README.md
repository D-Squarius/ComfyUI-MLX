# ComfyUI-MLX

Apple Silicon MLX testing fork for running selected native ComfyUI workflow templates through MLX island backends.

This is not an upstream ComfyUI release. It is a fork focused on proving MLX-backed model execution inside normal Comfy workflows on Apple Silicon.

## What This Fork Adds

The core goal is simple: use the normal Comfy workflow template, then choose an MLX alias or manifest in the ordinary loader node. No public MLX-only workflow node is required.

Validated workflow families:

| Pipeline | Native workflow usage | MLX precision paths | Current validation |
|---|---|---|---|
| FLUX.2 Klein 4B | Official Flux2 Klein T2I template | BF16 full weights, Q8, Q4 | 1024x1024 BF16 Comfy API validation passed |
| FLUX.2 Klein 9B | Official Flux2 9B T2I template | BF16 full weights, Q8, Q4 | 1024x1024 BF16 Comfy API validation passed |
| Z-Image Turbo | Official Z-Image Turbo T2I template | BF16 full weights, selected Q8 | 1024x1024 BF16 Comfy API validation passed |
| LTX 2.3 | Official or installed T2V, I2V, and first-last-frame templates | BF16 dev+LoRA native path; distilled MLX package context path | 720p native T2V BF16 passed; 512 I2V and first-last smokes passed |

MLX route proof is part of validation: successful rows must show MLX island route events and no Torch/MPS diffusion-transformer fallback.

## Install

Install base Comfy dependencies as usual. The base `requirements.txt` intentionally stays MLX-free.

Install the optional Apple Silicon MLX extras only when using MLX routes:

```bash
pip install -e ".[mlx-all]"
```

The validated runtime uses `mlx==0.31.2`; the optional extras pin the compatible `0.31.x` line with `mlx>=0.31.2,<0.32`.

MLX imports are lazy, so normal Comfy startup should not require MLX unless an MLX alias or manifest is selected.

## How To Use The Native Workflows

1. Open the normal Comfy workflow template.
2. Keep the workflow topology unchanged.
3. In the normal loader fields, select the MLX alias or manifest for the model component.
4. Run through the normal Comfy UI or Comfy API route.

Generated API prompts used in validation only changed loader aliases/manifests, prompt text, seed, resolution, frame/fps settings, input media filenames, and output prefixes.

## Model Sources

Keep dense Comfy weights in the normal Comfy model folders. Keep MLX packages or manifests in the MLX-specific locations referenced by the backend. Do not commit model weights.

| Pipeline | Source | Expected local placement |
|---|---|---|
| Klein 4B BF16 | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | `models/diffusion_models/flux-2-klein-4b.safetensors` |
| Klein 9B BF16 | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | `models/diffusion_models/flux-2-klein-9b.safetensors` |
| Klein 4B MLX Q8 | [mlx-community/flux2-klein-4b-8bit](https://huggingface.co/mlx-community/flux2-klein-4b-8bit) | MLX package root used by the Klein alias |
| Klein 4B MLX Q4 | [mlx-community/flux2-klein-4b-4bit](https://huggingface.co/mlx-community/flux2-klein-4b-4bit) | MLX package root used by the Klein alias |
| Klein 9B MLX Q8 | [mlx-community/flux2-klein-9b-8bit](https://huggingface.co/mlx-community/flux2-klein-9b-8bit) | MLX package root used by the Klein alias |
| Klein 9B MLX Q4 | [mlx-community/flux2-klein-9b-4bit](https://huggingface.co/mlx-community/flux2-klein-9b-4bit) | MLX package root used by the Klein alias |
| Z-Image Turbo BF16 | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo), [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | `models/diffusion_models/z_image_turbo_bf16.safetensors` or equivalent alias target |
| LTX 2.3 stock/reference | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) | Dense Comfy files under `models/checkpoints`, `models/loras`, text encoder, VAE, and upscaler folders |
| LTX 2.3 MLX native dev+LoRA BF16 | [dgrauet/ltx-2.3-mlx](https://huggingface.co/dgrauet/ltx-2.3-mlx) plus the stock [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) BF16 dev checkpoint and distilled LoRA | Manifest such as `ltx23_dev_lora_mlx_bf16_native.mlx_ltx.json`; uses `pipeline: dev_lora`, `ltx-2.3-22b-dev.safetensors`, and LoRA strength `0.5` |
| LTX 2.3 MLX distilled BF16 context | [dgrauet/ltx-2.3-mlx](https://huggingface.co/dgrauet/ltx-2.3-mlx) | Distilled package folder containing `transformer-distilled-1.1.safetensors`, VAE/audio/vocoder/connector files, and x2 upscaler files |
| LTX 2.3 MLX distilled Q8/Q4 context | [dgrauet/ltx-2.3-mlx-q8](https://huggingface.co/dgrauet/ltx-2.3-mlx-q8), [dgrauet/ltx-2.3-mlx-q4](https://huggingface.co/dgrauet/ltx-2.3-mlx-q4) | Optional quantized distilled package roots with the same folder layout as the BF16 distilled package |

More detail is in [docs/mlx_native_workflows.md](docs/mlx_native_workflows.md).

## Current Benchmark Evidence

Current local validation summary:

| Row | Resolution | Scored time | Validation |
|---|---:|---:|---|
| Klein 4B BF16 MLX | 1024x1024 | 35.68s avg | Valid image, MLX route proof |
| Klein 9B BF16 MLX | 1024x1024 | 71.64s avg | Valid image, MLX route proof |
| Z-Image Turbo BF16 MLX | 1024x1024 | 83.02s avg | Valid image, MLX route proof |
| LTX 2.3 BF16 dev+LoRA T2V | 512x512 / 49f | 166.75s | Valid MP4/audio, MLX route proof |
| LTX 2.3 BF16 dev+LoRA I2V | 512x512 / 49f | 191.99s | Valid MP4/audio, MLX route proof |
| LTX 2.3 BF16 dev+LoRA First-Last | 512x512 / 49f | 281.09s | Valid MP4/audio, sparse guide attention |
| LTX 2.3 BF16 dev+LoRA T2V | 1280x768 / 121f | 929.56s | Valid MP4/audio, MLX route proof |

Historical LTX distilled-package context rows:

| Row | Resolution | Scored time | Validation |
|---|---:|---:|---|
| LTX 2.3 distilled MLX BF16 context | 1024x768 / about 5s | 647.74s avg | Valid MP4/audio; context only |
| LTX 2.3 distilled MLX Q8 context | 1024x768 / about 5s | 506.19s avg | Valid MP4/audio; context only |
| LTX 2.3 distilled MLX Q4 context | 1024x768 / about 5s | 498.39s avg | Valid MP4/audio; context only |

The historical distilled Q8/Q4 context rows were roughly 46-47% lower wall time than the current `1280x768 / 121f` native dev+LoRA BF16 T2V row. They are reported separately because model path, resolution, and workflow path differ. Distilled workflow examples are committed in `docs/workflows/ltx23_mlx_q4_native_island_t2v.json` and `docs/workflows/ltx23_mlx_q4_native_island_i2v.json`; Q8 and BF16 distilled packages use the same folder layout from their linked model repos. Raw benchmark harnesses and generated result folders remain local development artifacts.

## Limitations

- These are local Apple Silicon MLX routes, not upstream ComfyUI defaults.
- Unsupported ControlNet/model-patch paths, unknown transformer wrappers, and unsupported reference-conditioning features should reject clearly instead of silently falling back.
- LTX 2.3 MPS BF16 did not produce a valid local baseline in the current environment, so this fork does not publish invalid MPS rows as speed comparisons.
- Q8/Q4 paths are useful compatibility and performance rows, but full-weight BF16 claims should not mix in quantized timings.

## Upstream

This fork is based on ComfyUI. For upstream documentation, installation guides, community links, and general Comfy usage, see the official project:

- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI website](https://www.comfy.org/)
- [Comfy workflow templates](https://github.com/Comfy-Org/workflow_templates)
