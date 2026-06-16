# ComfyUI-MLX Native Workflow Guide

This fork adds optional Apple Silicon MLX island backends for selected native Comfy workflows. The intended user experience is:

1. Open the normal Comfy workflow template.
2. Keep the workflow topology unchanged.
3. Select an MLX alias or manifest in the same loader node where the stock model would normally be selected.
4. Run through the standard Comfy API or UI path.

There is no public MLX workflow node. The backend routes through ordinary loader/model objects and rejects unsupported features instead of silently falling back to Torch/MPS.

## Install

Base Comfy dependencies stay MLX-free. Install the Apple Silicon extras only when using MLX island routes:

```bash
pip install -e ".[mlx-all]"
```

The validated local runtime used `mlx==0.31.2`; the extra pins MLX to `mlx>=0.31.2,<0.32`. MLX imports are lazy, so a normal Comfy startup should not require MLX unless an MLX alias or manifest is selected.

## Supported Pipelines

| Pipeline | Workflow family | MLX paths | Status |
|---|---|---|---|
| FLUX.2 Klein 4B | Official Flux2 Klein T2I template | BF16 full weights, Q8, Q4 | BF16 1024x1024 native workflow validation passed; Q8/Q4 are benchmarked package paths |
| FLUX.2 Klein 9B | Official Flux2 9B T2I template | BF16 full weights, Q8, Q4 | BF16 1024x1024 native workflow validation passed; Q8/Q4 are benchmarked package paths |
| Z-Image Turbo | Official Z-Image Turbo T2I template | BF16 full weights, selected Q8 | BF16 1024x1024 native workflow validation passed |
| LTX 2.3 | Official/installed T2V, I2V, first-last-frame templates | BF16 dev+LoRA native path; distilled MLX package context path | 720p T2V BF16 passed; 512 I2V and first-last BF16 smokes passed |

## Model Sources And Placement

Keep dense Comfy weights in the normal Comfy model folders. Keep MLX packages or manifests in the MLX-specific locations already referenced by the backend. Do not commit model weights.

| Pipeline | Source | Expected local placement |
|---|---|---|
| Klein 4B BF16 | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | `models/diffusion_models/flux-2-klein-4b.safetensors` |
| Klein 9B BF16 | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | `models/diffusion_models/flux-2-klein-9b.safetensors` |
| Klein 4B MLX Q8 | [mlx-community/flux2-klein-4b-8bit](https://huggingface.co/mlx-community/flux2-klein-4b-8bit) | MLX package root used by the Klein manifest/alias |
| Klein 4B MLX Q4 | [mlx-community/flux2-klein-4b-4bit](https://huggingface.co/mlx-community/flux2-klein-4b-4bit) | MLX package root used by the Klein manifest/alias |
| Klein 9B MLX Q8 | [mlx-community/flux2-klein-9b-8bit](https://huggingface.co/mlx-community/flux2-klein-9b-8bit) | MLX package root used by the Klein manifest/alias |
| Klein 9B MLX Q4 | [mlx-community/flux2-klein-9b-4bit](https://huggingface.co/mlx-community/flux2-klein-9b-4bit) | MLX package root used by the Klein manifest/alias |
| Z-Image Turbo BF16 | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo), [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | `models/diffusion_models/z_image_turbo_bf16.safetensors` or equivalent local filename used by the alias |
| LTX 2.3 stock/reference | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) | Dense Comfy files under `models/checkpoints`, `models/loras`, text encoder, VAE, and upscaler folders |
| LTX 2.3 MLX native dev+LoRA BF16 | Stock [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) BF16 dev checkpoint and distilled LoRA, loaded through MLX | Experimental manifest such as `ltx23_dev_lora_mlx_bf16_native.mlx_ltx.json`; uses `pipeline: dev_lora`, `ltx-2.3-22b-dev.safetensors`, and LoRA strength `0.5`; does not require preconverted MLX dev weights |
| LTX 2.3 MLX dev BF16 package + LoRA | [dgrauet/ltx-2.3-mlx](https://huggingface.co/dgrauet/ltx-2.3-mlx) | Validated package path containing `transformer-dev.safetensors`, `transformer-distilled.safetensors`, and companion LTX assets; use with LoRA strength `0.5` for the dev-package route |
| LTX 2.3 MLX distilled BF16 context | [dgrauet/ltx-2.3-mlx](https://huggingface.co/dgrauet/ltx-2.3-mlx) | Distilled package folder containing `transformer-distilled.safetensors` or `transformer-distilled-1.1.safetensors`, VAE/audio/vocoder/connector files, and x2 upscaler files |
| LTX 2.3 MLX distilled Q8/Q4 context packages | [dgrauet/ltx-2.3-mlx-q8](https://huggingface.co/dgrauet/ltx-2.3-mlx-q8), [dgrauet/ltx-2.3-mlx-q4](https://huggingface.co/dgrauet/ltx-2.3-mlx-q4) | Optional quantized distilled package roots with the same folder layout as the BF16 distilled package |

## Native Workflow Provenance

Validation uses real Comfy API execution. Source workflows are read-only, and generated API prompts may only change loader aliases/manifests, prompt text, seed, resolution, frame/fps settings, input media filenames, and output prefixes.

The current review validation used official templates from `comfyui-workflow-templates==0.9.82` and recorded source hashes in:

`benchmarks/mlx_vs_baselines/results/clean_guarded_ltx_retest_20260615-095035/official_workflow_provenance.md`

Recorded provenance summary:

- Klein 4B, Klein 9B, and Z-Image Turbo templates matched current Comfy-Org raw workflow files.
- LTX 2.3 first-last-frame matched the current Comfy-Org raw workflow file.
- LTX 2.3 T2V and I2V used exact installed official package templates; those installed templates differed from current raw GitHub hashes and are documented as such.

## Validated Review Evidence

The current clean guarded review pass is summarized below. Raw benchmark harnesses and generated result folders are local development artifacts and are not part of the first public push plan.

| Row | Resolution | Scored time | Validation |
|---|---:|---:|---|
| Klein 4B BF16 MLX | 1024x1024 | 35.68s avg | Clean run, MLX route proof, valid image |
| Klein 9B BF16 MLX | 1024x1024 | 71.64s avg | Clean run, MLX route proof, valid image |
| Z-Image Turbo BF16 MLX | 1024x1024 | 83.02s avg | Clean run, MLX route proof, valid image |
| LTX 2.3 BF16 dev+LoRA T2V | 512x512 / 49f | 166.75s | Clean run, MLX route proof, valid MP4/audio |
| LTX 2.3 BF16 dev+LoRA I2V | 512x512 / 49f | 191.99s | Clean run, MLX route proof, valid MP4/audio |
| LTX 2.3 BF16 dev+LoRA First-Last | 512x512 / 49f | 281.09s | Clean run, MLX route proof, sparse guide attention, valid MP4/audio |
| LTX 2.3 BF16 dev+LoRA T2V | 1280x768 / 121f | 929.56s | Clean run, MLX route proof, valid MP4/audio |
| LTX 2.3 BF16 dev package + LoRA T2V | 1280x768 / 121f | 906.32s avg | Clean run, MLX route proof, valid MP4/audio |

The separate native T2V timing pass for LTX dev+LoRA is:

`benchmarks/mlx_vs_baselines/results/ltx23_dev_lora_mlx_native_t2v_timing_20260614-180654/`

It measured `ComfyUI-MLX LTX 2.3 BF16 dev+LoRA native T2V workflow` at `1280x768 / 121f / 24fps` with a scored average of `948.32s`. Historical distilled-package rows at `1024x768 / 120f` measured about `629-648s`; those are useful context but are not strict apples-to-apples with the native dev+LoRA 1280x768 workflow.

The validated `dgrauet/ltx-2.3-mlx` dev package route measured `906.32s` at `1280x768 / 121f` in `ltx23_mlx_bf16_dev_weight_compare_20260615-175827`, which was `49.13s` faster than the stock-dev safetensor MLX route at `955.45s` on the same native T2V workflow, a `+5.1%` improvement.

### Historical LTX Distilled Context

The committed distilled workflow examples are:

- `docs/workflows/ltx23_mlx_q4_native_island_t2v.json`
- `docs/workflows/ltx23_mlx_q4_native_island_i2v.json`

Those examples use the distilled MLX package layout. The Q8 and BF16 distilled packages use the same folder layout from their linked model repos.

For LTX there are two distinct MLX paths:

- Native dev+LoRA BF16: use the stock BF16 dev checkpoint `ltx-2.3-22b-dev.safetensors` and the local distilled LoRA at strength `0.5`; this runs through MLX but does not require preconverted MLX dev weights.
- MLX dev package + LoRA: use `transformer-dev.safetensors` from `dgrauet/ltx-2.3-mlx` with the same LoRA strength `0.5`.
- Distilled MLX package: use one of `dgrauet/ltx-2.3-mlx`, `dgrauet/ltx-2.3-mlx-q8`, or `dgrauet/ltx-2.3-mlx-q4`, where the package folder contains `transformer-distilled.safetensors` or `transformer-distilled-1.1.safetensors` and the companion media files.

In other words, `MLX native dev+LoRA BF16` means MLX execution with the stock BF16 dev safetensor loaded into MLX arrays. `MLX dev package + LoRA BF16` uses the package `transformer-dev.safetensors`. `MLX distilled BF16/Q8/Q4` means the distilled transformer comes directly from the MLX package folder.

Older local distilled-package benchmark artifacts are useful context for the faster LTX path, but they are not the same as native dev+LoRA BF16 validation.

| Context row | Settings | MLX BF16 | MLX Q8 | MLX Q4 | Notes |
|---|---:|---:|---:|---:|---|
| Distilled T2V package | 1024x768 / about 5s | 647.74s avg | 506.19s avg | 498.39s avg | Valid MP4/audio; Q8/Q4 were roughly 46-47% lower wall time than the current native dev+LoRA 720p row |
| Distilled quant final table | 512x512 / 49f context | 120.15s avg | 87.28s avg | 75.22s avg | Valid MP4/audio; compared locally against older split-file MPS context rows |

Keep these rows labeled as historical context. They support documenting the faster distilled LTX path, but they should not be mixed into strict native dev+LoRA BF16 apples-to-apples claims.

## Runtime Validation Contract

A passing MLX row must prove all of the following:

- Real Comfy API execution.
- Source workflow topology preserved.
- MLX island route events present.
- No Torch/MPS diffusion-transformer fallback.
- Image output is the expected resolution, nonblank, and tied to the prompt/seed.
- LTX output has the expected resolution, frame count, fps, nonstatic video, non-silent audio, and correct guide/input-frame behavior for I2V and first-last-frame workflows.

## Current Limitations

- ControlNet, model patches, unknown transformer wrappers, and unsupported reference-conditioning paths may reject rather than run through MLX.
- LTX 2.3 MPS BF16 did not produce a valid local baseline in the current environment; this fork does not publish invalid MPS rows as speed comparisons.
- LTX distilled MLX package results are faster in local historical context, but they are reported separately from native dev+LoRA BF16 workflow validation.
- Q8/Q4 paths are useful compatibility and performance rows, but full-weight BF16 claims should not mix in quantized timings.
