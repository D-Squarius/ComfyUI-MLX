# ComfyUI-MLX Fork Review Package

Status: prepared for review only. No commit, push, or PR has been made.

## Dependency Notes

- Base `requirements.txt` remains MLX-free.
- MLX is still required for MLX island routes and remains optional in `pyproject.toml`.
- The validated local runtime used `mlx==0.31.2`; optional MLX extras are pinned to `mlx>=0.31.2,<0.32`.
- LTX MLX support also declares optional `ltx-core-mlx>=0.14.8,<0.15` and `ltx-pipelines-mlx>=0.14.8,<0.15`.
- MLX imports should stay lazy: normal Comfy startup should not require MLX unless an MLX alias or manifest is selected.

## Native Workflow Smoke Results

All smokes used real Comfy API execution and preserved source workflow topology. Generated prompts changed only loader aliases, prompt, seed, size/frame settings, input media, and output prefix.

Artifact root, excluded from push: `benchmarks/mlx_vs_baselines/results/fork_review_native_smoke_20260614-204030/`

| Pipeline | Native workflow path | MLX row | Time | Validation | Output |
|---|---|---|---:|---|---|
| Klein 4B BF16 | Flux2 Klein text-to-image template | `klein4_t2i_mlx_512x512` | 37.68s | MLX route events present; image 512x512 nonblank | `output/native_bf16/klein4_t2i_mlx_512x512/00_ceramic_product_00005_.png` |
| Klein 9B BF16 | Flux2 Klein 9B text-to-image template | `klein9_t2i_mlx_512x512` | 74.52s | MLX route events present; image 512x512 nonblank | `output/native_bf16/klein9_t2i_mlx_512x512/00_ceramic_product_00002_.png` |
| Z-Image Turbo BF16 | Z-Image Turbo text-to-image template | `zimage_t2i_mlx_512x512` | 37.17s | MLX route events present; image 512x512 nonblank | `output/native_bf16/zimage_t2i_mlx_512x512/00_red_sports_car_00005_.png` |
| LTX 2.3 BF16 dev+LoRA | `blueprints/Text to Video (LTX-2.3).json` | `ltx23_t2v_mlx_512x512` | 235.42s | MLX route events present; MP4 512x512, 49f, 24fps, audio present/not silent | `output/native_bf16/ltx23_t2v_mlx_512x512/00_coastal_rescue_00008_.mp4` |

## Static/Test Results

- `py_compile` passed for targeted MLX backend/text encoder/benchmark files.
- Targeted pytest passed: `168 passed, 2 warnings`.
- `git diff --check` passed.

Targeted pytest command:

```bash
.venv-mlx-e2e/bin/python -m pytest -q \
  tests-unit/comfy_test/mlx_backend_test.py \
  tests-unit/comfy_test/mlx_klein_backend_test.py \
  tests-unit/comfy_test/mlx_ltx_backend_test.py \
  tests-unit/comfy_test/mlx_denoiser_island_test.py \
  tests-unit/comfy_test/native_bf16_workflow_matrix_test.py
```

## Proposed File Groups For Review

### Core backend changes

- `comfy/backends/`
- `comfy/text_encoders/mlx.py`
- `comfy/text_encoders/acceleration.py`
- `comfy/text_encoders/baseline.py`
- MLX integration touchpoints in `comfy/sd.py`, `comfy/samplers.py`, `nodes.py`, `folder_paths.py`, `execution.py`, and `comfy/model_management.py`.

### Native workflow/node integration

- LTX routing and media compatibility changes in `comfy/ldm/lightricks/`, `comfy_extras/nodes_lt*.py`, and `comfy_extras/nodes_video.py`.
- Klein/Z-Image native loader interception and route proof paths through the shared MLX island backend.
- `comfy_api/latest/_input_impl/video_types.py` and related test updates for video validation support.

### Tests

- MLX backend tests under `tests-unit/comfy_test/mlx*_test.py`.
- Native workflow and benchmark tests under `tests-unit/comfy_test/native_bf16_workflow_matrix_test.py` and related benchmark test files.
- Existing touched tests in `tests-unit/comfy_api_test/` and `tests-unit/comfy_test/folder_path_test.py`.

### Docs

- `README.md`
- `docs/mlx_native_workflows.md`
- `docs/mlx_local_backend_spike.md`
- `docs/plans/june13plan.md`

### Benchmark/report tooling

- `benchmarks/mlx_vs_baselines/` scripts, manifests, and small workflow fixtures.
- `mlx_workflows/` API examples and tiny fixtures.
- `tests/e2e/` MLX API smoke harness.

## Intentionally Excluded From Push

The updated `.gitignore` keeps these local-only:

- `.venv*/`
- `models/`
- `output/`
- `input/` except `input/example.png`
- `benchmarks/mlx_vs_baselines/results/`
- `benchmarks/mlx_vs_baselines/custom_mlx_wheel_venvs/`
- `benchmarks/mlx_vs_baselines/native/build/`
- generated benchmark `.mp4`, `.mov`, `.trace`, `.xcresult`, and `.sqlite3` files
- `Vendor/`

## Current Git Review Snapshot

- Current branch: `codex/local-mlx-backend-spike`
- Fork remote: `https://github.com/D-Squarius/ComfyUI-MLX.git`
- No publish action has been taken.
- Current tracked diff stat before staging: 24 tracked files changed, 4079 insertions, 55 deletions.
- Additional untracked shippable files exist under `benchmarks/`, `comfy/backends/`, `docs/`, `mlx_workflows/`, and `tests-unit/comfy_test/`.

## 1024/720p Official Native Workflow Addendum

Artifact root:
`benchmarks/mlx_vs_baselines/results/fork_review_native_1024_720p_20260614-220732`

This pass used official Comfy workflow templates from installed `comfyui-workflow-templates==0.9.82` plus raw GitHub provenance checks. Klein 4B, Klein 9B, and Z-Image templates matched current Comfy-Org raw workflow files. LTX first-last matched current raw; LTX T2V/I2V were exact installed official package templates but differed from current raw GitHub, which is recorded in `official_workflow_provenance.md`.

Generated API graphs changed only loader aliases/manifests, prompt/seed, dimensions/frame settings, input media filenames, and output prefixes. No workflow topology was simplified.

| Pipeline | Official workflow | Row | Resolution | Scored avg | Status | Output |
|---|---|---|---|---:|---|---|
| Klein 4B BF16 | `image_flux2_klein_text_to_image.json` | `klein4_t2i_mlx_1024x1024` | `1024x1024` | 43.17s | pass, MLX route proof | `output/native_bf16/klein4_t2i_mlx_1024x1024/00_ceramic_product_00002_.png` |
| Klein 9B BF16 | `image_flux2_text_to_image_9b.json` | `klein9_t2i_mlx_1024x1024` | `1024x1024` | 85.81s | pass, MLX route proof | `output/native_bf16/klein9_t2i_mlx_1024x1024/00_ceramic_product_00002_.png` |
| Z-Image Turbo BF16 | `image_z_image_turbo.json` | `zimage_t2i_mlx_1024x1024` | `1024x1024` | 100.54s | pass, MLX route proof | `output/native_bf16/zimage_t2i_mlx_1024x1024/00_red_sports_car_00002_.png` |
| LTX 2.3 BF16 dev+LoRA T2V | `video_ltx2_3_t2v.json` | `ltx23_t2v_mlx_1280x768` | `1280x768 / 121f / 24fps` | 928.39s | pass, MLX route proof, valid MP4/audio | `output/native_bf16/ltx23_t2v_mlx_1280x768/00_coastal_rescue_00002_.mp4` |
| LTX 2.3 BF16 dev+LoRA I2V | `video_ltx2_3_i2v.json` | `ltx23_i2v_mlx_1280x768` | `1280x768 / 121f / 24fps` | 976.94s | pass, MLX route proof, valid MP4/audio | `output/native_bf16/ltx23_i2v_mlx_1280x768/00_greenhouse_i2v_00001_.mp4` |
| LTX 2.3 BF16 dev+LoRA First-Last | `video_ltx2_3_flf2v.json` | `ltx23_first_last_mlx_1280x768` | `1280x768 / 121f / 24fps` | 2101.16s | pass, MLX route proof, sparse guide attention, valid MP4/audio | `output/native_bf16/ltx23_first_last_mlx_1280x768/00_lighthouse_flf_00001_.mp4` |

Review files:

- `official_workflow_provenance.md`
- `native_1024_720p_validation_report.md`
- `native_1024_720p_review.html`
- `run_results.json`

Additional targeted test added for first-last template conversion: official group-widget defaults for `LTXVAddGuide` and `SamplerEulerAncestral` are restored into generated API prompts.
