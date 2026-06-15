# ComfyUI-MLX Fork Review Package - 2026-06-15

Status: reviewed for push. User approved staging, commits, and push to the fork branch; no PR is created.

## Publish Plan

- Working branch: `codex/local-mlx-backend-spike`
- Target remote: `fork` -> `https://github.com/D-Squarius/ComfyUI-MLX.git`
- Commit author:

```bash
git -c user.name="D-Squarius" \
  -c user.email="D-Squarius@users.noreply.github.com" \
  commit -m "<message>"
```

Commit sequence:

1. `Add MLX island backend support`
2. `Add MLX backend validation tests`
3. `Document ComfyUI-MLX model workflows`

Benchmark scripts and raw benchmark harnesses are now deferred from the first push. They remain useful locally, but the initial fork update should not upload the `benchmarks/` tree.

Push verification commands:

```bash
git status
git log --format='%h %an <%ae> %s' -3
git push fork codex/local-mlx-backend-spike
```

Push target: `fork codex/local-mlx-backend-spike`.

## Review Entry Points

- README preview: `README.md`
- Workflow/model guide: `docs/mlx_native_workflows.md`
- Previous review package: `docs/fork_review_package_20260614.md`
- Current package: `docs/fork_review_package_20260615.md`

## Dependency Notes

- Base `requirements.txt` remains MLX-free.
- MLX routes remain optional in `pyproject.toml`.
- Validated MLX range: `mlx>=0.31.2,<0.32`.
- LTX MLX extras remain optional through `ltx-core-mlx>=0.14.8,<0.15` and `ltx-pipelines-mlx>=0.14.8,<0.15`.
- MLX imports should stay lazy so normal Comfy startup does not require MLX unless an MLX alias or manifest is selected.

Install command for MLX users:

```bash
pip install -e ".[mlx-all]"
```

## Benchmark Evidence For README Claims

Current clean guarded artifact root:

`benchmarks/mlx_vs_baselines/results/clean_guarded_ltx_retest_20260615-095035/`

This run recorded `iogpu.wired_limit_mb=0`, no active benchmark-contaminating `git add`, real Comfy API execution, and MLX route proof.

| Pipeline | Workflow | Resolution | Scored time | Status |
|---|---|---:|---:|---|
| Klein 4B BF16 MLX | Official Flux2 Klein T2I | 1024x1024 | 35.68s avg | Pass, valid image, MLX route proof |
| Klein 9B BF16 MLX | Official Flux2 9B T2I | 1024x1024 | 71.64s avg | Pass, valid image, MLX route proof |
| Z-Image Turbo BF16 MLX | Official Z-Image Turbo T2I | 1024x1024 | 83.02s avg | Pass, valid image, MLX route proof |
| LTX 2.3 BF16 dev+LoRA MLX T2V | Installed official LTX T2V | 512x512 / 49f | 166.75s | Pass, valid MP4/audio, MLX route proof |
| LTX 2.3 BF16 dev+LoRA MLX I2V | Installed official LTX I2V | 512x512 / 49f | 191.99s | Pass, valid MP4/audio, MLX route proof |
| LTX 2.3 BF16 dev+LoRA MLX First-Last | Official LTX FLF | 512x512 / 49f | 281.09s | Pass, sparse guide attention, valid MP4/audio |
| LTX 2.3 BF16 dev+LoRA MLX T2V | Installed official LTX T2V | 1280x768 / 121f | 929.56s | Pass, valid MP4/audio, MLX route proof |

Native T2V timing artifact:

`benchmarks/mlx_vs_baselines/results/ltx23_dev_lora_mlx_native_t2v_timing_20260614-180654/`

- `ComfyUI-MLX LTX 2.3 BF16 dev+LoRA native T2V workflow`
- `1280x768 / 121f / 24fps`
- Scored average total: `948.32s`
- Average island time: `865.53s`
- Stage 1: `307.96s`
- Stage 2: `557.57s`

Historical distilled-package context:

- `ltx23_1024x768_121f_mlx_bf16_20260529-232046`: about `647.74s` average at `1024x768 / 120f`
- `ltx23_dialogue_custom_bf16_1024x768_121f`: about `629.38s` average at `1024x768 / 120f`

These distilled rows are useful context but not strict apples-to-apples with the native BF16 dev+LoRA `1280x768` workflow, so README wording should be conservative.

## Workflow Provenance

Recorded in:

`benchmarks/mlx_vs_baselines/results/clean_guarded_ltx_retest_20260615-095035/official_workflow_provenance.md`

Summary:

- Klein 4B, Klein 9B, and Z-Image Turbo templates matched current Comfy-Org raw workflow files.
- LTX first-last-frame matched current Comfy-Org raw workflow file.
- LTX T2V and I2V used exact installed templates from `comfyui-workflow-templates==0.9.82`; those installed templates differed from current raw GitHub hashes and are documented in the provenance report.
- Generated API prompts changed only loader aliases/manifests, prompt/seed, dimensions/frame settings, input media filenames, and output prefixes.

## Proposed File Groups

### Core MLX Backend And Adapters

- `comfy/backends/`
- MLX integration touchpoints in `comfy/sd.py`, `comfy/samplers.py`, `execution.py`, `folder_paths.py`, `nodes.py`, and `comfy/model_management.py`
- LTX routing support in `comfy/ldm/lightricks/` and `comfy_extras/nodes_lt*.py`
- Klein and Z-Image loader/interception paths through the shared island backend

### Text Encoder And Runtime Support

- `comfy/text_encoders/mlx.py`
- `comfy/text_encoders/acceleration.py`
- `comfy/text_encoders/baseline.py`
- `comfy/text_encoders/gemma4.py`

### Native Workflow Integration

- Existing Comfy loader nodes route to MLX aliases/manifests without adding a public MLX node.
- LTX T2V, I2V, and first-last-frame support preserves native workflow topology.
- First-last-frame support includes sparse guide-attention handling and CFG=1 batch-collapse metadata handling.

### Benchmark And Report Tooling

- `tests/e2e/`
- `mlx_workflows/`
- Lightweight API smoke fixtures and route-proof helpers that are needed by tests

Deferred from first push:

- `benchmarks/`
- raw generated benchmark outputs
- large workflow/report dashboards
- custom MLX wheel exploration tooling
- benchmark-harness tests that import ignored `benchmarks/` modules

### Tests

- MLX backend tests under `tests-unit/comfy_test/mlx*_test.py`
- Klein, Z-Image, LTX, text-encoder, planner, native-op, folder-path, and video API tests that do not depend on ignored benchmark modules
- Existing touched video and folder path tests

### Docs

- `README.md`
- `docs/mlx_native_workflows.md`
- `docs/fork_review_package_20260614.md`
- `docs/fork_review_package_20260615.md`
- `docs/mlx_local_backend_spike.md`
- `docs/plans/june13plan.md`

## Current Code Size Snapshot

- Tracked modified files after cleanup: `23`
- Tracked diff stat after cleanup: `23 files changed, 1808 insertions(+), 503 deletions(-)`
- Untracked files after ignoring `benchmarks/`: `84`
  - `comfy`: `34`
  - `docs`: `8`
  - `mlx_workflows`: `8`
  - `tests`: `2`
  - `tests-unit`: `32`

Approximate untracked text lines after excluding `benchmarks/`:

- `comfy/`: about `13,722`
- `docs/`: about `9,021`
- `mlx_workflows/`: about `201`
- `tests-unit/`: about `9,947`
- `tests/`: about `561`

The first public update should include source, docs, workflow fixtures, focused backend tests, and small e2e fixtures only. Keep the local benchmark tree and benchmark-harness tests for continued private development.

## Intentionally Excluded From Push

The repository ignore rules should keep these local-only:

- `.venv*/`
- `models/`
- `output/`
- generated `input/` media except committed tiny fixtures
- `benchmarks/` for the first public push
- `benchmarks/mlx_vs_baselines/results/`
- `benchmarks/mlx_vs_baselines/custom_mlx_wheel_venvs/`
- `benchmarks/mlx_vs_baselines/native/build/`
- custom MLX wheel outputs
- generated `.mp4`, `.mov`, `.trace`, `.xcresult`, and `.sqlite3` files
- `Vendor/`
- Hugging Face caches and other model/download caches outside the repo

## Push Checklist

- [x] User reviewed and approved pushing this package.
- [x] Run final `git diff --check`.
- [x] Run targeted MLX/backend/native workflow tests.
- [x] Run strict push-candidate PII/secret scan.
- [x] Show final `git status`.
- [x] Show final `git diff --stat`.
- [x] Commit only after approval with `D-Squarius <D-Squarius@users.noreply.github.com>`.
- [x] Push only after approval to `fork codex/local-mlx-backend-spike`.
