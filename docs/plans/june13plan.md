# June 13 Plan: Publish MLX Island Backend After Benchmark Refresh

## Summary

Prepare one clean fork push for the MLX island backend, native-workflow support notes, and benchmark evidence. Before README preview, commit, or push, run another focused testing and benchmark refresh so the published claims match the current code.

Push target must be the `fork` remote only: `https://github.com/D-Squarius/ComfyUI-MLX.git`.

## 1. Pre-Push Benchmark Refresh

- Re-run current-code warm benchmarks for the supported model families:
  - Klein 4B BF16/Q8/Q4 through native Flux2 workflows.
  - Klein 9B BF16/Q8/Q4 through native Flux2 workflows.
  - Z-Image Turbo BF16/Q8 through the native Z-Image workflow.
  - LTX 2.3 through stock T2V, I2V, and first-last-frame workflows where validated.
- Route every run through real Comfy API execution.
- Use warm scored runs only: one warmup per backend/workflow, then scored prompts.
- Save route proof showing MLX island execution and no Torch/MPS diffusion-transformer fallback for MLX rows.
- Validate all media before using results:
  - images are correct size, nonblank, and tied to prompt/seed;
  - videos have expected resolution, frame count, fps/duration, audio stream, non-silent audio, nonblack/nonstatic video, and non-flat late frames.
- Keep old benchmark results as context, but publish only current-code numbers or clearly label older context.

## 2. README And Model Support Docs

- Add a concise Apple Silicon MLX island backend section near the top of `README.md`.
- State the user-facing path clearly: use stock/native Comfy workflows and select MLX model aliases in normal loader nodes.
- Supported workflow families:
  - Klein 4B / 9B: native Flux2 workflows via `UNETLoader`.
  - Z-Image Turbo: native Z-Image workflow via `UNETLoader`.
  - LTX 2.3: native LTX T2V/I2V/first-last-frame workflows via LTX checkpoint/text-encoder aliases.
- List supported MLX model sources with links:
  - Klein BF16: [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
  - Klein quantized: [mlx-community/flux2-klein-4b-8bit](https://huggingface.co/mlx-community/flux2-klein-4b-8bit), `mlx-community/flux2-klein-4b-4bit`, [mlx-community/flux2-klein-9b-8bit](https://huggingface.co/mlx-community/flux2-klein-9b-8bit), `mlx-community/flux2-klein-9b-4bit`
  - Z-Image Turbo: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), MLX reference/conversion from [FiditeNemini/z-image-turbo-mlx](https://github.com/FiditeNemini/z-image-turbo-mlx)
  - LTX 2.3 MLX: [dgrauet/ltx-2.3-mlx](https://huggingface.co/dgrauet/ltx-2.3-mlx), [dgrauet/ltx-2.3-mlx-q8](https://huggingface.co/dgrauet/ltx-2.3-mlx-q8), [dgrauet/ltx-2.3-mlx-q4](https://huggingface.co/dgrauet/ltx-2.3-mlx-q4), [dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx)
- Add caveats:
  - validated stock workflow families should work by selecting aliases;
  - arbitrary ControlNet/model patches, some edit/reference-latent paths, exotic LTX guide masks, and unknown wrappers are unsupported unless explicitly listed.

## 3. Cleanup And Staging Hygiene

- Add ignore coverage for generated/local state:
  - `.venv-*`
  - benchmark result folders
  - native build outputs
  - `__pycache__`
  - generated media outputs
  - local model/input folders
- Do not commit:
  - model weights;
  - virtualenvs;
  - raw benchmark media blobs;
  - private HF tokens;
  - local cache/build artifacts.
- Produce a code-footprint report before commit:
  - modified core files;
  - new backend files;
  - benchmark/test/docs files;
  - final added/deleted line counts.

## 4. Validation Gates

- Static checks:
  - `git diff --check`
  - `py_compile` touched backend/benchmark scripts
  - targeted unit tests for MLX island, Klein, Z-Image, LTX, stock-workflow validation, and benchmark report generation.
- Runtime smoke after static checks pass:
  - Klein 4B BF16 MLX T2I stock workflow.
  - Z-Image Turbo BF16 MLX T2I stock workflow.
  - LTX 2.3 MLX short T2V or first-last-frame workflow.
- README preview gate:
  - render or open the README preview locally;
  - verify tables, links, and supported-model wording;
  - get final user approval before staging.

## 5. Commit And Push

- Verify Git identity:
  - push target must be `fork` only;
  - remote owner must be `D-Squarius`;
  - local commit author should be the user's GitHub identity or GitHub noreply email.
- Use a clean branch if needed:
  - `codex/mlx-island-native-workflows`
- Suggested commit message:

```text
Add MLX island backend support for native Comfy workflows

- Add shared MLX island runtime plumbing and model adapters for Klein, Z-Image, and LTX 2.3
- Route supported MLX weights through existing Comfy loader nodes without new user-facing nodes
- Add stock-workflow validation and benchmark harnesses for MPS vs MLX comparisons
- Document supported aliases, model sources, workflow compatibility, and benchmark results
```

- Push only with:

```bash
git push fork codex/mlx-island-native-workflows
```

- Do not push to `origin`.

## Assumptions

- More testing and benchmarking happens before README preview, commit, or push.
- The README will claim support for validated stock workflow families, not blanket support for every possible Comfy workflow.
- Raw benchmark outputs remain local; docs summarize final numbers and point to reproducible harnesses.
- No commit or push happens until the README preview and staged diff are approved.
