# PII And Sensitive Data Audit - 2026-06-15

Status: review audit complete. This report is sanitized and does not repeat local absolute paths from raw findings.

## Scope

The audit covered the current MLX fork work across three groups:

| Category | Input files | Text files scanned | Notes |
|---|---:|---:|---|
| Tracked diff | 24 | 24 | Current modified tracked files |
| Untracked shippable files | 83 | 81 | New source, docs, tests, and fixtures after `/benchmarks/` ignore |
| Ignored local artifacts | 18,575 | 8,618 | Local benchmark/docs/workflow artifacts, excluding binaries, media, caches, venvs, models, and pycache |

Excluded from scanning by default: `.venv*/`, `models/`, `output/`, `Vendor/`, `.git/`, `__pycache__/`, Hugging Face caches, generated media, model weights, traces, sqlite files, and other binary/build outputs.

## Checks

The scan searched for:

- Personal identity/path terms: user-name variants, local machine name, personal absolute paths, temp paths, and local IP patterns.
- Token/secret patterns: Hugging Face, GitHub, OpenAI/OpenRouter style token shapes, authorization headers, API keys, password/session/cookie terms, and inline manifest secret fields.
- Debug/professionalism residue: profanity, rough TODOs, debug/probe/experiment wording, `print`, `breakpoint`, and `pdb`.

## Findings And Actions

| Classification | Area | Action |
|---|---|---|
| Fixed | Klein MLX model root | Removed hard-coded personal local path. The default is now repo-parent-relative, with `COMFY_MLX_KLEIN_MODEL_ROOT` override. |
| Fixed | Z-Image Q8 model root | Removed hard-coded personal local path. The default is now repo-parent-relative, with `COMFY_MLX_Z_IMAGE_Q8_MODEL_PATH` override; legacy `COMFY_MLX_Z_IMAGE_Q8_PATH` still works. |
| Fixed | Redaction tests | Replaced contiguous fake token literals with runtime-constructed strings. |
| Fixed | Review docs | Replaced absolute local output/repo paths with relative paths. |
| Defer/exclude | `benchmarks/` | The local benchmark tree contains many local absolute paths and debug/probe labels. `/benchmarks/` is now ignored and should not be uploaded in the first public push. |
| Fixed | `comfy_extras/nodes_slg.py` | Removed the large LTX block-backbone probe/debug addition from the tracked diff. |
| False positive | Redaction code and `use_hf_token` fields | Scanner hits are defensive code or config flags, not embedded credentials. |
| False positive | Generic Windows examples in `nodes.py` | `C:/Users/username/...` examples are generic upstream-style docs, not personal PII. |

## Raw Scan Summary

The broad scanner intentionally found many local-only hits in ignored benchmark artifacts:

| Category | PII/path-like | Secret-like | Debug/probe-like |
|---|---:|---:|---:|
| Tracked diff | 7 | 2 | 67 |
| Untracked shippable | 0 | 18 | 11 |
| Ignored local artifacts | 113,929 | 172 | 3,031 |

The tracked PII/path-like hits are generic `C:/Users/username/...` examples. The tracked secret-like hits are `hf_gemma4` symbol names, not tokens. The untracked secret-like hits are redaction code, `use_hf_token` tests, `hf_snapshot` test names, and session-variable test names.

Full raw findings stayed local-only and are not part of the push candidate because they contain ignored benchmark paths.

## Acceptance Scan

The stricter push-candidate acceptance scan found no hits for:

- user-name variants requested for this audit
- personal absolute paths and project-directory paths
- local temp paths and `192.168.*` IPs
- token-shaped literals such as `hf_...`, `ghp_...`, `github_pat_...`, or `sk-...`

## Validation

Post-cleanup checks passed:

- `git diff --check`
- `py_compile` on touched backend/test files
- Focused path-resolution tests for Klein and Z-Image env overrides
- Focused manifest/redaction tests after fake-token cleanup

## Required Before Public Push

- Exclude `benchmarks/` from the public push.
- Keep benchmark-harness tests that import ignored `benchmarks/` modules out of the first staged push candidate.
- Re-run the strict acceptance scan after any final pruning or staging.
