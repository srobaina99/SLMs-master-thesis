---
date: 2026-03-12T12:00:00-03:00
researcher: Claude
git_commit: 2fe39cb
branch: feature/refactor
repository: SLMs-master-thesis
topic: "Is PROJECT_STRUCTURE.md up to date with the current codebase?"
tags: [research, codebase, documentation, project-structure]
status: complete
last_updated: 2026-03-12
last_updated_by: Claude
---

# Research: Is PROJECT_STRUCTURE.md up to date?

**Date**: 2026-03-12
**Researcher**: Claude
**Git Commit**: 2fe39cb
**Branch**: feature/refactor
**Repository**: SLMs-master-thesis

## Research Question
Is `Tesis/PROJECT_STRUCTURE.md` up to date with the current codebase?

## Summary

PROJECT_STRUCTURE.md is **mostly accurate** but has **~15 discrepancies** ranging from missing files/directories to incorrect filenames and a phantom file reference. The core directory layout and architecture descriptions are correct. The main gaps are: new debug scripts, new docs, missing `thoughts/shared/research/` section, a non-existent `paper/CLAUDE.md`, a GGUF filename case error, and several omitted `__init__.py` files.

## Detailed Findings

### Accurate sections (no issues)
- Top-level directory map (lines 17-35) — correct
- `src/framework/` layout (lines 64-81) — correct
- `src/text_complexity/` layout (lines 83-87) — correct
- `tests/` layout (lines 90-97) — correct
- `legacy/` layout (lines 126-134) — correct
- `data/vocabularies/` (lines 135-137) — correct
- `notebooks/legacy/` (lines 146-148) — correct
- `paper/` layout (lines 187-209) — mostly correct (one exception below)
- `Bibliografía/` layout (lines 224-240) — correct
- `reference_papers/` (lines 247-252) — correct
- Model wrapper hierarchy description — correct
- Experiment workflow description (lines 161-168) — correct

### Discrepancies Found

#### 1. Missing files in `scripts/` (lines 47-62)

Two untracked debug scripts exist but are not documented:
- `scripts/debug_logit_bias_effect.py`
- `scripts/debug_space_prefix.py`

Also missing: `scripts/legacy_tests/README.md`

#### 2. Missing `__init__.py` files throughout `src/`

The doc omits all `__init__.py` files. These exist in:
- `src/__init__.py`
- `src/framework/__init__.py`
- `src/framework/core/__init__.py`
- `src/framework/experiments/__init__.py`
- `src/framework/models/__init__.py`
- `src/text_complexity/__init__.py`

#### 3. `paper/CLAUDE.md` does not exist (line 203)

The doc lists `paper/CLAUDE.md` as "AI assistant instructions for paper editing" but this file does **not exist**. There is a `Tesis/CLAUDE.md` at the Tesis root (untracked), but nothing inside `paper/`.

#### 4. GGUF filename case mismatch (line 144)

- **Doc says**: `tinyllama-1.1b-chat-v1.0.Q4_0.gguf` (lowercase `t`)
- **Actual file**: `TinyLlama-1.1b-chat-v1.0.Q4_0.gguf` (uppercase `T`)

#### 5. Missing docs files (lines 169-182)

The doc omits:
- `docs/WEIGHTING_MECHANISM.md` (untracked)
- `docs/meeting_transcriptions/` directory with `21-11-25_progress`
- `docs/weekly_progress/week_29-09.md`

The docs table (lines 170-182) lists only 8 entries; there are actually 11 items (9 docs + 2 subdirectories with content).

#### 6. `thoughts/` section is incomplete (lines 256-269)

Missing entries:
- `thoughts/shared/research/` directory (2 research documents):
  - `2026-03-04-hyperparameter-experiments-mapping.md`
  - `2026-03-10-subtoken-space-prefix-handling.md`
- `thoughts/shared/handoffs/general/2026-03-05_08-29-01_hyperparameter-experiments-mapping.md` (third handoff)

#### 7. `Tesis/CLAUDE.md` not in top-level map

`Tesis/CLAUDE.md` exists (untracked) but is not listed in the top-level directory map (lines 17-35).

#### 8. `src/evaluation/` description could be clearer (line 88)

Described as "Legacy results placeholder (mostly empty)" — accurate, but could note it contains only `.DS_Store` files and no Python code. The entire subtree (`experiment_framework/results/{Phi3,Qwen2,Qwen3,TinyLlama}/`) is macOS metadata artifacts.

#### 9. SmolLM wrapper absence not documented

`SmolLM-1.7B-Instruct.Q4_K_M.gguf` is listed in models/gguf/ and a `test_smollm_llamacpp_integration.py` exists in legacy_tests, but there is no `smollm_llamacpp_wrapper.py` in `src/framework/models/`. The model wrapper hierarchy section doesn't mention SmolLM at all, which is correct (no wrapper exists), but the GGUF listing implies it's a supported model.

#### 10. `.cursor/rules/` not documented in detail

Listed in the directory map (line 152) but the 4 `.mdc` rule files are not enumerated:
- `brainstorming.mdc`, `concise.mdc`, `core.mdc`, `request.mdc`

### Minor/cosmetic notes
- `results/` section (lines 99-124) is simplified but accurate for the directory structure. Specific CSV/JSON filenames within subdirectories are not exhaustively listed (reasonable).
- `tests/__init__.py` exists but not listed (line 90-97).
- The date in PROJECT_STRUCTURE.md header says "2026-03-04" which was accurate when written but is now 8 days old.

## Severity Classification

| # | Issue | Severity |
|---|-------|----------|
| 1 | Missing debug scripts | Low (untracked, may be temporary) |
| 2 | Missing `__init__.py` files | Low (conventional omission) |
| 3 | **Phantom `paper/CLAUDE.md`** | **High** (references non-existent file) |
| 4 | **GGUF filename case error** | **Medium** (could cause file-not-found on case-sensitive FS) |
| 5 | Missing docs entries | Medium |
| 6 | Incomplete `thoughts/` section | Medium |
| 7 | Missing `CLAUDE.md` in top-level map | Low (untracked file) |
| 8 | `src/evaluation/` description | Low |
| 9 | SmolLM ambiguity | Medium |
| 10 | `.cursor/rules/` detail | Low |

## Open Questions
- Should `paper/CLAUDE.md` be created, or should the reference be removed from PROJECT_STRUCTURE.md?
- Should SmolLM be added as a supported model (with a wrapper), or should the GGUF be removed from the documented list?
- Are the debug scripts (`debug_logit_bias_effect.py`, `debug_space_prefix.py`) permanent additions that should be documented?
