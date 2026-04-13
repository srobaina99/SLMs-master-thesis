---
date: 2026-02-23T18:46:44-03:00
researcher: Claude Sonnet 4.6
git_commit: b04a67b544cab8d6606b715a748edf9f5bf55044
branch: feature/llamacpp-migration
repository: Tesis
topic: "Paper Shortening for 8-page limit"
tags: [paper, latex, shortening, lrec2026]
status: complete
last_updated: 2026-02-23
last_updated_by: Claude Sonnet 4.6
type: implementation_strategy
---

# Handoff: Paper shortening — LREC 2026 submission

## Task(s)

The paper `paper/simple-SLM-FINAL-SUBMISSION.tex` was 9.7 pages and needs to be at most 8
pages (conclusion and earlier; sections after conclusion are not counted). The session
produced a full audit of shortening opportunities and applied several of them directly to
the paper. Status:

- **Completed**: Introduction example replaced (P12 → P11)
- **Completed**: Discussion Section 5.3 condensed (3 paragraphs → 1)
- **In progress / pending application**: Related Work, Metrics, Computational Environment,
  Basic Weight Exploration — suggestion MD files written but not yet applied to the .tex

## Critical References

- `paper/simple-SLM-FINAL-SUBMISSION.tex` — the paper being shortened
- `Codigo/src/evaluation/experiment_framework/results/Phi3/full_data/Phi3_full_experiment_full_1023_0034.csv` — raw results used to pick the new intro example

## Recent changes

- `paper/simple-SLM-FINAL-SUBMISSION.tex:37-48` — Introduction example changed from prompt
  P12 ("What do you do at school?") to P11 ("What is a 'friend'?"). Control response
  shortened from 148 words to 92 words; FK stats updated (12.2→12.0, Grade 2.5→2.8);
  difficult words stat added (29→1).
- `paper/simple-SLM-FINAL-SUBMISSION.tex:486-492` — Discussion Section 5.3 "Automatic
  Metric Reliability" condensed from 3 paragraphs (~10 lines) to 1 paragraph (3 sentences).
  Redundant restatement of correlation values removed; Table reference added; Qwen3 failure
  mode paragraph retained as the sole interpretive contribution.

## Learnings

- The paper currently repeats the full correlation analysis (ρ values, per-metric
  conclusions) verbatim between Results 4.3.1 and Discussion 5.3 — only one copy is needed.
- The intro example swap saves ~5 lines while keeping the same FK contrast magnitude; P11
  ("friend") has the additional advantage of 29→1 difficult words which is a concrete stat.
- The four display-math blocks in the Metrics section each carry a variable-definition line
  below them — removing both together saves ~5 lines per metric (~20 total).
- The "Basic Weight Exploration" subsection conclusion (Selected Weight Factor paragraph) is
  fully redundant with its Results paragraph; collapsing to 2 sentences saves ~14 lines.
- Section 2 (Related Work) opening paragraph and the standalone "fundamental limitations"
  paragraph at the end of 2.1 are pure scaffolding and can be removed without information
  loss.

## Artifacts

Suggestion MD files (proposed LaTeX + cut rationale tables):

- `paper/related_work_condensed.md` — condensed Related Work (Section 2)
- `paper/metrics_condensed.md` — condensed Metrics definitions (Section 2.4 / 3.3)
- `paper/computational_env_condensed.md` — condensed Computational Environment subsection
- `paper/weight_exploration_condensed.md` — condensed Basic Weight Exploration subsection

Already applied to the paper:

- `paper/simple-SLM-FINAL-SUBMISSION.tex:37-48` — new intro example
- `paper/simple-SLM-FINAL-SUBMISSION.tex:486-489` — condensed Section 5.3

## Action Items & Next Steps

The following suggestion files are ready to apply to the .tex in order of estimated line
savings (largest first):

1. Apply `paper/weight_exploration_condensed.md` → replaces `simple-SLM-FINAL-SUBMISSION.tex:173-181` (~14 lines saved)
2. Apply `paper/metrics_condensed.md` → replaces `simple-SLM-FINAL-SUBMISSION.tex:198-246` (~20 lines saved)
3. Apply `paper/related_work_condensed.md` → replaces `simple-SLM-FINAL-SUBMISSION.tex:61-91` (~15 lines saved)
4. Apply `paper/computational_env_condensed.md` → replaces `simple-SLM-FINAL-SUBMISSION.tex:277-286` (~7 lines saved)
5. After applying all, compile the paper and verify it fits within 8 pages.
6. If still over 8 pages, the remaining candidates identified in the initial audit are:
   - Shorten the roadmap paragraph (`tex:58`) to 2 lines
   - Condense model-specific interaction patterns (Results 4.2.1) — each of 4 model
     paragraphs can lose 1-2 lines
   - Merge Tables 1a and 1b into a single table
   - Trim the Figure 1 description paragraph (`tex:341`)
   - Condense the Prompt Design bullet list (`tex:141-148`)

## Other Notes

- The paper uses the `lrec2026` style with `[review]` option (anonymized). Page count
  target is 8 pages up to and including the Conclusion section; Limitations, Ethics, and
  References do not count toward the limit.
- All four suggestion MD files follow the same format: proposed LaTeX block (ready to
  copy-paste) followed by a table of what was cut and why.
- The `humanlayer thoughts sync` command was not available in this environment.
