---
date: 2026-02-23T17:43:32Z
researcher: santiago
git_commit: 000f72f512bb26813513fa9ceeac238c7ad7afd0
branch: feature/llamacpp-migration
repository: Tesis
topic: "Include ReadCtrl and Malik et al. in Related Work"
tags: [paper, related-work, rctg, readability, lrec2026, readixstar]
status: complete
last_updated: 2026-02-23
last_updated_by: santiago
type: implementation_strategy
---

# Handoff: Include ReadCtrl and Malik et al. in Related Work

## Task(s)

**Status: planned/discussed — not yet implemented in the .tex file**

The session produced a full workshop submission assessment and a deep analysis identifying
two critical missing citations. The actual edits to the paper have NOT been made yet.
The task for the next agent is to implement the related work changes in the .tex file.

**Two papers to add:**

1. **Malik et al. (2024) — "From Tarzan to Tolkien: Controlling Language Proficiency Level
   of LLMs for Content Generation"**
   ACL Findings 2024. https://aclanthology.org/2024.findings-acl.926/
   — Same task as this paper (RCTG for language learners at CEFR levels). Finds sub-7B
   models struggle at A1. This paper directly addresses that gap. Most important citation
   for novelty positioning.

2. **ReadCtrl (Tran et al., 2024) — "Personalizing Text Generation with
   Readability-Controlled Instruction Learning"**
   arXiv:2406.09205. https://aclanthology.org/2025.in2writing-1.3/
   — Fine-tunes Mistral-7B to generate (not rewrite) text at target readability levels.
   Same task as this paper, training-time approach. Direct training-time vs. inference-time
   contrast.

## Critical References

- `thoughts/paper_assesment_for_workshop` — Full assessment including the deep per-paper
  analysis in Section 7. Contains exact recommended framing for each citation.
- `paper/simple-SLM-FINAL-SUBMISSION.tex` — The paper to be edited. Related Work is
  Sections 2.1–2.3 (lines 72–98).
- `reference_papers/Nie-acl2023.tex` — Confirmed that Nie et al. (2023) IS an RCTG paper
  (keywords → sentence with CEFR vocab constraints), not ATS. Important context for
  positioning the new citations.

## Recent changes

- `thoughts/paper_assesment_for_workshop` — Created and iteratively updated with:
  - Full workshop submission assessment (scores, findings, action items)
  - Section 7: Deep per-paper task analysis (RCTG vs. ATS distinction)
  - Revised tier structure for Related Work suggestions
  - Corrected Nie et al. classification from ATS to RCTG
  - Revised novelty score from 4/10 to 6/10

No changes to the paper .tex file yet.

## Learnings

**The core task distinction — essential for the next agent:**

This paper does **Readability-Controlled Text Generation (RCTG)**: given a question/prompt,
generate a simple response at A1 complexity level. It does NOT do classical ATS
(Automatic Text Simplification), which rewrites existing complex text into simpler form.

This matters because:
- SARI metric does not apply (no source text, no reference simplification)
- Malik et al. and ReadCtrl are RCTG papers — same task as this paper
- Nie et al. (already cited) is also RCTG (keywords → sentence), not ATS
- TinyStories, MCTune are ATS/training-time; DExperts, FUDGE are decoding-time for general
  attributes — these are already correctly cited
- Kew et al. (BLESS), Farajidizaji et al., Cripwell et al. (all ATS, named in the CFP)
  should also be added for venue credibility but are lower priority than Malik/ReadCtrl

**Recommended framing for Malik et al.:**
"Malik et al. (2024) investigate CEFR-level content generation for language learners across
model scales, finding that sub-7B models consistently underperform at extreme proficiency
levels (A1). Our work targets this sub-4B regime and demonstrates that combined prompting
and vocabulary weighting enables A1-level generation in models as small as 0.6B parameters."

**Recommended framing for ReadCtrl:**
"ReadCtrl (Tran et al., 2024) addresses the same generation task through instruction tuning,
conditioning Mistral-7B on target readability scores during training. Unlike ReadCtrl's
training-time approach, our method requires no model modification and operates entirely at
inference time."

**Recommended Related Work restructuring** (from `thoughts/paper_assesment_for_workshop` §7):

```
2.1 Training-Time Complexity Control
    - ATS: TinyStories, MCTune, Agrawal & Carpuat [already cited: first two]
    - RCTG: Nie et al. [already cited], ReadCtrl [ADD]
    - Shared limitation: fixed at training time

2.2 Decoding-Time Control (already present: DExperts, FUDGE)

2.3 Readability-Controlled Text Generation with LLMs  ← NEW SUBSECTION
    - Malik et al. (2024) [ADD — most important]
    - Farajidizaji et al. (2024) [ADD — ATS variant, CFP-named, note task distinction]
    - Kew et al. (2023/BLESS) [ADD — ATS, CFP-named, note task distinction]
    - One-line mention of Cripwell et al. (2023) [ADD — ATS, CFP-named, weakest fit]

2.4 Research Gap and Contributions (already present — revise to contrast with §2.3)
```

**Bibtex entries will need to be added** to `paper/references.bib` for both new papers.
The existing bibliography uses natbib/lrec2026-natbib style.

## Artifacts

- `thoughts/paper_assesment_for_workshop` — Full assessment document (primary artifact)
  - Section 4: Tiered related work gap analysis
  - Section 5: Revised novelty score (6/10) and SARI clarification
  - Section 7: Deep per-paper task analysis with exact recommended framing

## Action Items & Next Steps

1. **Add bibtex entries** to `paper/references.bib`:
   - Malik et al. (2024) ACL Findings — https://aclanthology.org/2024.findings-acl.926/
   - ReadCtrl / Tran et al. (2024) — arXiv:2406.09205

2. **Add a new subsection 2.3** "Readability-Controlled Text Generation" to the Related Work
   in `paper/simple-SLM-FINAL-SUBMISSION.tex` (insert after line 92, before the existing
   "Research Gap and Contributions" subsection). Use the exact framing from the Learnings
   section above and from `thoughts/paper_assesment_for_workshop` §7.

3. **Optionally add ReadCtrl to Section 2.1** as a training-time RCTG contrast (alongside
   Nie et al.), with a forward pointer to the new §2.3.

4. **Revise Section 2.4 "Research Gap and Contributions"** (lines 94–98) to explicitly
   contrast with Malik et al. and ReadCtrl rather than only DExperts/FUDGE/MCTune.

5. **Secondary priority** (for venue credibility): also add Kew et al., Farajidizaji et al.,
   Cripwell et al. bibtex entries and brief mentions in the new §2.3 subsection, explicitly
   noting the ATS vs. RCTG task distinction.

## Other Notes

- The paper is currently anonymized (`\usepackage[review]{lrec2026}` at line 5) — correct
  for double-blind submission.
- Page count is borderline at ~8 pages — adding a new subsection may push over the limit.
  Condense the readability metric formulas (lines 198–232) to free ~0.75 pages before adding
  new related work text. This is also recommended in the assessment independently.
- The paper already has 7 limitations listed — the related work expansion does not require
  changes to that section.
- `paper/references.bib` exists but was not read in this session; verify it exists and check
  for any existing (possibly incomplete) entries for these papers before adding new ones.
