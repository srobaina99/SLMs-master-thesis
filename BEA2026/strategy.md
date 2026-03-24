# Strategy for Improving Performance

## High-Impact (try first)

1. **Feature-augmented regression.** The baseline concatenates text fields and feeds them to a transformer. Add explicit numerical features before the regression head:
   - **Cognate similarity:** Levenshtein / Jaro-Winkler distance between `en_target_word` and `L1_source_word`. Spanish cognates (supermarket/supermercado) are easy; false friends are hard.
   - **Word length** of the target word (strong correlation with difficulty). Syllable count tested and found redundant.
   - **Log word frequency** from a standard corpus (e.g., SUBTLEX-US, Zipf scale). Rare words are harder.
   - **Clue informativeness:** ratio of revealed characters in `en_target_clue` to total word length.
   - **POS one-hot encoding** (nouns are generally easier than adverbs).

2. **Model choice: mDeBERTa-v3-base** (~86M effective params). DeBERTa's disentangled attention consistently outperforms RoBERTa on regression/NLI tasks at similar or smaller size. This is the single highest-leverage swap.

3. **Ensemble averaging.** Train 3-5 models with different seeds and/or input field orderings, then average predictions. Typically reduces RMSE by 2-5% for free at inference time.

## Medium-Impact

4. **Input ordering matters.** The baseline concatenates fields as `L1_source_word; L1_context; en_target_clue; en_target_word`. Try putting the English target word first (the thing being predicted) so the model attends to it throughout. Also try dropping `en_target_clue` (it may add noise since it's a masked version of the target word already present).

5. **Cross-lingual joint training with L1 adapters.** Train a shared backbone on all L1s simultaneously but add small L1-specific adapter layers (LoRA or bottleneck). The moderate cross-L1 correlations (r~0.65) mean there's shared signal worth exploiting, while L1-specific heads capture divergence.

6. **Learning rate and scheduler tuning.** The baseline uses lr=3e-5 for 5 epochs. Try:
   - Cosine annealing with warmup (10% of steps)
   - Lower lr (1e-5 to 2e-5) with more epochs (8-10) and early stopping on dev RMSE
   - Discriminative learning rates (lower for pretrained layers, higher for the regression head)

7. **Target scaling.** The official README notes baselines were trained **without target variable scaling**. Normalizing GLMM scores to zero mean / unit variance during training (and inverse-transforming predictions) often improves regression convergence.

## Open Track Specific

8. **External word difficulty features.** Use psycholinguistic databases (MRC Psycholinguistic Database, Brysbaert concreteness/AoA norms) as additional features. Age-of-acquisition is a near-direct proxy for vocabulary difficulty.

9. **LLM distillation.** Prompt GPT-4 / Claude to estimate word difficulty for all items, then use these predictions as an additional feature (not as the sole predictor). This combines LLM world knowledge with the precision of a fine-tuned model.

10. **Data augmentation with back-translation.** Generate additional training pairs by translating L1 context sentences through a different language and back, creating paraphrased contexts for the same difficulty score.

## Things to Avoid

- Don't use models larger than ~350M params for the closed track; it defeats the SLM thesis angle.
- Don't over-tune on dev set -- with only 677 items per L1, overfitting to dev is easy. Use k-fold CV on train instead.
- Don't treat this as classification (binning scores). It's a regression task; RMSE penalizes prediction errors continuously.
