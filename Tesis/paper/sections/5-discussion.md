\section{Discussion}

\subsection{Implications of Intervention Mechanisms}

The experimental results reveal two main insights for inference-time complexity control in Small Language Models.

\subsubsection{Instruction-Following as Primary Control Mechanism}

The dominance of contextual prompting over vocabulary weighting suggests that instruction-tuned models exhibit stronger responsiveness to high-level semantic constraints than low-level probability manipulation.

The minimal aggregate benefit of combining interventions (5.7\% more words, 6.7\% fewer difficult words) indicates that prompting already captures most achievable simplification. This suggests that instruction-tuning creates strong semantic priors that are difficult to modulate through token-level probability adjustments alone. For practitioners, this implies that prompt engineering should be prioritized over decoding manipulation for complexity control tasks.

\subsubsection{Architectural Determinants of Intervention Responsiveness}

The substantial heterogeneity in interaction patterns—ranging from negative (Qwen2: +33.5\% FK) to positive (TinyLlama: -17.2\% FK)—challenges the assumption of universal controllability mechanisms. Three architectural factors appear to determine intervention responsiveness:

\textbf{1. Instruction-following capability.} Models with strong instruction-following (Phi3) show minimal benefit from vocabulary weighting, while models with weak instruction-following (TinyLlama) exhibit positive interaction effects. This suggests that vocabulary constraints can partially compensate for insufficient instruction-tuning, though through syntactic restructuring rather than vocabulary substitution.

\textbf{2. Training data composition.} Qwen3's unique positive response to standalone vocabulary weighting (16.8\% FK reduction) and efficient simplification without compensatory verbosity distinguishes it from other tested architectures. This pattern suggests that certain pre-training regimes may predispose models to benefit from lexical constraints.

\textbf{3. Architectural stability under constraints.} Qwen2's output instability and catastrophic complexity increases under combined interventions suggest architectural limitations in handling conflicting generation constraints. This implies that not all small model architectures are suitable for multi-constraint generation tasks.

\subsection{Compensatory Verbosity as a Design Constraint}

The 10\% aggregate word count increase under standalone vocabulary weighting reveals a fundamental tension in constrained generation: when lexical choices are restricted without semantic guidance, models compensate through syntactic elaboration. This compensation represents a critical challenge for soft constraint satisfaction in language models.

The observation that only Qwen3 resisted compensatory verbosity while other models exhibited modest (Phi3, Qwen2) to severe (TinyLlama) complexity escalation suggests that this behavior is not universal but architecture-dependent. This has important implications for vocabulary-based intervention design: lexical constraints must be paired with explicit contextual guidance to prevent compensatory sentence lengthening. Standalone vocabulary manipulation is insufficient for complexity control in most architectures.

\section{Limitations and Future Directions}

\textbf{Sample Size:} The current sample of 15 prompts per model provides preliminary evidence but further exploration is required. Future work should expand the prompt set to include diverse A1-level interaction types (conversation, error correction, cultural topics).

\textbf{Semantic Correctness:} The evaluation framework measures text complexity exclusively, with no quantitative assessment of semantic correctness, factual accuracy, or response completeness. Simpler text may sacrifice important information or introduce factual errors. Future work must incorporate content quality metrics through human evaluation protocols or automated semantic coherence measures.

\textbf{Vocabulary Optimization:} The target vocabulary was selected based on established educational standards but was not systematically optimized for the experimental task. Future work should explore the completeness of this list, evaluating whether the selected vocabulary contains sufficient words to generate complete and coherent sentences.

\textbf{Generation Parameters:} The interaction between vocabulary weighting and other decoding parameters (temperature, top-k, top-p) remains unexplored. Future work should systematically investigate whether adjusted sampling thresholds modulate the effectiveness of vocabulary constraints.

\textbf{Weight Factor Optimization:} The 1.5× weight factor was selected based on preliminary exploration of three discrete values (1.5×, 2.0×, 4.0×). A more systematic optimization approach exploring the continuous weight factor space (e.g., 1.1-2.0×) with finer granularity could identify model-specific optimal values. Additionally, adaptive weight scheduling—varying the weight factor during generation based on context—remains unexplored and may provide superior control compared to static weight application. Future work should also investigate whether different weight factors interact differently with prompting interventions across models.

\textbf{Multi-Turn Dialogue:} The experimental design evaluated single-turn question-answering scenarios. Educational applications frequently require multi-turn dialogue, where complexity control must maintain consistency across conversation history.

\section{Conclusion}

This study demonstrates the viability of inference-time complexity control for Small Language Models in educational applications, with two key contributions:

\textbf{1. Understanding of intervention mechanisms:} Contextual prompting operates through high-level semantic constraints that leverage instruction-tuning, while vocabulary weighting provides low-level probability manipulation that is largely redundant when strong instruction-following is present. The minimal aggregate benefit of combining interventions (5.7\% more words, 6.7\% fewer difficult words) indicates that these mechanisms do not synergize in most architectures.

\textbf{2. Architectural heterogeneity:} Intervention responsiveness is not a universal property but depends critically on instruction-following capability, training data composition, and architectural stability under constraints. This challenges the assumption that complexity control techniques generalize uniformly across model families and necessitates architecture-specific validation.

For deploying Small Language Models in resource-constrained contexts, the findings suggest prioritizing prompt engineering over decoding manipulation and conducting architecture-specific validation before deployment with attention to compensatory verbosity.

Future work should address the semantic correctness limitation through human evaluation and expand to multi-turn dialogue contexts to establish the broader applicability of these findings.
