# State of the Art: Controllable Generation for Text Complexity

Controlling language model outputs to meet specific linguistic constraints represents a critical challenge for educational applications requiring adaptive text generation. Recent advances in controllable generation have explored diverse methodologies, from fine-tuning approaches that embed complexity controls during training to inference-time interventions that manipulate generation without model modification. This review examines two complementary approaches to output control—fine-tuning for multi-objective linguistic control and prompt-based constrained generation—positioning them relative to the present work's focus on real-time complexity control in small language models for beginner language learners.

## Fine-tuning for Multi-Objective Linguistic Control

Nguyen et al. (2024) introduced Multi-Control Tuning (MCTune), a fine-tuning methodology that incorporates multiple linguistic complexity values as explicit controls during instruction tuning. The approach fine-tunes LLaMA2-7B on datasets including Alpaca-GPT4 and WizardLM, embedding complexity specifications directly into the input prompt structure during training. This methodology enables precise control over multiple complexity dimensions simultaneously while maintaining response quality and semantic coherence.

The MCTune framework demonstrates substantial improvements in multi-complexity controllability across diverse generation tasks. By conditioning the model on target complexity values during training, the approach achieves fine-grained control over output characteristics without sacrificing linguistic naturalness or factual accuracy. The methodology represents a significant advance in multi-objective optimization for language models, demonstrating that models can learn to modulate complexity as a controllable attribute rather than a fixed property.

However, the fine-tuning paradigm presents significant limitations for resource-constrained educational deployments. The approach requires substantial computational resources for model retraining, labeled training data annotated with complexity metrics, and locks the model into complexity levels observed during training. Critically, MCTune provides no mechanism for dynamic inference-time adjustment—complexity targets must be specified during fine-tuning rather than adapted per-request based on learner proficiency. For educational contexts requiring offline-first deployment and adaptive complexity adjustment, fine-tuning approaches present prohibitive resource barriers and insufficient flexibility.

## Prompt-based Constrained Generation

Li et al. (2024) investigated lexically constrained generation (LCG) in large language models, focusing on prompt-based control mechanisms for satisfying hard lexical constraints. The study systematically evaluated LLM performance in generating text that includes specified keywords, identifying three critical limitations: (1) position bias, where models exhibit differential constraint satisfaction rates depending on keyword position within the prompt; (2) low responsiveness to decoding parameters, with temperature and top-k sampling demonstrating minimal impact on constraint adherence; and (3) difficulty handling inherent complexity in compound word constraints, where subword tokenization fragments target vocabulary.

To address these limitations, Li et al. proposed a Divide and Conquer Generation strategy that decomposes constraint satisfaction into iterative refinement steps. The approach generates an initial response, identifies unsatisfied constraints, and regenerates text incorporating missed keywords. This decomposition strategy achieved over 90% improvement in constraint satisfaction rates on challenging LCG benchmarks, demonstrating that iterative refinement substantially outperforms single-pass generation for complex constraint scenarios.

The Divide and Conquer methodology provides valuable insights into LLM constraint-following behavior and effective prompting strategies. However, the approach targets hard lexical constraints—requirements that specific words must appear in the output—rather than soft vocabulary preferences suitable for text simplification. Hard constraints enforce binary satisfaction (word present or absent), while educational applications require probabilistic vocabulary steering that guides models toward simpler alternatives without rigid requirements. Additionally, the iterative refinement strategy introduces substantial latency overhead, incompatible with interactive educational applications requiring sub-second response times.

## Research Gap and Positioning

Existing approaches to controllable generation exhibit a fundamental trade-off: fine-tuning methods (MCTune) provide precise control but require substantial resources and lack inference-time flexibility, while prompt-based methods (Divide & Conquer) enable zero-shot control but target hard constraints rather than soft complexity preferences. Critically, no prior work addresses inference-time complexity control for small language models (sub-4B parameters) through soft vocabulary manipulation mechanisms suitable for resource-constrained educational deployments.

The present work addresses four specific gaps in the existing literature:

1. **Inference-time complexity control for SLMs**: Existing approaches require fine-tuning (MCTune) or apply hard constraints (Li et al.). No prior methodology combines inference-time prompting with probability-based vocabulary weighting for small models suitable for on-device educational deployment.
2. **Soft vocabulary manipulation**: Prior work enforces specific word inclusion (hard constraints) rather than probabilistic vocabulary simplification. Probability weighting via logits processors enables soft constraints that guide vocabulary selection without rigid requirements, better suited to text simplification objectives.
3. **Systematic small model evaluation**: Existing research focuses on large models (LLaMA2-7B, GPT-series) requiring cloud infrastructure. Systematic evaluation across small models (33M-3.8B parameters) for offline-first educational contexts remains absent from the literature.
4. **Factorial intervention comparison**: Prompting and decoding manipulation have been studied in isolation. No prior work employs factorial experimental designs to isolate individual effects and interaction effects between intervention types, limiting understanding of synergistic mechanisms.

## Contributions of the Present Work

This research introduces a novel approach combining inference-time contextual prompting with probability weighting via vocabulary-aware logits processors, enabling real-time complexity control without model retraining. The methodology applies soft constraints through 2.0× probability amplification for target vocabulary tokens (493 words from A1 "Starters" curriculum), guiding models toward simpler alternatives while preserving generation flexibility.

The experimental framework employs a 2×2 factorial design (prompting × weighting) across six small language models spanning 33M to 3.8B parameters (TinyStories, Qwen2, Qwen3, TinyLlama, SmolLM, Phi3), generating 192 observations evaluated across six primary readability metrics. This design isolates individual intervention effects and interaction effects, quantifying the relative contribution of prompting versus probability manipulation and identifying synergistic combinations.

The work directly addresses deployment requirements for the Ceibal educational initiative serving 550,000 Uruguayan students, emphasizing on-device inference, offline-first functionality, and sub-second response latency. By focusing on small models and inference-time interventions, the methodology enables cost-efficient, scalable deployment across resource-limited educational institutions while maintaining adaptive complexity control calibrated to A1-level English learners.

---

**References:**

- Li, B., Wang, Y., Meng, T., Chang, K.-W., & Peng, N. (2024). Control Large Language Models via Divide and Conquer. *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 15183–15203. https://aclanthology.org/2024.emnlp-main.850
- Nguyen, D., Malon, C., Dernoncourt, F., Bui, T., & Rossi, R. A. (2024). Multi-Objective Linguistic Control of Large Language Models. *Findings of the Association for Computational Linguistics: ACL 2024*, 4313–4330. https://aclanthology.org/2024.findings-acl.257
