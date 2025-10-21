# Related Work

Text generation with language models has evolved from rule-based systems and statistical n-gram models to neural architectures capable of producing fluent, contextually coherent text. Recent work has demonstrated that even small-scale models can generate simple, coherent narratives when trained on appropriately constrained data. For instance, \citet{eldan2023tinystories} introduced TinyStories, a synthetic dataset of short stories using vocabulary typically understood by 3-4 year-olds, showing that models with fewer than 10 million parameters can produce grammatically correct, multi-paragraph stories. This demonstrates the viability of small models for generating simple text, though controlling linguistic complexity dynamically at inference time remains an open challenge.

Controllable text generation has emerged as a critical research area for applications requiring precise linguistic attributes. This work intersects two primary research streams: linguistic complexity control in language models and decoding-time probability manipulation. We position our contributions relative to recent advances in both domains, emphasizing the unique challenges posed by resource-constrained educational deployments.

## Linguistic Complexity Control

Recent work has explored methods for controlling linguistic complexity in language model outputs. \citet{nguyen2024multi} introduced Multi-Control Tuning (MCTune), which fine-tunes LLaMA2-7B by embedding multiple linguistic complexity values directly into the instruction tuning process. By conditioning the model on target complexity specifications during training, MCTune achieves control over multiple complexity dimensions simultaneously while maintaining response quality and semantic coherence.

\citet{nie2023lexical} introduced lexical complexity-controlled sentence generation specifically for language learning. Their approach embeds complexity representations directly into the model architecture and enforces constraints on sentence length and syntactic complexity during decoding.

However, these training-time approaches present fundamental limitations. Both methods require considerable computational resources for model retraining, labeled training data, and lock the model into complexity levels observed during training. Also, complexity targets must be specified during training rather than adapted based on individual learner proficiency.

## Decoding-Time Control via Probability Manipulation

Decoding-time control methods manipulate token probabilities during generation to steer language model outputs without retraining. \citet{liu2021dexperts} introduced DExperts, which combines a pretrained language model with expert and anti-expert models in a product of experts formulation, adjusting token logits by computing the difference between expert and anti-expert predictions.

\citet{yang2021fudge} introduced FUDGE (Future Discriminators for Generation), which trains a binary classifier to predict whether a desired attribute will be satisfied in the complete future sequence based on the current partial generation. During generation, FUDGE multiplies the base model's token probabilities with the predictor's future satisfaction probabilities, then renormalizes according to Bayes' Rule, enabling attribute control such as formality or topic adherence without modifying the base model.

While these decoding-time methods demonstrate the effectiveness of probability manipulation for attribute control through learned models (expert/anti-expert models or future discriminators), they primarily target general attributes (toxicity, sentiment, formality) rather than linguistic complexity for educational applications. Furthermore, both approaches require training auxiliary models (expert models or predictors) and focus on large models.

## Research Gap and Contributions

Existing approaches to controllable generation exhibit a fundamental trade-off: fine-tuning methods (MCTune) provide precise control but require substantial resources and lack inference-time flexibility, while decoding-time methods (DExperts, FUDGE) enable inference-time control but require training auxiliary models (expert models or predictors) and target general attributes rather than linguistic complexity. Additionally, no prior work addresses inference-time complexity control for small language models (sub-4B parameters) through predefined vocabulary manipulation mechanisms suitable for resource-constrained deployments.

This work explores a novel approach combining inference-time contextual prompting with probability weighting via vocabulary-aware logits processors, enabling real-time complexity control without model retraining or auxiliary model training. Unlike MCTune's fine-tuning paradigm, our methodology applies soft constraints through probability amplification for target vocabulary tokens from the A1 "Starters" curriculum. Unlike DExperts and FUDGE which learn probability models for attribute control, our approach employs direct vocabulary probability amplification using predefined A1 word lists, eliminating training overhead.

% References (BibTeX)

```bibtex
@Inproceedings{liu2021dexperts,
  author = 	 "Alisa Liu and Maarten Sap and Ximing Lu and Swabha Swayamdipta and Chandra Bhagavatula and Noah A. Smith and Yejin Choi",
  title = 	 "DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts",
  booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
  year = 	 "2021",
  month = 	 "august",
  pages = 	 "6691--6706",
  address =  "Online",
  publisher = "Association for Computational Linguistics"}

@Inproceedings{nguyen2024multi,
  author = 	 "Dung Nguyen and Christopher Malon and Franck Dernoncourt and Trung Bui and Ryan A. Rossi",
  title = 	 "Multi-Objective Linguistic Control of Large Language Models",
  booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
  year = 	 "2024",
  month = 	 "august",
  pages = 	 "4313--4330",
  address =  "Bangkok, Thailand",
  publisher = "Association for Computational Linguistics"}

@Inproceedings{yang2021fudge,
  author = 	 "Kevin Yang and Dan Klein",
  title = 	 "FUDGE: Controlled Text Generation With Future Discriminators",
  booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
  year = 	 "2021",
  month = 	 "june",
  pages = 	 "3511--3535",
  address =  "Online",
  publisher = "Association for Computational Linguistics"}

@Inproceedings{nie2023lexical,
  author = 	 "Jianghao Nie and Yun Chen and Gondy Leroy",
  title = 	 "Lexical Complexity Controlled Sentence Generation for Language Learning",
  booktitle = "Proceedings of the 22nd Chinese National Conference on Computational Linguistics",
  year = 	 "2023",
  pages = 	 "611--626",
  publisher = "Chinese Information Processing Society of China"}

@article{eldan2023tinystories,
  author = 	 "Ronen Eldan and Yuanzhi Li",
  title = 	 "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?",
  journal = "arXiv preprint arXiv:2305.07759",
  year = 	 "2023"}
```
