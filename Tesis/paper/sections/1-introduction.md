# Introduction

Language learning applications increasingly rely on AI assistants to provide personalized instruction and practice opportunities. However, a critical challenge emerges when deploying these systems for beginner learners: pretrained language models typically generate text that exceeds the linguistic proficiency of early-stage learners. Consider the following example from a small language model responding to an A1-level English learner:

```
PROMPT: "What does the word 'library' mean?"

CONTROL (Qwen2-0.5B, no interventions):
"The word 'library' means a collection of books, magazines, and other printed 
materials organized and maintained by a library staff..."
→ Flesch-Kincaid Grade: 11.5 | Reading Ease: 42.8

WITH INTERVENTIONS (Qwen2-0.5B, prompting + weighting):
"A library is a place where you can find lots of books to read. It's like a 
big house with many special rooms where people keep their toys and stories!"
→ Flesch-Kincaid Grade: 5.8 | Reading Ease: 83.3
```

The uncontrolled response, while semantically accurate, employs vocabulary and sentence structures far beyond A1-level proficiency, which targets learners with approximately 500-word vocabularies and simple sentence constructions. This complexity mismatch limits the pedagogical value of AI-assisted language learning, as learners cannot understand responses intended to help them.

Existing approaches to controlling text complexity face significant limitations for educational deployments. Fine-tuning methods require substantial computational resources, labeled training data, and lock models into fixed complexity levels determined during training. Recent decoding-time control methods demonstrate promising results but require training auxiliary models (expert models, predictors, or discriminators) and focus predominantly on large language models requiring cloud infrastructure. Critically, no prior work addresses real-time complexity control for small language models through lightweight, training-free mechanisms suitable for resource-constrained educational contexts.

## Research Context

This work is conducted within the broader research project **"Métodos de generación controlada para la construcción de agentes conversacionales de apoyo a la enseñanza"** (Controlled generation methods for building conversational agents to support teaching), funded by ANII (Agencia Nacional de Investigación e Innovación, Uruguay)

**Project Overview:**

The parent project aims to develop a reliable and safe conversational agent prototype for educational purposes. Achieving this goal requires addressing critical limitations of Large Language Models (LLMs)—particularly hallucination, bias reproduction, and lack of source traceability. While LLMs demonstrate excellent linguistic correctness and pragmatic relevance, their tendency to generate "invented" information poses significant risks in educational contexts where accuracy is paramount. The prototype employs multiple mitigation strategies: fine-tuning on educational data, rule-based constraints, and advanced prompting methods to ensure source reliability.

**Target Student Population:**

The project focuses on **Uruguayan English learners with basic to no prior English experience**, particularly students in the first years of schooling. This represents a critical and underserved educational context where traditional resources and tutoring are limited. By developing accessible, appropriately-simplified AI tutors through the Ceibal initiative—Uruguay's national digital education program reaching approximately 550,000 students—the project aims to democratize high-quality English language education across the country's public education system.

**Strategic Focus on Small Language Models:**

The emphasis on Small Language Models (rather than large proprietary models) emerges from institutional and infrastructural imperatives. Deployment of large-scale LLMs (billions of parameters, cloud-based infrastructure) presents substantial barriers: computational costs, dependency on stable internet infrastructure, and licensing constraints. In contrast, SLMs (33M to 3.8B parameters) enable:

- **Cost-Efficient Deployment:** On-device inference eliminates cloud infrastructure costs and subscription fees, enabling sustainable scaling across resource-limited educational institutions
- **Offline-First Functionality:** Critical for Uruguayan contexts with geographically dispersed populations and variable internet connectivity; models execute locally, requiring only initial download
- **Latency Guarantees:** On-device processing ensures sub-second response times essential for interactive learning experiences, independent of network conditions
- **Educational Equity:** Universal accessibility without requiring persistent high-bandwidth connections or institutional server infrastructure

**Contribution to the Larger Research:**

This SLM complexity control study directly addresses a fundamental challenge identified in the parent project: **adapting language models to match learner proficiency levels**. While the broader project tackles reliability and source control, this work specifically addresses the complexity mismatch problem for beginner language learners (A1 level). The developed real-time complexity control methods enable on-device deployment of educational chatbots capable of generating appropriately simple responses for A1 English learners—a critical requirement for Ceibal's large-scale deployment context serving 550,000 students nationwide.

## Research Question and Contributions

This work investigates the following research question: **Can small language models be controlled to produce appropriately simple text for A1 English learners through decoding manipulation and prompt engineering?**

A novel approach combining inference-time contextual prompting with probability weighting via vocabulary-aware logits processors is introduced, enabling real-time complexity control without model retraining or auxiliary model training. The methodology applies soft constraints through probability amplification for target vocabulary tokens from the A1 "Starters" curriculum (494 words), guiding models toward simpler alternatives while preserving generation flexibility.

The key contributions are:

1. **Novel inference-time approach:** First combination of contextual prompting and vocabulary probability weighting for complexity control, requiring no model training or fine-tuning.
2. **Comprehensive small model evaluation:** Systematic evaluation across five models spanning 0.5B to 3.8B parameters (Qwen2, Qwen3, TinyLlama, SmolLM, Phi3), addressing the gap in small model research for educational applications.
3. **Factorial experimental design:** 2×2 design (prompting × weighting) across 5 models and 15 prompts (300 observations) isolates individual intervention effects and interaction effects, quantifying the relative contribution of each mechanism.
4. **Deployment-focused methodology:** Emphasis on on-device inference, offline-first functionality, and sub-second response latency enables practical deployment for 550,000 students via the Ceibal initiative.
5. **Comprehensive readability evaluation:** Six primary readability metrics (Flesch-Kincaid Grade, Gunning Fog, SMOG, Spache, Word Count, Difficult Words) provide multi-dimensional assessment of text complexity calibrated to A1-level proficiency.

## Paper Organization

The remainder of this paper is organized as follows: Section 2 reviews related work on controllable generation, linguistic complexity control, and decoding-time probability manipulation methods. Section 3 presents the experimental methodology, including the two interventions (contextual prompting and probability weighting), the factorial design, model selection, and readability metrics. Section 4 presents results across five models and four configurations, analyzing intervention effectiveness and model comparisons. Section 5 discusses findings, including why certain interventions succeed or fail, trade-offs between complexity and latency, and practical deployment recommendations. Section 6 concludes with limitations and directions for future work.
