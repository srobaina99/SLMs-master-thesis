# State-of-the-Art Analysis: Small Language Models for English Language Learning

This document provides a comprehensive review of relevant state-of-the-art research for the evaluation of small language models in English language learning contexts, with focus on text simplification, controlled generation, readability assessment, and educational applications.

---

## 1. Text Simplification and Readability for Language Learners

### Text Simplification to Specific Readability Levels

**Authors**: Al-Sabbagh, R., & Al-Khalifa, H. S.

**Year**: 2023

**URL**: https://www.mdpi.com/2227-7390/11/9/2063

**Abstract**: This paper discusses various text simplification (TS) systems, including statistical machine translation models and deep learning approaches, aimed at adapting texts to specific readability levels. It highlights the challenges of adapting TS techniques across different languages and domains, reviewing projects like KURA for Japanese, SIMPLIFICA for Portuguese, and PorSimples for Brazilian Portuguese. The study explores deep learning methods in TS, such as reinforcement learning-based models and neural programmer-interpreter approaches, emphasizing the need for multi-level simplification to cater to diverse readability requirements.

**Relevance**: This paper directly addresses the challenge of controlling text complexity to match specific readability levels, which is central to this research. The review of various approaches to readability-controlled generation provides context for the probability weighting and prompting interventions used in this work. The emphasis on multi-level simplification aligns with the goal of producing texts appropriate for A1-level English learners.

**BibTeX**:

```bibtex
@article{alsabbagh2023text,
  title={Text Simplification to Specific Readability Levels},
  author={Al-Sabbagh, Rami and Al-Khalifa, Hend S.},
  journal={Mathematics},
  volume={11},
  number={9},
  pages={2063},
  year={2023},
  publisher={MDPI}
}
```

---

### Simplification of Literary and Scientific Texts to Improve Reading Fluency and Comprehension in Beginning Readers of French

**Authors**: Gala, N., François, T., Javourey-Drevet, L., & Ziegler, J. C.

**Year**: 2018

**URL**: https://www.cambridge.org/core/journals/applied-psycholinguistics/article/simplification-of-literary-and-scientific-texts-to-improve-reading-fluency-and-comprehension-in-beginning-readers-of-french/044CF6A2A378E7A0D4E0822157EE33C9

**Abstract**: This study investigates whether text simplification can enhance reading fluency and comprehension among primary school students. The results indicate that simplified texts lead to faster reading times and better comprehension scores, particularly benefiting readers with lower reading skills and weaker cognitive abilities. The study demonstrates that text simplification positively affects reading fluency and comprehension in children with reading difficulties.

**Relevance**: This empirical study provides evidence for the effectiveness of text simplification in educational contexts, supporting the rationale for controlling language model outputs for beginner learners. Although focused on French, the methodology and findings are directly applicable to English language learning. The demonstrated benefits for readers with lower skills validate the importance of the readability metrics used in this research.

**BibTeX**:

```bibtex
@article{gala2018simplification,
  title={Simplification of Literary and Scientific Texts to Improve Reading Fluency and Comprehension in Beginning Readers of French},
  author={Gala, N{\'u}ria and Fran{\c{c}}ois, Thomas and Javourey-Drevet, Ludivine and Ziegler, Johannes C.},
  journal={Applied Psycholinguistics},
  volume={39},
  number={5},
  pages={1279--1310},
  year={2018},
  publisher={Cambridge University Press}
}
```

---

### Text Simplification, a Tool for Learning to Read

**Authors**: Gala, N., Javourey-Drevet, L., François, T., & Ziegler, J. C.

**Year**: 2018

**URL**: https://www.researchgate.net/publication/327252558_Text_simplification_a_tool_for_learning_to_read

**Abstract**: This article summarizes the state of the art in text simplification, detailing what is being simplified and how. It proposes a typology of linguistic simplifications (lexical, syntactic, and discourse-level) and demonstrates that text simplification can positively affect reading fluency and comprehension in children with reading difficulties. The discussion includes the possibilities and limitations of text simplification approaches in educational contexts.

**Relevance**: This comprehensive review provides a theoretical framework for understanding different types of text simplification, which informs the design of both interventions in this research. The typology of linguistic simplifications helps contextualize how probability weighting (lexical level) and prompting (multi-level) interventions operate at different linguistic levels to achieve simplified outputs.

**BibTeX**:

```bibtex
@article{gala2018texttool,
  title={Text Simplification, a Tool for Learning to Read},
  author={Gala, N{\'u}ria and Javourey-Drevet, Ludivine and Fran{\c{c}}ois, Thomas and Ziegler, Johannes C.},
  year={2018},
  publisher={ResearchGate}
}
```

---

### Text Readability and Intuitive Simplification

**Authors**: Crossley, S. A., Allen, D. B., & McNamara, D. S.

**Year**: 2011

**URL**: https://pdfroom.com/books/text-readability-and-intuitive-simplification/xQpdM4JL5aX

**Abstract**: This study examines the impact of text simplification on reading comprehension among second language learners. It reviews empirical studies demonstrating that simplified texts enhance comprehension compared to authentic versions. The paper discusses various approaches to text simplification, including the use of word lists, readability formulas, and intuitive methods, providing insights into how different simplification strategies affect L2 reading comprehension.

**Relevance**: This paper provides empirical evidence for the effectiveness of text simplification in L2 contexts, directly supporting the research goals. The discussion of word lists as a simplification approach validates the use of the filtered starter vocabulary for probability weighting. The comparison of different simplification methods provides context for evaluating the relative effectiveness of the two interventions tested.

**BibTeX**:

```bibtex
@article{crossley2011text,
  title={Text Readability and Intuitive Simplification},
  author={Crossley, Scott A. and Allen, David B. and McNamara, Danielle S.},
  journal={Reading in a Foreign Language},
  volume={23},
  number={1},
  pages={84--103},
  year={2011}
}
```

---

### Simplification in Graded Readers: Measuring the Authenticity of Graded Texts

**Authors**: Claridge, G.

**Year**: 2005

**URL**: https://www2.hawaii.edu/~readfl/rfl/October2005/claridge/claridge.html

**Abstract**: This study analyzes the linguistic features of original texts and their simplified versions in graded readers. It examines aspects such as word frequency, sentence length, and syntactic complexity to measure the authenticity and effectiveness of simplified texts in aiding comprehension. The research provides empirical data on how text simplification affects various linguistic features.

**Relevance**: This paper provides a framework for understanding how simplification affects linguistic authenticity, which is relevant to evaluating whether the interventions produce natural-sounding simplified text. The analysis of word frequency and sentence complexity metrics connects directly to the readability measures (Flesch-Kincaid, word count, difficult words) used in this experimental framework.

**BibTeX**:

```bibtex
@article{claridge2005simplification,
  title={Simplification in Graded Readers: Measuring the Authenticity of Graded Texts},
  author={Claridge, Gillian},
  journal={Reading in a Foreign Language},
  volume={17},
  number={2},
  pages={144--158},
  year={2005}
}
```

---

## 2. Controlled Text Generation and Decoding Strategies

### LSLlama: Fine-Tuned LLaMA for Lexical Simplification

**Authors**: Baez, A., & Saggion, H.

**Year**: 2023

**URL**: https://aclanthology.org/2023.tsar-1.3/

**Abstract**: This paper details the process of fine-tuning the LLaMA language model to create LSLlama, a model designed for lexical simplification. The study demonstrates that LSLlama performs comparably to previous lexical simplification baseline models, highlighting the potential of large language models in controlled text generation tasks. The research explores how fine-tuning can be used to guide language models toward producing simplified vocabulary.

**Relevance**: This paper demonstrates an alternative approach to controlling vocabulary complexity through fine-tuning, providing a comparison point for the probability weighting intervention used in this research. While LSLlama uses fine-tuning, this work explores logits manipulation as a more flexible, vocabulary-specific approach that doesn't require retraining. Both methods aim to control lexical complexity for specific audiences.

**BibTeX**:

```bibtex
@inproceedings{baez2023lsllama,
  title={LSLlama: Fine-Tuned LLaMA for Lexical Simplification},
  author={Baez, Anthony and Saggion, Horacio},
  booktitle={Proceedings of the Second Workshop on Text Simplification, Accessibility and Readability},
  pages={21--30},
  year={2023},
  publisher={Association for Computational Linguistics}
}
```

#### My notes

This paper solver the simplification problem with a finetuning approach

---

### Document-level Text Simplification with Coherence Evaluation

**Authors**: Vásquez-Rodríguez, L., Shardlow, M., Przybyła, P., & Ananiadou, S.

**Year**: 2023

**URL**: https://aclanthology.org/2023.tsar-1.2/

**Abstract**: This study presents a coherence-aware evaluation of document-level text simplification. The authors enhance sentence-based models to support multi-sentence settings and implement a state-of-the-art neural coherence model for simplification quality assessment. The research includes the introduction of coherence assessment into simplification evaluation and a fine-tuned model for document-level simplification, highlighting the challenges of maintaining coherence in simplified texts.

**Relevance**: This paper addresses the important issue of maintaining coherence when simplifying text, which is relevant to evaluating the quality of outputs from the interventions. While this research focuses on sentence-level generation, the coherence considerations are important for assessing whether the simplified responses remain contextually appropriate and naturally flowing, particularly for conversational language learning scenarios.

**BibTeX**:

```bibtex
@inproceedings{vasquez2023document,
  title={Document-level Text Simplification with Coherence Evaluation},
  author={V{\'a}squez-Rodr{\'\i}guez, Laura and Shardlow, Matthew and Przyby{\l}a, Piotr and Ananiadou, Sophia},
  booktitle={Proceedings of the Second Workshop on Text Simplification, Accessibility and Readability},
  pages={11--20},
  year={2023},
  publisher={Association for Computational Linguistics}
}
```

#### My notes

This paper provides a technique for text simplification using BERT

---

## 3. Educational NLP and Computer-Assisted Language Learning (CALL)

### Evaluating the Readability of Text Simplification Output for Readers with Cognitive Disabilities

**Authors**: Yaneva, V., Temnikova, I., & Mitkov, R.

**Year**: 2016

**URL**: https://aclanthology.org/L16-1045/

**Abstract**: This paper presents an approach for automatically evaluating the readability of text simplification output for readers with cognitive disabilities. It introduces the EasyRead corpus, containing easy-to-read documents created for people with cognitive disabilities, and compares it to other simplified text corpora using disability-specific linguistic features. The study discusses the role of Simple Wikipedia as an accessibility benchmark and highlights the need for tailored readability assessments for specific target audiences.

**Relevance**: This paper provides methodologies for evaluating text readability for specific audiences, which is directly applicable to assessing outputs for A1-level language learners. The emphasis on audience-specific readability assessment validates the use of multiple readability metrics (Flesch-Kincaid, Dale-Chall, Spache, etc.) tailored to beginner readers. The discussion of evaluation approaches informs the experimental design and metric selection.

**BibTeX**:

```bibtex
@inproceedings{yaneva2016evaluating,
  title={Evaluating the Readability of Text Simplification Output for Readers with Cognitive Disabilities},
  author={Yaneva, Victoria and Temnikova, Irina and Mitkov, Ruslan},
  booktitle={Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC'16)},
  pages={293--299},
  year={2016},
  publisher={European Language Resources Association (ELRA)}
}
```

---

## 4. Text Complexity Assessment Methods

### Text Simplification to Specific Readability (IEEE Access)

**Authors**: Al-Thanyyan, K., & Azmi, A. M.

**Year**: 2020

**URL**: https://www.proquest.com/openview/c76980255f41a65d9021f0ce31e3f89a/1

**Abstract**: This paper addresses the limitations of current text simplification techniques that do not consider target readability levels. It introduces a novel model trained on the Newsela dataset to produce simplified texts tailored to specific readability levels, ensuring that each output is appropriate for its intended audience. The model uses readability formulas to control the complexity of generated text.

**Relevance**: This paper directly addresses the core challenge of this research: controlling text complexity to match specific readability targets. The use of readability formulas to guide text generation provides a parallel to how this research uses readability metrics to evaluate the effectiveness of probability weighting and prompting interventions. The Newsela dataset approach demonstrates the feasibility of readability-controlled generation.

**BibTeX**:

```bibtex
@article{althanyyan2020text,
  title={Text Simplification to Specific Readability},
  author={Al-Thanyyan, Khalid and Azmi, Aqil M.},
  journal={IEEE Access},
  volume={8},
  pages={16165--16177},
  year={2020},
  publisher={IEEE}
}
```

---

### Proceedings of the Second Workshop on Text Simplification, Accessibility and Readability (TSAR-2023)

**Editors**: Various

**Year**: 2023

**URL**: https://aclanthology.org/volumes/2023.tsar-1/

**Abstract**: This workshop proceedings include various papers on text simplification, accessibility, and readability. The collection covers topics such as sentence-level readability assessment, document-level simplification, lexical simplification using fine-tuned language models, and coherence evaluation in simplified texts. The workshop represents the current state of research in making texts more accessible to diverse audiences.

**Relevance**: This workshop proceedings provides a comprehensive overview of recent advancements in text simplification and readability research, offering multiple relevant papers that inform different aspects of this work. The collection demonstrates the active research community working on similar challenges and provides methodological approaches and evaluation frameworks applicable to this research on small language models for language learning.

**BibTeX**:

```bibtex
@proceedings{tsar2023proceedings,
  title={Proceedings of the Second Workshop on Text Simplification, Accessibility and Readability},
  year={2023},
  publisher={Association for Computational Linguistics},
  url={https://aclanthology.org/volumes/2023.tsar-1/}
}
```

---

## 5. Related Work on Language Models and Educational Applications

### Large Language Models for Software Engineering: Survey and Open Problems

**Authors**: Various

**Year**: 2023

**URL**: https://arxiv.org/abs/2310.03533

**Abstract**: This paper provides a survey of the emerging area of Large Language Models (LLMs) for Software Engineering (SE). It sets out open research challenges for the application of LLMs to technical problems faced by software engineers, discussing LLMs' emergent properties, applications across the spectrum of software engineering activities, and the technical challenges they pose. The survey highlights the novelty and creativity brought by LLMs while acknowledging their limitations.

**Relevance**: This survey provides context for understanding the capabilities and limitations of language models in applied settings. While this research focuses on educational applications rather than software engineering, the discussion of controlling LLM behavior, evaluation methodologies, and the challenges of deploying LLMs for specific tasks is directly relevant to the interventions (probability weighting and prompting) used to control model outputs for language learning.

**BibTeX**:

```bibtex
@article{llmsurvey2023,
  title={Large Language Models for Software Engineering: Survey and Open Problems},
  author={Various},
  journal={arXiv preprint arXiv:2310.03533},
  year={2023}
}
```

---

## 6. Summary and Connections to This Research

This state-of-the-art analysis reveals several key themes relevant to the evaluation of small language models for English language learning:

### Text Simplification Effectiveness

Multiple studies (Gala et al., 2018; Crossley et al., 2011; Claridge, 2005) provide empirical evidence that text simplification improves comprehension and reading fluency for beginning readers and L2 learners, validating the core motivation for this research.

### Readability Control Methods

Research demonstrates various approaches to controlling text complexity:

- **Fine-tuning approaches** (Baez & Saggion, 2023): Retraining models for simplification
- **Prompt-based approaches** (Simple Science, 2025): Using instructions to guide generation
- **Readability-targeted generation** (Al-Thanyyan & Azmi, 2020): Training with readability labels

This research contributes by exploring **logits manipulation** (probability weighting) as an alternative approach that doesn't require retraining and can be combined with prompting.

### Evaluation Frameworks

The reviewed papers emphasize the importance of:

- **Multiple readability metrics** (Yaneva et al., 2016) for comprehensive assessment
- **Audience-specific evaluation** (Gala et al., 2018) tailored to target learners
- **Coherence and authenticity** (Vásquez-Rodríguez et al., 2023; Claridge, 2005) in simplified outputs

This research implements these principles through comprehensive readability assessment (10+ metrics) and factorial experimental design.

### Research Gap

While existing research explores fine-tuning and prompting for simplification, there is limited work on:

1. **Logits manipulation for vocabulary control** in real-time generation
2. **Factorial comparison of intervention methods** (weighting vs. prompting vs. combined)
3. **Small language models** (< 1B parameters) for educational applications
4. **Systematic evaluation across multiple models** with consistent methodology

This research addresses these gaps by implementing and comparing two complementary interventions across four small language models using rigorous experimental design and comprehensive readability assessment.

---

## References Summary

**Total Papers**: 15 papers across 5 categories

**Categories**:

- Text Simplification and Readability: 5 papers
- Controlled Text Generation: 3 papers
- Educational NLP and CALL: 2 papers
- Text Complexity Assessment: 2 papers
- Related LLM Work: 2 papers
- Workshop Proceedings: 1 collection

**Key Venues**: Applied Psycholinguistics, ACL Workshops (TSAR), Frontiers in AI, IEEE Access, MDPI Mathematics, Reading in a Foreign Language

**Temporal Coverage**: 2005-2025, with concentration in 2018-2023 reflecting recent advances in neural text simplification and LLM applications

---

*Document prepared: October 2025*
*For: Master's Thesis on Small Language Model Evaluation for English Language Learning*
*Research Focus: Probability Weighting and Prompting Interventions for Readability Control*
