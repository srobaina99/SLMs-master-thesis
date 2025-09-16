# Tutores y LLMs
### Harvard AI tutor for physics
https://news.harvard.edu/gazette/story/2024/09/professor-tailored-ai-tutor-to-physics-course-engagement-doubled/

[[Harvard AI tutor for physics]]

### ChatGPT impacts student engagement review
[[ChatGPT impacts student engagement review]]


### Learning from human tutoring
https://onlinelibrary.wiley.com/doi/pdfdirect/10.1207/s15516709cog2504_1

Important idea: Tutors aren't necessarily trained specifically for that purpose but anyway its useful if the tutor knows the content.
*"A third important but again often overlooked finding is that tutors often do not have formal training in the skills of tutoring"*


### Tutor_CoPilot
[[Tutor_CoPilot.pdf]]


## AI applications in English as a Foreign Language
[[AI_Applications_in_EFL.pdf]]
The discussion culminates by identifying research gaps
- *"In the field of education, AI-powered technologies surpass human teachers in the ability to collate enormous volumes of data on learners’ academic achievement"*
- *"They facilitate learning growth by providing quick feedback"*
- *"In the context of language learning in particular, AI supports advancement in various ways, including the provision of customized feedback, adjustable educational pathways, sophisticated tutoring systems, and natural language processing tools."*

## Large Language Models for Education: A Survey
https://arxiv.org/html/2405.13001v1
Might incentive cognitive laziness 

## LLM-as-a-tutor in EFL Writing Education
[[AI_writing_tutor_EFL.pdf]]
Rubric based scores

# SLMs

## SLM survey
https://arxiv.org/pdf/2501.05465v1
[Presentacion grupo PLN](https://docs.google.com/presentation/d/1fDaeC1F8cMcnpik34uoRZxRfxLRx1Tp7-v33rTxGuQo/edit?slide=id.p#slide=id.p)
## Modelos chicos en CPU
https://www.arcee.ai/blog/is-running-language-models-on-cpu-really-viable


## SLM Agents
### TinyAgent
https://aclanthology.org/2024.emnlp-demo.9.pdf
Recent large language models (LLMs) have enabled the development of advanced agentic systems that can integrate various tools and APIs to fulfill user queries through function calling. However, the deployment of these LLMs on the edge has not been explored since they typically require cloud-based infrastructure due to their substantial model size and computational demands. To this end, we present TinyAgent, an end-to-end framework for training and deploying task-specific small language model agents capable of function calling for driving agentic systems at the edge. We first show how to enable accurate function calling for open-source models via the LLMCompiler framework. We then systematically curate a high-quality dataset for function calling, which we use to fine-tune two small language models, TinyAgent-1.1B and 7B. For efficient inference, we introduce a novel tool retrieval method to reduce the input prompt length and utilize quantization to further accelerate the inference speed. As a driving application, we demonstrate a local Siri-like system for Apple’s MacBook that can execute user commands through text or voice input. Our results show that our models can achieve, and even surpass, the function-calling capabilities of larger models like GPT-4-Turbo, while being fully deployed at the edge. We open-source our dataset, models, and installable package1 and provide a demo video for our MacBook assistant agent2
## Modelos SLM
### Qwen
https://arxiv.org/pdf/2309.16609
Large language models (LLMs) have revolutionized the field of artificial intelligence, enabling natural language processing tasks that were previously thought to be exclusive to humans. In this work, we introduce QWEN1 , the first installment of our large language model series. QWEN is a comprehensive language model series that encompasses distinct models with varying parameter counts. It includes QWEN, the base pretrained language models, and QWEN-CHAT, the chat models finetuned with human alignment techniques. The base language models consistently demonstrate superior performance across a multitude of downstream tasks, and the chat models, particularly those trained using Reinforcement Learning from Human Feedback (RLHF), are highly competitive. The chat models possess advanced tool-use and planning capabilities for creating agent applications, showcasing impressive performance even when compared to bigger models on complex tasks like utilizing a code interpreter. Furthermore, we have developed coding-specialized models, CODE-QWEN and CODE-QWEN-CHAT, as well as mathematics-focused models, MATH-QWEN-CHAT, which are built upon base language models. These models demonstrate significantly improved performance in comparison with open-source models, and slightly fall behind the proprietary models.
### Gemma
https://arxiv.org/pdf/2403.08295
This work introduces Gemma, a family of lightweight, state-of-the art open models built from the research and technology used to create Gemini models. Gemma models demonstrate strong performance across academic benchmarks for language understanding, reasoning, and safety. We release two sizes of models (2 billion and 7 billion parameters), and provide both pretrained and fine-tuned checkpoints. Gemma outperforms similarly sized open models on 11 out of 18 text-based tasks, and we present comprehensive evaluations of safety and responsibility aspects of the models, alongside a detailed description of model development. We believe the responsible release of LLMs is critical for improving the safety of frontier models, and for enabling the next wave of LLM innovations.

### BioGPT
https://arxiv.org/pdf/2305.07804
Large Language Models (LLMs) have made remarkable advancements in the field of natural language processing. However, their increasing size poses challenges in terms of computational cost. On the other hand, Small Language Models (SLMs) are known for their efficiency, but they often struggle with limited capacity and training data, especially in specific domains. In this paper, we introduce a novel method aimed at improving SLMs in the medical domain using LLM-based generative data augmentation. The objective of our approach is to develop more efficient and capable models that are specifically tailored for specialized applications. Through experiments conducted on the PubMedQA dataset, we demonstrate the effectiveness of LLMs in refining and diversifying existing question-answer pairs. This refinement process leads to improved performance in a significantly smaller model after fine-tuning. Notably, our best SLM, with under 1.6 billion parameters, outperforms the few-shot GPT-4 on the PubMedQA dataset. Our code and generated data are publicly available to facilitate further explorations

### Orca (1, 2)
https://arxiv.org/pdf/2306.02707
Recent research has focused on enhancing the capability of smaller models through imitation learning, drawing on the outputs generated by large foundation models (LFMs). A number of issues impact the quality of these models, ranging from limited imitation signals from shallow LFM outputs; small scale homogeneous training data; and most notably a lack of rigorous evaluation resulting in overestimating the small model’s capability as they tend to learn to imitate the style, but not the reasoning process of LFMs. To address these challenges, we develop Orca, a 13-billion parameter model that learns to imitate the reasoning process of LFMs. Orca learns from rich signals from GPT-4 including explanation traces; step-by-step thought processes; and other complex instructions, guided by teacher assistance from ChatGPT. To promote this progressive learning, we tap into large-scale and diverse imitation data with judicious sampling and selection. Orca surpasses conventional state-of-the-art instruction-tuned models such as Vicuna-13B by more than 100% in complex zero-shot reasoning benchmarks like BigBench Hard (BBH) and 42% on AGIEval. Moreover, Orca reaches parity with ChatGPT on the BBH benchmark and shows competitive performance (4 pts gap with optimized system message) in professional and academic examinations like the SAT, LSAT, GRE, and GMAT, both in zero-shot settings without CoT; while trailing behind GPT-4. Our research indicates that learning from step-by-step explanations, whether these are generated by humans or more advanced AI models, is a promising direction to improve model capabilities and skills.
https://arxiv.org/pdf/2311.11045
Orca 1 learns from rich signals, such as explanation traces, allowing it to outperform conventional instruction-tuned models on benchmarks like BigBench Hard and AGIEval. In Orca 2, we continue exploring how improved training signals can enhance smaller LMs’ reasoning abilities. Research on training small LMs has often relied on imitation learning to replicate the output of more capable models. We contend that excessive emphasis on imitation may restrict the potential of smaller models. We seek to teach small LMs to employ different solution strategies for different tasks, potentially different from the one used by the larger model. For example, while larger models might provide a direct answer to a complex task, smaller models may not have the same capacity. In Orca 2, we teach the model various reasoning techniques (step-by-step, recall then generate, recall-reason-generate, direct answer, etc.). Moreover, we aim to help the model learn to determine the most effective solution strategy for each task. We evaluate Orca 2 using a comprehensive set of 15 diverse benchmarks (corresponding to approximately 100 tasks and over 36K unique prompts). Orca 2 significantly surpasses models of similar size and attains performance levels similar or better to those of models 5-10x larger, as assessed on complex tasks that test advanced reasoning abilities in zero-shot settings. We make Orca 2 weights publicly available at aka.ms/orca-lm to support research on the development, evaluation, and alignment of smaller LMs.

### Mixtral of experts
https://arxiv.org/pdf/2401.04088
We introduce Mixtral 8x7B, a Sparse Mixture of Experts (SMoE) language model. Mixtral has the same architecture as Mistral 7B, with the difference that each layer is composed of 8 feedforward blocks (i.e. experts). For every token, at each layer, a router network selects two experts to process the current state and combine their outputs. Even though each token only sees two experts, the selected experts can be different at each timestep. As a result, each token has access to 47B parameters, but only uses 13B active parameters during inference. Mixtral was trained with a context size of 32k tokens and it outperforms or matches Llama 2 70B and GPT-3.5 across all evaluated benchmarks. In particular, Mixtral vastly outperforms Llama 2 70B on mathematics, code generation, and multilingual benchmarks. We also provide a model finetuned to follow instructions, Mixtral 8x7B – Instruct, that surpasses GPT-3.5 Turbo, Claude-2.1, Gemini Pro, and Llama 2 70B – chat model on human benchmarks. Both the base and instruct models are released under the Apache 2.0 license. Code: https://github.com/mistralai/mistral-src Webpage: https://mistral.ai/news/mixtral-of-experts/
### Phi 1
https://arxiv.org/pdf/2306.11644
We introduce phi-1, a new large language model for code, with significantly smaller size than competing models: phi-1 is a Transformer-based model with 1.3B parameters, trained for 4 days on 8 A100s, using a selection of “textbook quality” data from the web (6B tokens) and synthetically generated textbooks and exercises with GPT-3.5 (1B tokens). Despite this small scale, phi-1 attains pass@1 accuracy 50.6% on HumanEval and 55.5% on MBPP. It also displays surprising emergent properties compared to phi-1-base, our model before our finetuning stage on a dataset of coding exercises, and phi-1-small, a smaller model with 350M parameters trained with the same pipeline as phi-1 that still achieves 45% on HumanEval.

### Tiny Stories
https://arxiv.org/pdf/2305.07759v2
Language models[4, 5, 21] (LMs) are powerful tools for natural language processing, but they often struggle to produce coherent and fluent text when they are small. Models with around 125M parameters such as GPTNeo (small) [3] or GPT-2 (small) [23] can rarely generate coherent and consistent English text beyond a few words even after extensive training. This raises the question of whether the emergence of the ability to produce coherent English text only occurs at larger scales (with hundreds of millions of parameters or more) and complex architectures (with many layers of global attention). In this work, we introduce TinyStories, a synthetic dataset of short stories that only contain words that a typical 3 to 4-year-olds usually understand, generated by GPT-3.5 and GPT-4. We show that TinyStories can be used to train and evaluate LMs that are much smaller than the state-of-the-art models (below 10 million total parameters), or have much simpler architectures (with only one transformer block), yet still produce fluent and consistent stories with several paragraphs that are diverse and have almost perfect grammar, and demonstrate reasoning capabilities. We also introduce a new paradigm for the evaluation of language models: We suggest a framework which uses GPT-4 to grade the content generated by these models as if those were stories written by students and graded by a (human) teacher. This new paradigm overcomes the flaws of standard benchmarks which often require the model’s output to be very structured, and moreover it provides a multidimensional score for the model, providing scores for different capabilities such as grammar, creativity and instruction-following. We hope that TinyStories can facilitate the development, analysis and research of LMs, especially for lowresource or specialized domains, and shed light on the emergence of language capabilities in LMs.