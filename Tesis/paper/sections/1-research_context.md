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
