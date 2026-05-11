```
============================================================================ 
LREC 2026 Reviews for Submission #1461
============================================================================ 

Title: Inference-Time Complexity Control for Small Language Models in Educational Applications
Authors: Santiago Robaina, Aiala Rosá and Luis Chiruzzo
============================================================================
                            META-REVIEW
============================================================================ 

Comments: This paper investigates inference-time strategies for controlling linguistic complexity in Small Language Models, with a focus on generating A1-level English suitable for language learning applications.

Reviewers agree that the topic is timely and relevant. However, they also raise several important concerns that limit the strength of the contribution. These concerns relate to the absence of key evaluation and analysis components that are essential in work on linguistic complexity control for text generation. In particular, reviewers note that the evaluation relies exclusively on traditional readability metrics, whose reliability on short generated texts is questionable, and that no human evaluation, fluency assessment, or error analysis is provided. In addition, methodological details, especially regarding the implementation of vocabulary probability weighting, require clearer description, and some claims would benefit from more cautious phrasing.

Considering these substantial weaknesses consistently highlighted by all reviewers, I believe that the paper is not ready for acceptance at LREC in its current form.

============================================================================
                            REVIEWER #1
============================================================================

---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                               Relevance: 3
                  Knowledge of the Field: 3
                               Soundness: 3
                                 Clarity: 4
             Originality of the Approach: 3
                 Significance of Results: 3
                           Replicability: Yes
                      Overall Assessment: 3
                     Reviewer Confidence: 4

Detailed Comments
---------------------------------------------------------------------------
Abstract seems to be an exposition of results rather than providing a means to appreciate importance, contribution etc. 
The opening example seems to have a key semantic difference â€“ the first definition relates a collection of items but the second relates a building, and identifies toys as a priority over kept stories (not books), and no other reading materials. Could the intervention be considered as potentially confusing? There seems to be a relegation of importance of this to Limitations, and upfront discussion would be merited.  
It should be noted that there are various commonalities in the measures used, and so what appears to be being measured is how each LLM reacts to a prompt. Counts of sentence lengths, polysyllabic words, and other input features would offer greater insight into results being produced and hence where most gain was being achieved â€“ and note e.g. W/S = ASL (GF vs FK / Spache); consider, for example, that a different model may have a different number range relating a fuzzy linguistic variable such as "short". It would help to have target values related by the various tables such that distance to target is always clear â€“ that only seems to exist for the figure.
It should also be noted that readability measures have been variously posed as unreliable with short passages â€“ SMOG was formulated with respect to selecting 30 sentences: "By a process of trial and error I have found that 30 sentences is a suitable number for the criterion of readability used here. Other readability prediction systems invite one to use samples of only 100 words. Such a sample is so small that it may be quite uncharacteristic of the text being assessed. To get a more reliable prediction, you have to calculate the reading difficulty of several samples and then average them".
Authors would be advised to go deeper into how results are being obtained as well as what the measures used are telling them about these.
---------------------------------------------------------------------------


---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                          Ethical Issues: No


============================================================================
                            REVIEWER #2
============================================================================

---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                               Relevance: 4
                  Knowledge of the Field: 3
                               Soundness: 3
                                 Clarity: 2
             Originality of the Approach: 2
                 Significance of Results: 2
                           Replicability: No
                      Overall Assessment: 2
                     Reviewer Confidence: 4

Detailed Comments
---------------------------------------------------------------------------
The paper presents two strategies to adapt text generated by small language models for A1 English readers. The first strategy optimizes the prompt using simplification rules (contextual prompting), while the second limits and adjusts the output vocabulary of the models (vocabulary probability weighting). The combined interventions were tested on four language models across 15 prompt trials. Outcomes were evaluated using four automatic text-complexity metrics. Results show that prompt intervention is the most effective strategy, and that vocabulary probability weighting provides only slight improvements when combined with it. The results also vary significantly depending on the model used.

Strengths:

The study has strong potential for educational applications.

Weaknesses:

- The number of trials is too small (only 15), and the criteria for selecting the prompts are not clearly explained.
- Although the study uses four text-complexity metrics, they are outdated. Given the dataset size, a human evaluation and an error analysis would help extract more insights.
- It is also unclear how much the strategies affect the fluency of the responses, and an analysis of response fluency is missing.
---------------------------------------------------------------------------


---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                          Ethical Issues: No


============================================================================
                            REVIEWER #3
============================================================================

---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                               Relevance: 4
                  Knowledge of the Field: 4
                               Soundness: 3
                                 Clarity: 4
             Originality of the Approach: 2
                 Significance of Results: 2
                           Replicability: No
                      Overall Assessment: 3
                     Reviewer Confidence: 4

Detailed Comments
---------------------------------------------------------------------------
Briefly describe what the submission is about: 
The paper study two inference strategies to generate answers targeting A1-level English with Small Language Models (0.5B-3.8B parameters).

Contributions:
Two inference strategies with linguistic control are compared to generate A1-level English sentences. A first one relies on probability boosting for words of a short vocabulary list. A second one relies on the use of prompt that specifies audience, lexical selection and sentence structure directive.

Strengths:
The principles of the two approaches are clear.
They are tested using four different readability metrics and complemented with word counts and difficult words Counts.

Weaknesses:
The implementation of the weighting bias for the starters list's words is not clearly described.
Although the results differ depending on the LM used, the weighting approach seems to be less effective than the prompting one. You should provide the results for prompting only for each LM, by adding for example a specific column for this setting in Table 4.
The evaluation does not consider syntactic levels (explicitly mentioned in the prompt) or more importantly, faithfulness.




Minor comments:
In the Introduction, the "Control" tag is not clear. You only explain it in section 3.1.
2.3: Why couldn't the decoding-time methods (DExperts, FUDGE) target linguistic complexity?
3.2.1: I would not use "real-time", which is incorrectly used as a synonymous as "on-line".
3.2.3: the introduction of "young" differs from the general "A1-level English learners" target mentioned in the abstract. From the example of the introduction, it can be seen that the generation is childish, which may be improper for some A1-level English learners.
5.2: "Standalone vocabulary manipulation is insufficient for complexity control in most architectures." Even if the weighting approach has a different effect depending on the LM, the results show that this method is not sufficient without a specific prompting.
---------------------------------------------------------------------------


---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                          Ethical Issues: No
```
