============================================================================ 
BEA 2026 Reviews for Submission #156
============================================================================ 

Title: Feature-Enriched mDeBERTa for Word Difficulty Prediction
Authors: Santiago Robaina, Aiala Rosá and Luis Chiruzzo


============================================================================
                            REVIEWER #1
============================================================================

---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                   Appropriateness (1-5): 5
                           Clarity (1-5): 5
      Originality / Innovativeness (1-5): 4
           Soundness / Correctness (1-5): 3
             Meaningful Comparison (1-5): 5
                      Thoroughness (1-5): 5
        Impact of Ideas or Results (1-5): 4
                    Recommendation (1-5): 5
               Reviewer Confidence (1-5): 3

Detailed Comments
---------------------------------------------------------------------------
This paper is very interesting, particularly the numerous manual features proposed (which I had not thought of!). The information provided seems sufficiently precise to me and the text is clear.

I have only one general remark. I am not as convinced as the authors by the arguments put forward to support the following claim in the abstract and the text: 
"First, a LaBSE (Language-Agnostic BERT Sentence Embedding) cross-lingual cosine between the L1 source word and the English target word is the single highest-impact feature in our experiments."
"3.3 Observations Table 1 shows the adding of features to our feature set. The largest single-feature addition is embed_cosine, which drops average dev RMSE by 0.091 (row 5 â†’ row 6). The effect is largest on Mandarin (CN dev RMSE 1.257 â†’ 1.181, âˆ’0.076), which is consistent with CN having no meaningful orthographic signal between the L1 and English; LaBSE provides the missing cross-lingual bridge. On ES and DE, embed_cosine still gives the largest single drop, alongside target-word Zipf frequency and cross-lingual Jaroâ€“Winkler similarity, which exploit cognate structure."
The improvement achieved by feature SET 5 ("+ Jaroâ€“Winkler + n-gram + Zipf freq.") is significantly greater than that achieved by SINGLE feature 6 (+ embed_cosine (LaBSE)). However, it is unclear whether any of the features in SET 5 could perform better than the SINGLE feature 6. It follows that the claim that single feature 6 is the most useful single feature is correct on the basis of the analyses carried out, but further analyses are needed and so it is clearly not proven.
I have noted the comment: "We do not report a drop-one ablation of embed_cosine from the full v2 feature set; the claim that it is the highest-impact feature is based on incremental-addition evidence only." However, the most likely consequence seems to me to be that feature set 5 is favoured over feature 6, since what is common is attributed to the first one introduced. 
In sum, the reader does not know whether any of the single features in set 5 could be better than single feature 6 (and would still be so if it were introduced later into the model). Would it be possible to qualify this paragraph and its main conclusion by using part of the additional page allocated for the review?

Some elements of the challenge are mentioned before being presented in the text such as "closed-track" ("The closed-track baseline is a fine-tuned XLM-RoBERTa-base (Conneau et al., 2019).") et "dev" ("and Â§5 reports and discusses dev and official-test results.")

"The progression in Table 1 uses linear regression for the first three feature sets and XGBoost for the rest (n_estimators=300, max_depth=5, learning_rate=0.05)."
> Could you explain why this is the case and whether it has any impact?

I am not sure I understand this: "Each individual seed already beats the XLM-RoBERTa-base baseline on ES dev (Table 2); the ensemble is a âˆ¼0.02â€“0.03 dev RMSE refinement on top of that, not the source of the gap to the baseline."
Compared to certain seeds, the gain is 0.07.

Finally, I think a few sentences could be improved, but my native language is not English. 

"Two findings we highlight." 
> " We highlight two findings." or something like "Two findings deserve special attention."

"is a core problem for intelligent tutoring and for adaptive reading and vocabulary tools."
> "is a core problem for intelligent tutoring, adaptive reading and vocabulary tools."

"(Spanish, German, Mandarin)" 
> "(Spanish, German and Mandarin)"

"We extract a set of handcrafted features per item, grouped by type rather than by order of addition."
> "We extract a set of handcrafted features per item, grouped ?here/in this section? by type rather than by order of addition."

"This is the single feature that most reduces RMSE in our ablations and is the main narrative thread of this paper."
> "This is the single feature that most reduces RMSE in our ablation experiments and is the main narrative thread of this paper."
---------------------------------------------------------------------------



============================================================================
                            REVIEWER #2
============================================================================

---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                   Appropriateness (1-5): 5
                           Clarity (1-5): 5
      Originality / Innovativeness (1-5): 5
           Soundness / Correctness (1-5): 5
             Meaningful Comparison (1-5): 5
                      Thoroughness (1-5): 5
        Impact of Ideas or Results (1-5): 5
                    Recommendation (1-5): 5
               Reviewer Confidence (1-5): 5

Detailed Comments
---------------------------------------------------------------------------
This paper is very clearly written and the methodology and results are easy to follow.

I have just a couple of notes regarding some of the data in the tables:

* Table 1: It seems that feature set 5 (Jaroâ€“Winkler + n-gram + Zipf freq.) has the biggest impact on RMSE (âˆ’0.151), rather than feature set 6. Perhaps this is a typo in the results?
* Table 2/Results and Analysis paragraph 1: It looks like the feature-only XGBoost model does not beat the dev baseline for DE.
---------------------------------------------------------------------------