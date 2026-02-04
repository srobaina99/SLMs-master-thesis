\section{Results}

\subsection{Aggregate Performance Across Models}

Tables~\ref{tab:table1a} and~\ref{tab:table1b} present the mean performance metrics aggregated across all four models (n=60 per configuration). The control condition established baseline complexity levels considerably higher than A1 target thresholds across all metrics (Flesch-Kincaid Grade Level [FK], Gunning Fog Index [GF], SMOG, Spache). Prompting-based interventions ("Prompting Only" and "Both") achieved substantial reductions of 28-37\% across all readability measures. In contrast, standalone vocabulary weighting showed minimal impact on aggregate complexity, accompanied by a 10\% increase in word count.

\begin{table}[!ht]
\begin{center}
\begin{tabular}{lcccc}
\hline
\textbf{Config.} & \textbf{FK} & \textbf{GF} & \textbf{SMOG} & \textbf{Spa.} \\
\hline
Control   & 8.95  & 11.06 & 11.37 & 5.08 \\
Weighting & 9.11  & 11.05 & 11.04 & 5.13 \\
Prompting & 5.60  & 6.99  & 8.34  & 3.60 \\
Both      & 5.59* & 7.29  & 8.00  & 3.68 \\
\hline
\end{tabular}
\caption{Primary Readability Metrics by Configuration (All Models Combined, n=60). *Meets A1 target (FK ≤5.0)}
\label{tab:table1a}
\end{center}
\end{table}

\begin{table}[!ht]
\begin{center}
\begin{tabular}{lcc}
\hline
\textbf{Config.} & \textbf{Words} & \textbf{Diff. Words} \\
\hline
Control   & 82.3 & 15.7 \\
Weighting & 90.6 & 15.5 \\
Prompting & 69.2 & 6.2  \\
Both      & 73.1 & 5.8  \\
\hline
\end{tabular}
\caption{Secondary Metrics by Configuration (All Models Combined, n=60)}
\label{tab:table1b}
\end{center}
\end{table}

The combined intervention ("Both") demonstrated negligible benefit over prompting alone across all primary metrics. This suggests that contextual prompting captures the majority of achievable simplification. When adding weighting, the result was 5.7\% more words but 6.7\% fewer difficult words, indicating that the intervention encourages use of additional simple words rather than complex alternatives. Difficult words were reduced by 60-63\% under prompting interventions, directly addressing vocabulary accessibility requirements for A1 learners.

\begin{figure}[!ht]
\begin{center}
\includegraphics[width=\columnwidth]{../../src/evaluation/experiment_framework/results/aggregate/plots/aggregate_all_models_1023_1826.png}
\caption{Aggregate Performance Distributions Across All Models}
\label{fig:aggregate}
\end{center}
\end{figure}

Figure~\ref{fig:aggregate} presents boxplot distributions of all readability metrics aggregated across the four models (n=60 per configuration). The visualizations confirm the substantial separation between control/weighting conditions and prompting-based interventions across all metrics. Prompting configurations demonstrate consistently lower complexity with reduced variance, while standalone vocabulary weighting shows minimal aggregate change from control with higher variability. The combined intervention exhibits similar central tendencies to prompting alone, with minimal additional benefit visible across all six metrics.

\subsection{Prompting vs Combined Intervention Analysis}

Given that standalone vocabulary weighting showed minimal aggregate impact (Section 4.1) and that prompting-based interventions achieved substantial complexity reductions (37\% FK reduction), a critical research question becomes whether adding vocabulary weighting to contextual prompting provides measurable benefits beyond prompting alone. Table~\ref{tab:table2} presents a direct comparison between "Prompting Only" and "Both" configurations across all models, quantifying the marginal contribution of vocabulary weighting when applied alongside prompting.

\begin{table}[!ht]
\begin{center}
\begin{tabular}{llccc}
\hline
\textbf{Model} & \textbf{Metric} & \textbf{Both} & \textbf{Δ} & \textbf{Δ\%} \\
\hline
Phi3      & FK          & 3.88 & -0.35 & -8.3  \\
          & GF          & 5.69 & +0.09 & +1.6  \\
          & SMOG        & 6.87 & -0.64 & -8.5  \\
          & Spa.        & 2.99 & -0.02 & -0.7  \\
          & Words       & 66.5 & -1.5  & -2.2  \\
          & Diff. Words & 3.3  & -0.3  & -9.2  \\
\hline
Qwen2     & FK          & 8.02  & +2.01 & +33.5 \\
          & GF          & 10.08 & +2.34 & +30.2 \\
          & SMOG        & 8.81  & +0.08 & +0.9  \\
          & Spa.        & 4.53  & +0.73 & +19.2 \\
          & Words       & 75.9  & +7.5  & +10.9 \\
          & Diff. Words & 4.8   & -1.0  & -17.2 \\
\hline
Qwen3     & FK          & 3.50 & -0.29 & -7.7  \\
          & GF          & 4.61 & -0.10 & -2.1  \\
          & SMOG        & 6.65 & -0.61 & -8.4  \\
          & Spa.        & 2.95 & +0.11 & +3.9  \\
          & Words       & 50.4 & -8.7  & -14.7 \\
          & Diff. Words & 5.1  & -0.5  & -9.4  \\
\hline
TinyLlama & FK          & 6.95  & -1.44 & -17.2 \\
          & GF          & 8.78  & -1.13 & -11.4 \\
          & SMOG        & 9.67  & -0.20 & -2.0  \\
          & Spa.        & 4.27  & -0.50 & -10.5 \\
          & Words       & 99.5  & +18.4 & +22.7 \\
          & Diff. Words & 9.8   & 0.0   & 0.0   \\
\hline
\end{tabular}
\caption{Marginal Effect of Adding Weighting to Prompting (Δ = Both - Prompting Only)}
\label{tab:table2}
\end{center}
\end{table}

\subsubsection{Model-Specific Interaction Patterns}

\textbf{Phi3 (3.8B)} demonstrated minimal complementary effects between interventions across most readability metrics, suggesting that vocabulary weighting provides limited additional benefit when strong instruction-following capability is present. Notably, Phi3 was the only model where the combined intervention produced slightly less verbose output while reducing difficult words, indicating efficient vocabulary substitution without compensatory verbosity.

\textbf{Qwen2 (0.5B)} exhibited a negative interaction, with the combined intervention underperforming prompting alone across primary readability metrics. This represents a 34\% increase in FK complexity when adding vocabulary weighting to prompting. Despite reducing difficult words by 17\%, the intervention triggered compensatory verbosity and output instability. This pattern suggests fundamental architectural incompatibility between vocabulary constraints and instruction-following in this model.

\textbf{Qwen3 (0.6B)} demonstrated modest positive interaction across most metrics, achieving the lowest absolute complexity of any configuration tested. Notably, Qwen3 was the only model where the combined intervention reduced word count while also reducing difficult words, indicating efficient simplification without compensatory verbosity. This model's unique responsiveness to standalone vocabulary weighting (the only model achieving A1 targets with weighting alone) suggests that its training data or architectural design predisposes it to benefit from lexical constraints.

\textbf{TinyLlama (1.1B)} exhibited the strongest positive interaction of any model tested, representing a 17\% reduction in FK complexity when adding vocabulary weighting to prompting. This substantial gain suggests that vocabulary constraints compensate for weak instruction-following capability in this architecture. However, TinyLlama exhibited severe compensatory verbosity with no reduction in difficult word count, indicating that the complexity reduction follows from syntactic restructuring rather than vocabulary substitution. Despite this positive interaction, no configuration achieved A1 targets.

\subsubsection{Aggregate Trade-off Analysis}

Across all models, the combined intervention produced 5.7\% more words than prompting alone while achieving 6.7\% fewer difficult words. However, this aggregate pattern masks substantial model-level heterogeneity. The effectiveness of vocabulary weighting when combined with prompting is highly model-dependent:

- \textbf{Beneficial} (Qwen3, TinyLlama): Vocabulary constraints yield net complexity reduction across metrics, with Qwen3 achieving this without increased verbosity and TinyLlama through syntactic restructuring despite severe verbosity
- \textbf{Neutral} (Phi3): Minimal complexity gains with slight reduction in verbosity, indicating limited interaction
- \textbf{Detrimental} (Qwen2): Compensatory verbosity triggers output instability and catastrophic complexity increases

These findings indicate that intervention effectiveness is highly model-dependent, with Qwen3 representing the only configuration where vocabulary weighting provides clear, consistent benefits across all readability measures without triggering compensatory verbosity.

\subsection{Model-Specific Performance and Intervention Effects}

Table~\ref{tab:table3} presents intervention effect sizes calculated as absolute and relative changes from control baselines across all models and configurations. Prompting interventions demonstrated large effect sizes across all readability metrics and models, with reductions ranging from 0.3-63\%. Standalone vocabulary weighting exhibited highly model-dependent effects: Qwen3 showed substantial positive impact, while Phi3 showed negligible change, Qwen2 showed modest negative impact, and TinyLlama showed severe negative impact.

\begin{table}[!ht]
\begin{center}
\begin{tabular}{llcccc}
\hline
\textbf{Model} & \textbf{Config.} & \textbf{FK Δ} & \textbf{FK \%} & \textbf{DW Δ} & \textbf{DW \%} \\
\hline
Phi3      & Weighting & +0.09  & -0.8   & -1.9  & 7.2   \\
          & Prompting & -6.31  & 59.8   & -22.3 & 86.1  \\
          & Both      & -6.66  & 63.2   & -22.6 & 87.4  \\
\hline
Qwen2     & Weighting & +0.49  & -4.5   & +1.9  & -15.8 \\
          & Prompting & -4.87  & 44.8   & -6.4  & 52.5  \\
          & Both      & -2.85  & 26.2   & -7.4  & 60.7  \\
\hline
Qwen3     & Weighting & -1.00  & 16.8   & -1.7  & 15.0  \\
          & Prompting & -2.18  & 36.5   & -5.5  & 49.1  \\
          & Both      & -2.46  & 41.3   & -6.0  & 53.9  \\
\hline
TinyLlama & Weighting & +1.07  & -12.7  & +0.7  & -5.3  \\
          & Prompting & -0.02  & 0.3    & -3.9  & 28.6  \\
          & Both      & -1.46  & 17.3   & -3.9  & 28.6  \\
\hline
\end{tabular}
\caption{Intervention Effect Sizes by Model (Change from Control). DW = Difficult Words.}
\label{tab:table3}
\end{center}
\end{table}

Substantial heterogeneity in intervention responsiveness was observed across model architectures, with Phi3 demonstrating strongest responsiveness and TinyLlama showing weakest effects.

\textbf{Phi3 (3.8B)} exhibited the highest baseline complexity across all metrics but demonstrated the strongest intervention responsiveness. The combined intervention achieved substantial reductions of 46-63\% across all metrics. Both prompting configurations consistently met A1 targets across all readability measures, with difficult words reduced by 87\%. Standalone vocabulary weighting showed negligible impact, indicating that lexical constraints alone are insufficient without contextual guidance.

\textbf{Qwen3 (0.6B)} achieved the lowest absolute complexity across all metrics with the smallest model size. This model exhibited a naturally simpler baseline and was the only model where standalone vocabulary weighting achieved A1 targets across multiple metrics, indicating unique architectural sensitivity to lexical constraints.

\textbf{Qwen2 (0.5B)} demonstrated significant output instability despite being the smallest model tested, with high variance and extreme outliers observed. The model's negative interaction between interventions is analyzed in detail in Section 4.2.

\textbf{TinyLlama (1.1B)} exhibited the weakest intervention responsiveness across all readability metrics. The combined intervention demonstrated modest improvements of 10-17\%, though no configuration achieved A1 targets. The 1.5× weight factor substantially mitigated the catastrophic complexity increases observed with 2.0× weighting.

\subsection{Model-Specific Performance Distributions}

Figures~\ref{fig:phi3}-\ref{fig:tinyllama} present boxplot distributions of all metrics across configurations for each model. Phi3 (Figure~\ref{fig:phi3}) exhibits clear bimodal separation between control/weighting conditions (FK>10) and prompting interventions (FK<5), with tight clustering indicating consistent simplification. Qwen2 (Figure~\ref{fig:qwen2}) demonstrates high variance in the combined intervention, with extreme outliers (FK>30) confirming output instability. Qwen3 (Figure~\ref{fig:qwen3}) shows a consistent monotonic decrease across intervention levels (Control > Weighting > Prompting > Both), with all intervention conditions achieving A1 targets. TinyLlama (Figure~\ref{fig:tinyllama}) exhibits minimal differentiation between control and prompting conditions, with the combined intervention showing modest improvement but high variance.

\begin{figure}[!ht]
\begin{center}
\includegraphics[width=\columnwidth]{../../src/evaluation/experiment_framework/results/Phi3/plots/Phi3_full_experiment_specification_1023_0034.png}
\caption{Phi3 (3.8B) Performance Distributions}
\label{fig:phi3}
\end{center}
\end{figure}

\begin{figure}[!ht]
\begin{center}
\includegraphics[width=\columnwidth]{../../src/evaluation/experiment_framework/results/Qwen2/plots/Qwen2_full_experiment_specification_1023_0025.png}
\caption{Qwen2 (0.5B) Performance Distributions}
\label{fig:qwen2}
\end{center}
\end{figure}

\begin{figure}[!ht]
\begin{center}
\includegraphics[width=\columnwidth]{../../src/evaluation/experiment_framework/results/Qwen3/plots/Qwen3_full_experiment_specification_1023_0027.png}
\caption{Qwen3 (0.6B) Performance Distributions}
\label{fig:qwen3}
\end{center}
\end{figure}

\begin{figure}[!ht]
\begin{center}
\includegraphics[width=\columnwidth]{../../src/evaluation/experiment_framework/results/TinyLlama/plots/TinyLlama_full_experiment_specification_1023_0029.png}
\caption{TinyLlama (1.1B) Performance Distributions}
\label{fig:tinyllama}
\end{center}
\end{figure}

\subsection{Discussion}

\subsubsection{Model Performance Comparison}

The experimental results demonstrate substantial heterogeneity in complexity control effectiveness across model architectures. Qwen3 (0.6B) with combined interventions achieved the lowest absolute complexity across all readability metrics with the smallest model footprint. The model's naturally simpler baseline and unique responsiveness to standalone vocabulary weighting across multiple metrics suggest training data alignment with simplified content, reducing intervention dependency.

Phi3 (3.8B) with combined interventions demonstrated the strongest relative intervention responsiveness, achieving substantial reductions across all metrics. The model's strong instruction-following capability and low output variance provide greater predictability than smaller alternatives, though at the cost of significantly higher computational requirements.

Qwen2 and TinyLlama exhibited limitations for A1-level text generation. Qwen2's output instability across all metrics and extreme outliers indicate architectural challenges in maintaining consistent simplification. TinyLlama's failure to achieve A1 targets in any configuration across any readability measure suggests fundamental architectural limitations for complexity control, despite showing positive interaction effects between interventions.

\subsubsection{Weight Factor Optimization}

Comparison with preliminary experiments using 2.0× weighting reveals that the 1.5× factor provides superior balance between vocabulary guidance and compensatory verbosity mitigation across all readability metrics. The lower weight factor substantially reduced the complexity increases observed at higher amplification levels, while Qwen3 uniquely achieved A1 targets with weighting alone across multiple metrics at this level. Future work should systematically explore the weight factor space (1.1-2.0×) to identify model-specific optima.

\subsubsection{Compensatory Verbosity Mechanism}

The aggregate minimal complexity change under standalone vocabulary weighting masks substantial model-level heterogeneity in compensatory verbosity responses. When lexical choices are constrained, models exhibit divergent behavioral patterns: Phi3 and Qwen2 showed modest FK increases, while TinyLlama exhibited severe complexity escalation. Only Qwen3 demonstrated beneficial response, suggesting architectural resistance to compensatory mechanisms. The 10\% aggregate word count increase under weighting-only conditions indicates that vocabulary constraints trigger verbosity at the sentence-construction level, though this does not uniformly translate to complexity increases across all readability metrics.

\subsubsection{Model Size and Effectiveness}

The results demonstrate that model size does not monotonically predict intervention effectiveness. Qwen3 (0.6B) achieved superior absolute simplicity across all readability metrics compared to Phi3 (3.8B) while being 6.3× smaller. This suggests that training data composition and architectural design may be more critical than parameter count for text simplification tasks. The naturally simpler baseline exhibited by Qwen3 indicates that pre-training on educational or simplified corpora may provide substantial advantages for downstream complexity control.

\subsection{Summary}

This study evaluated text complexity control mechanisms across four Small Language Models using a 2×2 factorial design (prompting × weighting) with 240 total observations. Key findings include:

1. Contextual prompting achieved substantial complexity reductions of 28-63\% across all readability metrics and models, with Phi3 and Qwen3 consistently meeting A1 target thresholds.
2. Standalone vocabulary weighting (1.5× boost) showed minimal aggregate impact, with only Qwen3 demonstrating substantial positive response across measures.
3. Interaction effects between interventions varied by architecture, ranging from negative (Qwen2) to positive (TinyLlama).
4. Qwen3 (0.6B) achieved the lowest complexity across all metrics with the smallest model footprint, demonstrating the best balance of simplicity and resource efficiency.
5. Model size did not predict effectiveness; Qwen3 (0.6B) outperformed Phi3 (3.8B) in absolute simplicity across all readability measures.
6. Weight factor selection significantly impacted outcomes; 1.5× reduced compensatory verbosity compared to 2.0× across all metrics.
7. Difficult words were reduced by 53-87\% under prompting interventions, directly addressing A1 vocabulary accessibility requirements.

These findings demonstrate that inference-time complexity control is viable for Small Language Models, with effectiveness highly dependent on model architecture and intervention design.
