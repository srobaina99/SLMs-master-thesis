![[Pasted image 20250610224422.png]]
# Task agnostic
## Tiny Lama
Optimized handling of data
Massive data 1T tokens
QA Common sense performs like bigger model

## Mistral 7B
Optimized training 
Efective size
Behaves like 38B in code and math

## Mixture of experts MOE
8 Experts and router to input to the best expert
- [ ] Very efficient for inference time (once its routed just one responds) 

## Phi 1
quality of the data important rather than quantity
beats models that are 400x in code tasks

phi 2 close to lama 70B
phi 3 mini 3.8 competes MOE and gpt 3.5
phi 4 competes gpt 4 mini in reasoning

## Orca 13B
Explanation tuning
learning the thought process

## Gemini nano Multimodal (destiled)
-> Good for audio
nano 1 (1.8 b)
nano 2 (3.25b)

## Qwen
1.8 b - 14b
3.5 - 7.2b
optimized processes 
long context 

## zamba jamba mamba
hybrid transformer and something else


# Task specific
Wizard math (7b 13b) beats lama 70B

wizard code (2b - 7b)

Slade (200M) Code decompilation  Better than chatGPT

flame 60M for excel formulas beats GPT4

Prometheus 7b evaluation agreement as AI judge

# How?
- Evolution how they learn from larger models
## Training techniques
- Imitation -> Progressive learning (orca) Explanation tuning
- Kwoldege distilation
	- Distining the intermediate 
	- CoT KD
	- Reasining distillation
	- Decompostion distilation
		- A big problem is broken down and smaller models solve the pieces
			- Boosts benchmarks
## Modularized training techniques
- Blended Ensables
	- Ensamble of small 3 models (6-7b) was better 175b in user engagement in A/B testing

## Data strategies
- QUALITY OVER QUANTITY
	- High quality synthetic data

TinyGSM [[74](https://arxiv.org/html/2501.05465v1#bib.bib74)], a synthetic dataset of 12.3M grade school math problems paired with Python solutions, generated fully by GPT-3.5. After finetuning on TinyGSM, 1.3B generation model and a 1.3B verifier model can achieve 81.5 accuracy, outperforming existing models that are orders of magnitude larger. This also rivals the performance of the GPT-3.5 “teacher” model (77.4), from which model’s training data is generated due to the high-quality dataset TinyGSM and the use of a verifier model, which selects the final outputs from multiple candidate generations.