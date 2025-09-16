## Monolithic specialist
Single SLM finetuned for a task.
Phi3 mini- 3.8B - in vehicle agent. Car voice assistant, on device

Tiny agent framework 1.1B
For Mac books

SFT with task focus specific data 

## Factored Agent
Planing \ Execution
LLM as planner and SLM executes and formats. <- robust

## Agent distill
KT without training
Distil from LLM to SLM
-> MCP created by the LLM
External library that the SLM can use. Skills plug and play

## React and SLM
show its reasoning
thought -> action -> pause -> observation
1. articulate its thought
2. tool
3. output pause token, external script that runs the tool and feeds output into the context of the SLM


# Actualy deployed
## Phi3 mini
11 token on CPU

Mercedes benz
Airline

### Gemma familiy 
Gemma 3 -> support Tool calling with react
To run locally:
Allama 
Llama CPP

## TinyLama 1.1B
base to calama
fine tune for robust function calling

Runs on CPU

Bad at math
Good language skills

## Nvidia
## Hybrid architecture
nimatron
hymba

speed! 

hymba 1.5B 3.5x thruoughput
	how much work a model can do a model can do. It can handle a lot of tokens (efficiency at scale)

Advanced RL to ensure correct toolcalling

# Real case studies

Conversa preda base | 
Call center conversation

3B parameter

## Air train
7B for patients chatbot
on pair with GPT3.5

## Tinder
detecting policy violation
Good recall and precisition

# Engenering 
SFT and DPO

## PF (parameter eficient fine tuning)
LORA
CALORA (Lora + quantization)

RL for sophisticated behavior
- GRPO (Nvidia) for tool use and instruction following

## Hardware
- Tinylama 1.1B 16 GB vram 

- Phi3 mini LORA Apple MLX framework M2 (8.3 gb vram)

## Inference
phi3 mini on edge CPU (mercedes example)
LamaCPP

Gemma 

## Optimization
- Quantization
- Structure pruning (cutting redundant or less critical)
	- Mercedes pruned phi3 mini from 3.8B to 1.8B cutting decoding layers
	- After pruning they do some light finetuning (healing)
- For Func calling
	- Tiny Agent -> Prompt optimization with tool retrieval (ToolAG) Only include whats needed

# Performance analysis
- function calling tool use benchmarks
	- berkeley function calling leader board: Salesforce model (XLM2 8B) 72% accuracy
	- 8B are good at accuracy with tools
- For good tool use (format) FT with good data is the way

### Task specific is where SLM shine
MAC SLM 1.1B assistant nails 80.3% success rate on function calling
GPT 4 turbo 79%





