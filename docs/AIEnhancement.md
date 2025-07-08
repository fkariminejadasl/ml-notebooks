# AI Enhancement

## Enhancing AI Capabilities: Post-Training, Reasoning, and Agent

This document outlines how post-training techniques, reasoning, and agent development interrelate. Each topic contributes to a common goal: enhancing the capabilities of foundation models to perform complex tasks, reason through problems step-by-step, and interact with real-world environments effectively.

### Post-training Foundation Models

Post-training refers to the process of further refining a pre-trained foundation model. The goal is to improve the model’s performance, safety, and alignment with human expectations by applying reinforcement learning–inspired methods after the initial training phase. This refinement process allows models to better adapt to specific tasks or user requirements without altering the foundational knowledge acquired during pre-training.

The typical methods are:

- SFT (Supervised Fine-Tuning)
- [RLHF](https://arxiv.org/abs/1909.08593) (Reinforcement Learning from Human Feedback)
- RM (Reward Model) rule-based RM or model-based RM
- DPO (Direct Preference Optimization)
- PPO (Proximal Policy Optimization)
- GRPO (Group Relative Policy Optimization), arXiv:2402.03300 from DeepSeekMath
- Sequence Policy Optimization (GSPO), arXiv:2507.18071 from Alibaba Qwen announced Group 
- RLVR (Reinforcement Learning with Verifiable Reward)

#### Examples

- [DeepSeek v3](https://arxiv.org/pdf/2412.19437) used SFT, (rule-based, model-based) RM, GRPO
- [Tülu 3](https://arxiv.org/pdf/2411.15124) used RLVR

### Reasoning

Reasoning refers to a model's ability to generate or simulate step-by-step thought processes. The goal is to break down complex problems into smaller, more manageable steps, making the decision-making process more transparent and interpretable. Enhanced reasoning capabilities, often achieved through effective reinforcement learning during post-training, enable models not only to arrive at answers but also to provide insights into how those answers were derived. Examples of such reasoning techniques include chain-of-thought prompting. Notable models demonstrating these capabilities include OpenAI's o1 and o3, DeepSeek-R1, Qwen QwQ, and Google Gemini 2.0 Flash Thinking.

### Agents

An [agent](https://huggingface.co/blog/smolagents) is an AI-powered program that acts as a bridge between a language model and the external world. Instead of just generating text, it interprets the model's outputs to execute real-world tasks—like retrieving information, interacting with software tools, or controlling hardware. Essentially, agents give language models the ability to "do" things, making them active participants in various workflows.

Agents transform passive language model outputs into actionable commands that can manipulate external systems, thereby expanding the practical applications of AI beyond simple text generation.

#### Examples of Agents

Agents can integrate multiple tools or services simultaneously to handle complex tasks.

- Search Agent: An agent that receives a query from a language model and automatically uses a search API (like Google or Bing) to retrieve relevant information. It then processes and presents the search results, allowing the AI to provide up-to-date and accurate responses.

- Data Analysis Agent: An agent that integrates with data analysis libraries or environments (such as Python’s Pandas and Matplotlib). When the language model generates a request for data analysis or visualization, the agent executes the necessary code, performs the analysis, and returns the output to the user.

- Task Automation Agent: Consider a virtual assistant that schedules meetings. Here, the language model interprets user requests (e.g., "Schedule a meeting with John tomorrow at 3 PM") and the agent interacts with calendar APIs to set up the meeting, send invites, and confirm availability.

- Chatbot with External API Calls: A conversational agent that not only chats with users but also interacts with external services, such as weather or news APIs. If a user asks, “What’s the weather like in New York?”, the agent processes the request, calls a weather API, and then integrates the retrieved data into its response.

### Libraries and Codes

Post-Training Libraries:

- TRL from Hugging Face: A cutting-edge library designed for post-training foundation models using advanced techniques like Supervised Fine-Tuning (SFT), Proximal Policy Optimization (PPO), and Direct Preference Optimization (DPO).

Reasoning:

- [Training a small math reasoner with RL: Qwen 0.5b on GRPO](https://colab.research.google.com/drive/1bfhs1FMLW3FGa8ydvkOZyBNxLYOu0Hev?usp=sharing) [from @LiorOnAI](https://x.com/LiorOnAI/status/1886850813911556351)
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero) is a reproduction of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) built upon [veRL](https://github.com/volcengine/verl).
- [Fully open reproduction of DeepSeek-R1 by Hugging Face](https://github.com/huggingface/open-r1)

Agents:

- [Open-source DeepResearch by Hugging Face – Freeing our search agents](https://huggingface.co/blog/open-deep-research)

### Courses

- [deeplearning.ai short courses: post training of llms](https://www.deeplearning.ai/short-courses/post-training-of-llms)
- [smol-course](https://github.com/huggingface/smol-course) on Agents and post-trainig.

### (Fully) Open Models

Fully open models release not only the model and weights but also data strategies and implementation details. Examples of fully open models include [NVIDIA Eagle 2](https://arxiv.org/abs/2501.14818v1), [Cambrian-1](https://arxiv.org/abs/2406.16860) and the LLaVA family ([LLaVA-OneVision](https://arxiv.org/abs/2408.03326), [LLAVA-v1](https://arxiv.org/abs/2304.08485)), [DeepSeek-R1](https://arxiv.org/abs/2501.12948).

## Model and Architecture Enhancement

Titan (Memorize at test time), Jamba (Mamba, MoE, Attention), Mamba (SSM: state space model), Transformer (Attention, MLP), MoE (Mixture of experts) instead of MLP. DeepSeek AI's MLA (Multi-Head Latent Attention) instead of MQA (Multi-Quary Attention), GQA (Group-Quary Attention), MHA (Multi-Head Attention).

Improvements: Rotary embeddings, RMSNorm, QK-Norm, and ReLU², softcap logits, Muon optimizer

Example improvement is training [nanoGPT](https://github.com/KellerJordan/modded-nanogpt) from 45 minutes to 3 minutes using 8 x H100 GPUs. From the github:

```
This improvement in training performance was brought about by the following techniques:

- Modernized architecture: Rotary embeddings, QK-Norm, and ReLU²
- Muon optimizer [writeup] [repo]
- Untie head from embedding, use FP8 matmul for head, and softcap logits (latter following Gemma 2)
- Projection and classification layers initialized to zero (muP-like)
- Skip connections from embedding to every block as well as between blocks in U-net pattern
- Extra embeddings which are mixed into the values in attention layers (inspired by Zhou et al. 2024)
- FlexAttention with long-short sliding window attention pattern (inspired by Gemma 2) and window size warmup
```