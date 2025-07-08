# Companies

The methodology employed categorizes companies based on their primary focus or offerings, detailing each company's capabilities alongside its name.

**Key to Keywords:**  
- **Pre:** Pretraining capabilities or services  
- **FT:** Finetuning services  
- **Custom:** Support for building custom models  
- **API:** Offers an API to integrate ML functionality  
- **Dev:** Development tools or environments  
- **Dep:** Deployment tools or platforms  
- **Model:** Develops or provides their own ML models  
- **GPU:** GPU-based compute resource provider  
- **Storage:** Data storage provider  
- **Data:** Data platform or data indexing provider

## Highlights

Models are accessed via their respective companies, such as OpenAI (ChatGPT 4o, o1, o3), DeepSeek (DeepSeek-R1), Anthropic (Claude Sonnet), and Google (Gemini). `LM Studio` is a desktop application for running models locally. `Together AI` provides a platform for running models that might not fit on your local machine. There are other serving locations, such as the [Hugging Face Inference Playground](https://huggingface.co/spaces/huggingface/inference-playground), and `Hyperbolic.xyz`, which also offers base models. 

Manus AI (Monica.im) autonomously executes complex, goal-driven workflows as an [AI agent](https://huggingface.co/blog/LLMhacker/manus-ai-best-ai-agent), Devin AI serves as an AI-powered software engineer for coding and build automation, and Cursor functions as an intelligent IDE assistant that augments human developers across the software development lifecycle.

For better prompting, it’s recommended to add a `llms.txt` file at your site’s root (analogous to `robots.txt`) to provide large language models with structured, machine-readable guidance. Additionally, [Gitingest](https://gitingest.com) converts any Git repository into a single, prompt-friendly text digest—complete with directory structure and file content—making it ideal for feeding codebases into LLMs.


The Model Context Protocol (MCP), introduced by Anthropic, is an open standard for connecting LLMs to external data sources and tools—much like a USB-C port for AI applications (e.g., see [How to Build an MCP Server with Gradio](https://huggingface.co/blog/gradio-mcp)).

DeepWiki, from Cognition Labs, offers AI-powered, interactive documentation for any GitHub repository. Just replace `github.com` with `deepwiki.com` in your repo URL to generate searchable, context-rich docs instantly.

---

### Model Creators / Foundation Model Providers

- [OpenAI](https://openai.com/): Model, Pre, FT, API  
- [Anthropic](https://www.anthropic.com/): Model, Pre, FT  
- [Google DeepMind, Google](https://www.deepmind.com/): Model, Pre, FT, API  
- [xAI](https://x.ai/): Model, Pre, FT, API  
- [Meta AI](https://ai.meta.com/): Model, Pre, FT, API  
- [Mistral AI](https://mistral.ai/): Model, Pre, FT, API  
- [Nous Research](https://www.nousresearch.com/): Model, Pre, FT  
- [Cohere](https://cohere.com/): Model, Pre, FT, API  
- [Allen Institute for AI (Ai2)](https://allenai.org/): Model, Pre, FT  
- [EleutherAI](https://www.eleuther.ai/): Model (Open-Source Models)  
- [AssemblyAI](https://www.assemblyai.com/): Model, API  
- [Deepgram](https://deepgram.com/): Model, API  
- [ElevenLabs](https://elevenlabs.io/): Model, API  
- [Perplexity AI](https://www.perplexity.ai/): Model, API
- [Stability AI](https://stability.ai/): Model, Pre, FT  
- Ideogram: Model
- Midjourney: Model
- Black Forest Labs: Model
- Pika Labs: Model
- Luma Labs: Model
- Minimax: Model
- Tencent AI Lab: Model
- ByteDance (owns TikTok): Model
- Inflection AI: Model
- Character.AI: Model


### API/Inference Providers

- [Replicate](https://replicate.com/): API, GPU, Custom
- [Together AI](https://together.xyz/): API, FT, Custom, GPU. Also finetune by WebUI.
- [OpenRouter](https://openrouter.ai/): API
- [Groq.AI](https://groq.com/): API  
- [Novita.ai](https://novita.ai/): API, GPU 
- [Lepton.ai](https://lepton.ai/): API, GPU
- [Hyperbolic.xyz](https://hyperbolic.xyz/): API, GPU  
- [Fireworks AI](https://fireworks.ai/): API, GPU  
- [Baseten](https://www.baseten.co/): API  
- [Deepinfra](https://deepinfra.com/): API, GPU
- [Octo AI](https://octoai.com/): API

### Development Tools, Code Generation, and Productivity

- [Codeium](https://www.codeium.com/): Dev (It developed both the Codeium AI code acceleration platform and the Windsurf IDE.)
- [Replit AI](https://replit.com/): Dev
- Bolt from StackBlitz: Dev (Bolt platform)
- Anthropic Computer Use
- [Amazon CodeWhisperer](https://aws.amazon.com/codewhisperer/): Dev
- [GitHub Copilot](https://github.com/features/copilot): Dev  
- [Cursor](https://www.cursor.com/): Dev
- [Cognition Labs](https://www.cognitionlabs.ai/): Dev  

Platforms like Bolt, Replit Agent, Vercel V0 use agentic workflows to improve code quality. They also help deploy generated applications. 

Other tools related:

- Llama Stack developed by Meta is a collection of APIs that standardize the building blocks necessary for developing generative AI applications.
- Adept's ACT-1: Adept AI has developed ACT-1, an AI model designed to interact with software applications through natural language commands.  It can perform tasks such as navigating web pages, clicking buttons, and entering data, effectively acting as a digital assistant to automate workflows.

### Deployment Tools and Platforms

- [Vercel](https://vercel.com/): Dev, Dep (V0 by Vercel generates UI from text)
- Open WebUI: Dep (Manage and deploy AI models locally. UI for interacting with various LLMs. Integrate with LLM runners like Ollama and OpenAI-compatible APIs)

### Libraries and Frameworks

- Ollama, vLLM: tool designed to run LLMs locally. It can be used with [Chatbox.ai](https://chatboxai.app) app desktop, mobile and web-based app.
- unsloth (Daniel Han), torchtune, Oxolotl: Fast Training (Pre/FT related).  They enhance the speed and efficiency of LLM fine-tuning.
- LiteLLM: LiteLLM is an open-source Python library designed to streamline interactions with a wide range of LLMs by providing a unified interface.
- [ai-gradio](https://github.com/AK391/ai-gradio): A Python package that makes it easy for developers to create machine learning apps powered by various AI providers. Built on top of Gradio, it provides a unified interface for multiple AI models and services.
- [DSPy](https://dspy.ai/): Dev (Provides a programming model for developing and optimizing language model pipelines)  
- [Torchrun](https://pytorch.org/docs/stable/elastic/torchrun.html): Dev  
- [Cog (by Replicate)](https://github.com/replicate/cog): Packaging Custom ML Models for Deployment (Custom, Dep)  
- [ComfyUI](https://comfyui.org/): Dev (GUI for Workflow)  
- [Tinygrad](https://github.com/geohot/tinygrad): Dev  
- Agentic AI systems: LangChain, CrewAI, LlamaIndex, Haystack, Devin (Cognition)
- [OpenHands](https://openhands.ai): Dev (providing frameworks for building AI-driven applications with agentic capabilities)  

### Data Platforms and Providers

- [Encord](https://encord.com/): Data  
- [LlamaIndex](https://www.llamaindex.ai/): Dev, Data Indexing  
- [LAION](https://laion.ai/): Data Provider, Open-Source Datasets

### End-to-End Cloud-Based ML Platforms

- Google Vertex AI, AI Studio, GenAI SDK: API, Pre, FT, Custom, Dev, Dep, Model
- [Amazon Bedrock](https://aws.amazon.com/bedrock/): API, Pre, FT, Custom, Dev, Dep, Model  
- [Microsoft Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning/): API, Pre, FT, Custom, Dev, Dep, Model  
- [Databricks](https://databricks.com/): API, Dev, Dep, Model  
- [IBM Watson Studio](https://www.ibm.com/cloud/watson-studio): Dev, Dep, Model  
- [DataRobot](https://www.datarobot.com/): Dev, Dep, Model  
- [H2O.ai](https://www.h2o.ai/): Dev, Dep, Model  
- [Domino Data Lab](https://www.dominodatalab.com/): Dev, Dep, Model  
- [Algorithmia](https://algorithmia.com/): Dev, Dep, Model
