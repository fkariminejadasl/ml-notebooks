This text provides a detailed overview of various tasks that different large models, such as Large Language Models (LLMs), Large Multimodal Models (LMMs), and Large Vision Models (LVMs), are capable of handling, including text generation, translation, image analysis, multimodal reasoning, and more, illustrating the diverse applications of these advanced AI systems.

### LLM (Large Language Model): 
- Examples: 
    - Text generation and writing: content creation, summarization, paraphrasing, scriptwriting, poetry, social media posts, email drafting, product descriptions, resume/cover letter writing, letter writing.
    - Text comprehension and analysis: question answering, text analysis, critical review, legal document analysis, code understanding.
    - Translation and language tasks: language translation, language learning assistance, text correction, dialect and style adaptation.
    - Research and information retrieval: fact-checking, information gathering, historical context, current events analysis.
    - Creative and artistic assistance: creative brainstorming, character creation, plot development, meme creation.
    - Programming and technical tasks: code generation, debugging assistance, algorithm explanation, data analysis, simulation, and modeling.
    - Decision-making support: pros and cons analysis, risk assessment, scenario planning, strategic planning.
    - Education and tutoring: subject tutoring, test preparation, essay assistance, learning resources.
    - Communication and interaction: chatbot support, role-playing, interview simulation.
    - Entertainment: game dialogue creation, trivia and quizzes, interactive storytelling.
    - Personal assistance: task management, goal setting, personal advice.
    - Customization and personalization: persona development, tone adjustment, content filtering.
    - Simulation and modeling: scenario simulation, market analysis, behavioral modeling.
    - Explanation: explaining complex concepts.
    - Ethics and philosophy: ethical analysis, debate simulation.

- Examples from Llama 3: general knowledge and instruction following, code, math, reasoning, tool use (search engine, Python interpreter, mathematical computational engine), multilingual capabilities, long context (code reasoning, summarization, question answering).

### LMM (Large Multimodal Model) / VLM (Vision Language Model): 
- Examples from Unified-IO 2 (images, text, audio, action, points, bounding boxes, keypoints, and camera poses): image editing, image generation, free-form VQA, depth and normal generation, visual-based audio generation, robotic manipulation, reference image generation, multiview image completion, visual parsing and segmentation, keypoint estimation, visual audio localization, future frame prediction.

- Examples from 4M-21: multimodal retrieval (e.g., given image retrieve caption, segment), multimodal generation (e.g., given image generate depth), out-of-the-box (zero-shot) tasks (e.g., normal, depth, semantic segmentation, instance segmentation, 3D human pose estimation, clustering). 

- Examples from ImageBind (images, text, audio, depth, thermal, and IMU): cross-modal retrieval, embedding-space arithmetic, audio to image generation.

- Examples from Llama 3 on image: visual grounding (grounding information such as points, bounding boxes, and masks), MMMU or multimodal reasoning (understand images and solve college-level problems in multiple-choice and open-ended questions), VQA / ChartQA / TextVQA / DocVQA (image, chart, diagram, text in the image).
- Examples from Llama 3 on video: QA: PerceptionTest (answer temporal reasoning questions focusing on skills like memory, abstraction, physics, semantics, and different types of reasoning such as descriptive, explanatory, predictive, counterfactual), temporal and causal reasoning, with a focus on open-ended question answering, compositional reasoning requiring spatiotemporal localization of relevant moments, recognition of visual concepts, and joint reasoning with subtitle-based dialogue, reasoning over long video clips to understand actions, spatial relations, temporal relations, and counting.
- Examples from Llama 3 on speech: speech recognition, speech translation, spoken question answering.

- Examples: something (audio, text) to image generation, open-set object detection, image captioning, image description, visual question answering, visual commonsense reasoning, text-based image retrieval, image-based text generation, text-guided image manipulation, data visualization.

### LVM (Large Vision Model)
- Examples from SAM2: promptable visual segmentation (prompts: clicks, boxes, or masks), promptable segmentation.
- Examples from SAM: (prompts: point, box, segment, text): zero-shot single point valid mask evaluation (segmenting an object from a single foreground point), zero-shot edge detection, zero-shot object proposals, zero-shot instance segmentation, zero-shot text-to-mask. 
- Fine-tuning on specific tasks. 
    - Image task examples: classification, object detection, semantic segmentation, instance segmentation, image generation, style transfer, super-resolution, image inpainting, face recognition and analysis, optical character recognition, scene understanding, anomaly detection, gesture and pose recognition, image retrieval, emotion recognition, visual enhancement.
    - Multiple images/video task examples: action recognition, video object detection, multiple object tracking, visual object tracking, track anypoint, optical flow, 3D object reconstruction, SLAM/SfM.

### References
- LLM, LMM, VLM: [Llama 3](https://arxiv.org/abs/2407.21783), [4M-21](https://arxiv.org/abs/2406.09406), [Unified-IO 2](https://arxiv.org/abs/2312.17172), Florence
- LVM: [DINOv2](https://arxiv.org/abs/2304.07193), [SAM](https://arxiv.org/abs/2304.02643), [SAM 2](https://arxiv.org/abs/2408.00714). CLIP, SigLIP, [ImageBind](https://arxiv.org/abs/2305.05665)

### SOTA and Popular Off-the-Shelf Models:

The following list highlights some of the current state-of-the-art (SOTA) and previously leading methods used in various domains such as tracking, depth estimation, optical flow, 3D reconstruction, segmentation, and language models. These methods are selected based on papers and research studies I have read, where they were commonly employed as off-the-shelf solutions.

Please note that the field of machine learning and computer vision is rapidly evolving, and new models are frequently introduced. This list reflects a snapshot of current practices, but advancements in the field may lead to newer and potentially better-performing techniques over time.

- **Vision Encoders:** Vision-Only Model Encoders: PE (Perception Encoder, Mets), DINO, MAE (Masked Autoencoders), ResNet. Vision-Text Model Encoders: AIMv2, SigLIP, CLIP, BLIP. Vision-Text Encoder for Generation: LlamGen. E.g. In Janus (DeepSeek), LlamaGen is used for Geneneation Encoder and SigLIP for Understanding Encoder.
- **Depth Map:** DepthAnything, Depth Pro, DepthCrafter (for video), MiDaS, Depth Anything v2, DPT, ZoeDepth
- **Optical Flow:** SEA RAFT, RAFT
- **3D/4D Reconstruction:** Depth Anything 3, MapAnything, ViPE, VGGT, COLMAP (Non-Deep Learning), DuSt3R, MASt3R, 4D (ViPE, St4RTrack, Easi3D, CUT3R, DAS3R, MonST3R, Dynamic Point Maps), 4D Online (ODHSR, only human), VideoMimic, ACE Zero (ACE0), noposplat (potentially for sparse reconstruction), 
- **Point Matching and Point Tracking:** TAP (CoTracker3, TAPIR, PIP), SAM2 (Segment and Tracking Anything: SAM combined with DeAOT);SuperPoint combined with lightGLUE or SuperGLUE, MASt3R, 
- **Multi-Object Tracking (MOT):**: SAM2 (Segment Anything Model 2) for image and video, MOTR, ByteTrack, BoT-Sort, FairMOT
- **Referred Multi-Object Tracking / Text-guided Spatial Video Grounding (SVG):** TempRMOT
- **Text-guided Video Temporal Grounding (VTG):** VTG aims to get start and end frames by prompting the action. FlashVTG
- **Object Detection:** Florence-2, Grounding DINO / DINO-X, PaliGemma 2, moondream2, small: Yolo Family (latest [Generalist YOLO paper](https://openaccess.thecvf.com/content/WACV2025/papers/Chang_Generalist_YOLO_Towards_Real-Time_End-to-End_Multi-Task_Visual_Language_Models_WACV_2025_paper.pdf), YOLO12)
- **Segmentation:** Grounding DINO combined with SAM, Florence-2
- **Pose Estimation:** OpenPose
- **Unified Multimodal Image Understanding and Generation:** BLIP3-o (Saleforce), Jaus Pro (DeepSeek), EMU3 (Beijing Baai), MetaQuery, Metamorph, Chameleon(Meta)
- **Image Captioning:** xGen-MM (BLIP-3), RAM (Recognize Anything), CogVLM2, PaliGemma 2
- **Visual Question Answering:** Any of the VLMs or LMM such as Phi-3.5, PaliGemma 2, moondream2. older ones for multi-image LMM: Mantis, OpenFlamingo, Emu, Idefics 
- **Generative Video Models, Text-to-Video Generation Models:** CogVideoX (THUDM Tsinghua University), Pika Labs (Pika Labs), Stable Video Diffusion (Stability AI), Movie Gen and Emu Video (Meta), [Sora](https://icml.cc/virtual/2024/39514) (OpenAI), Gen-3 Alpha (Runway AI), Veo (Google DeepMind), Flux (Black Forest Labs), Hunyuan Video, DynamiCrafter, VideoCrafter (Tencent AI Lab), PixelDance and Seaweed (ByteDance owns TikTok can access via Jimeng AI platform), MiniMax T2V-01, Video-01 (Minimax can access via Hailuo AI platform), Luma Dream Machine (Luma Labs), Kling (Kuaishou), Alibaba (Wan), Open-Sora (HPC AI Tech)
- **Text-to-Image Generation Models:** [SANA](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px) (MIT, NVIDIA, Tsinghua), FLUX1 (Black Forest Labs), Ideogram v2 (Ideogram), Midjourney v6 (Midjourney), Stable Diffusion 3.5 (Stablity AI), DALLE 3 (OpenAI), Firefly 3 (Adobe), Imagen 3, Flamingo (Google DeepMind), Aurora of Grok (xAI), Pixtral (Mistral), PixArt-alpha (Huawei), Janus (DeepSeek), CogView4 (THUDM Tsinghua University)
- **Speech Models: Speech-to-Text, Text-to-Speech, Speech-to-Speech**: Moshi (Kyutai), ElevenLabs, Google, OpenAI, Speech-to-Text: Whisper (OpenAI), Wav2Vec (Meta), SuperWhisper/Wisper Flow
- **Control Video by Action**: Genie 2 (Google DeepMind)
- **Vision Language Models (VLMs)**: image, multi-image, video. Open VLMs: PLM (Perception LM, Meta), Cosmos Nemotron (NVidia), DeepSeek-VL2 (DeepSeek), QWen2-VL (Alibaba), InternVL2 (OpenGVLab), LLAVA 1.5, LLama3.2 (mainly LLM), Cambrian-1, CogVLM2 (Tsinghua University), MolMo (Ai2), SmolVLM (Hugging Face). Proprietary: GPT-40 (OpenAI), Claude Sonnet 3.5 (Claude), Gemini 1.5 Pro (Google)
- **Large Language Models (LLMs):**  Open source: DeepSeek v3 (DeepSeek),Qwen (Alibaba), LLAMA-3 (Meta), Phi-3 (Microsoft), Gemma 3 (Google),  OLMo 2 (Ai2), Helium-1 (Kyutai), Sky-T1-32B (UC Berkeley), Cerebras-GPT (Cerebras). Proprietary: Claude3 (Anthropic), Gemini (Google DeepMind), Nova (Amazon), Flash 3, Nexus (Reka AI)
  Note that: Some of the models are Multimodal.
- **Point Cloud Encoders**: Sonata, Point Transformer V3 (PTv3), MinkowskiNet
- **Point Cloud with Text**: OpenScene (Open-vocabulary 3D Scene Understanding/ OV search in 3)
- **Multimodal**: OmniVinci (NVIDIA)
- **Other Foundation Models**: Motor Control: HOVER. Weather Forcast: FourCastNet (NVidia), GenCast (DeepMind). Multilingual: SeamlessM4T (Meta), Brain2Qwerty (Meta). Remote Sensing: LISAT (Darrell), EarthGPT, RS-GPT4V

### Companies

OpenAI (ChatGPT), Anthropic (Claude), X (Grok), Meta (LLaMa) 

DeepSeek, Moonshot AI (Kimi), Alibaba (QWen, Wan), Zhipu AI (GLM), MiniMax (Hailuo), ByteDance / TikTok, Tencent Hunyuan

### Finding Models

#### Models in General

- [Hugging Face](https://huggingface.co/models)
- [Roboflow](https://roboflow.com/model-feature/zero-shot-detection) contains the list of top models.

#### LMM (Large Multimodal Model) / VLM (Vision Language Model): 

- Find open-source models: The [Open VLM leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) shows the scores of top VLMs, and you can see which models are open-source or proprietary.
- [Hugging Face Video Generation Leaderboards](https://huggingface.co/spaces/ArtificialAnalysis/Video-Generation-Arena-Leaderboard), [artificialanalysislu](https://artificialanalysis.ai/text-to-video/arena?tab=Leaderboard)
- [Text-to-Video](https://artificialanalysis.ai/text-to-video/arena?tab=Leaderboard)
- Blog post on VLM: [Implement a Vision Language Model from Scratch](https://huggingface.co/blog/AviSoori1x/seemore-vision-language-model); [Vision Language Using and Finetuning](https://huggingface.co/blog/vlms); [vision language explanation](https://huggingface.co/blog/vision_language_pretraining);

#### LLM (Large Language Models)

- Find open-source models: [LLM Arena](https://lmarena.ai) LMArena formerly LMSYS shows the current top LLMs. You can see which models are open-source or proprietary. [scale.com](https://scale.com/leaderboard).

### Speech Models

- [Speech Arena](https://artificialanalysis.ai/text-to-video/arena?tab=Leaderboard)

### API Providers 

[Artificial Analysis](https://artificialanalysis.ai)


