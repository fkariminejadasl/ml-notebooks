# Time Series

### Time-Series Representation Learning

- [TF-C](https://zitniklab.hms.harvard.edu/projects/TF-C): Time-Frequency Consistency (TF-C) Model (Harvard MIMS Lab, NeurIPS 2022) - A cross-domain time-series representation model that leverages contrastive learning between time-domain and frequency-domain views of the same signal​. By enforcing consistent embeddings in both domains, TF-C learns general features transferable to diverse sensors (EEG, accelerometer, etc.). Weights are available.
- [TS-TCC](https://arxiv.org/abs/2106.14112): Time-Series Representation Learning via Temporal and Contextual Contrasting: contrastive. augmentation: jitter, scale the amplitude and permutation.
- [TimesURL](https://arxiv.org/abs/2312.15709): Learning Universal Representations of Time Series via Contrastive Learning: contrastive and reconstruction based (MAE) based on **TS2Ve** model. augmentation: left-righ cropping with frequency swaping of the negative example. hard negatives are temporal and instance based. In temporal way, the augmentation mix in different time. In the instance-based augmentation, this mixing occurs in mixing the different instances.
- [TS2Vec](https://arxiv.org/abs/2106.10466): Towards universal representation of time series. Augmentation: croping
- [CPC](https://arxiv.org/abs/1807.03748): Representation Learning with Contrastive Predictive Coding: The contrastive learning works such as SimCLR, MoCo, SwAV, DINO are based on the contrastive loss introduced in this paper. The postive are the predictions.
- [TNC](https://arxiv.org/abs/2106.00750) (Temporal Neighborhood Coding): Triplet loss (neighbor positive and further negative)

### Generalized Category Discovery (GCD)

- [GCD](https://arxiv.org/pdf/2201.02609): Generalized Category Discovery. Use supervised contrastive and self-supervised contrastive for representation learning. Then use KMean with forcing the labeled data stays in a cluster. Hungarian matching assignment defines the accuracy of clustering on the labeled data, where this accuracy defines the number of clusters. 

### Imbalanced Generalized Category Discovery (GCD)

- [SimGCD](https://arxiv.org/pdf/2211.11727v4): Clustering is trained jointly with the representation learning network. This is not an imbalanced data setting. However, entropy is used as a regularization to prevent the model from over-predicting certain label classes. This issue is more about imbalanced prediction, which arises during joint learning of clustering and representation, but not in separate training as done in the original GCD method. [MASA: Multi-Activity Sequence Alignment via Implicit Clustering](https://arxiv.org/pdf/2503.12519) is related in terms of parametric clustering, though it addresses a different task.
- [LegoGCD](https://openaccess.thecvf.com/content/CVPR2024/papers/Cao_Solving_the_Catastrophic_Forgetting_Problem_in_Generalized_Category_Discovery_CVPR_2024_paper.pdf)
- [AGCD](https://arxiv.org/pdf/2403.04272)
- [BaCon](https://proceedings.neurips.cc/paper_files/paper/2023/file/b7216f4a324864e1f592c18de4d83d10-Paper-Conference.pdf)
- [Generalized Category Discovery under the Long-Tailed Distribution](https://openreview.net/pdf?id=0CIS2nthtK)
- [DebiasGCD](https://openreview.net/pdf?id=JRcfgNg2ZJ)
- [Long-tailed GCD](https://arxiv.org/pdf/2401.05352v2)
- [ImbaGCD](https://arxiv.org/pdf/2401.05353)

### General Time-Series Models

[MOIRAI-MOE](https://arxiv.org/abs/2410.10469): Time-series foundation model, which uses mixture of expertes 
to select for different data frequencies. It is build upon [MOIRAI](https://arxiv.org/abs/2402.02592). 
Other time-series foundation models are [Moment](https://github.com/moment-timeseries-foundation-model/moment), 
[MOIRAI](https://arxiv.org/abs/2402.02592), [Chronos](https://arxiv.org/abs/2403.07815), [PatchTST](https://arxiv.org/abs/2211.14730)
[TimesFM](https://arxiv.org/abs/2310.10688), [Lag-Llama](https://github.com/time-series-foundation-models/lag-llama), [TimeGPT-1].

[Autoformer](https://arxiv.org/abs/2106.13008), [Informer](https://arxiv.org/pdf/2012.07436), 
Reformer for the long-term forecasting. Some of these methods are 
provided in [HuggingFace Time Series Models](https://huggingface.co/docs/transformers/en/model_doc/autoformer). 
In [Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504), argues the transformers are not needed. 

older works: [TIME-LLM](https://arxiv.org/abs/2310.01728), [LLMTime](https://arxiv.org/abs/2310.07820), [AutoTimes](https://arxiv.org/abs/2402.02370)
GPT-4TS


#### Multimodal Time Series


**Time-series → text (captioning).**
[TSML](https://arxiv.org/abs/2501.01832) introduces a multimodal encoder–decoder that merges a 1-D CNN–based time-series encoder with a positional text‐token stream, and learns this stack end-to-end on an in-context–generated, cross-modally denoised synthetic caption corpus—setting a new state-of-the-art in descriptive accuracy across multiple benchmarks. [TADACap](https://arxiv.org/abs/2504.11441), in contrast, requires no gradient updates: it employs a novel diverse‐retrieval strategy to pull the most relevant series–caption pairs from a domain‐specific memory bank and reuses those captions directly—achieving comparable semantic quality with dramatically lower annotation effort and zero fine-tuning. Together, these approaches illustrate the full spectrum—from fully trained specialist decoders to pure retrieval–plus–reuse pipelines—for interpretable time-series narration.


**Chat-style time-series assistants.**
[ChatTS](https://arxiv.org/abs/2412.03104) treats multivariate time series as a first-class modality by generating attribute-rich synthetic data (via an attribute-based time-series generator and the Time Series Evol-Instruct algorithm) and fine-tuning both a lightweight context-aware time-series encoder and a 14 B LLM on six alignment and four reasoning tasks—yielding a chat interface that can answer questions, detect anomalies, and explain forecasts directly from raw numbers. [ChatTime](https://arxiv.org/abs/2412.11376), by contrast, reframes each normalized and discretized numeric value as a new “foreign-language” token, expands a 1 B-parameter LLM’s vocabulary accordingly, and then applies continuous pre-training plus instruction fine-tuning—updating only the added token embeddings and heads (≈ 350 M trainable parameters)—to deliver zero-shot forecasting and seamless bidirectional dialogue between time series and text without touching the core model weights. Together, they span the design space from full LLM fine-tuning for maximal conversational fidelity to parameter-efficient tuning for lightweight, on-device time-series assistants.


**Forecasting & reasoning with LLMs in the loop.**
[TimeXL](https://arxiv.org/abs/2503.01013) integrates a prototype‐based multimodal encoder with a closed-loop trio of LLM stages—prediction, reflection, and refinement—to produce up to an 8.9 % AUC improvement alongside human-centric, case-based rationales, without requiring full LLM fine-tuning. [CAPTime](https://arxiv.org/abs/2505.10774) freezes both its pretrained time-series encoder and base LLM, then aligns temporal patterns with exogenous text via learnable interactions and a mixture-of-distribution experts, yielding calibrated, multimodal probabilistic forecasts. [SMETimes](https://arxiv.org/abs/2503.03594) systematically evaluates sub-3 B-parameter “Small Language Models” using statistical prompting, an adaptive fusion embedding architecture, and a dynamic mixture-of-experts framework to rival 7 B baselines—achieving 3.8× faster training, 5.2× lower memory, and state-of-the-art accuracy. [TimeCMA](https://arxiv.org/abs/2406.01638) employs dual-branch encoding—weak, disentangled time-series embeddings alongside robust LLM-derived prompt embeddings—and aligns them via cross-modality similarity, passing only the last token to downstream predictors to cut computation, outperforming prior methods on eight datasets.


Only ChatTS, ChatTime, SMETimes weights are released.


### Discriminative Representation

The representation that can be used in `GCN (Generalized Category Discovery)` ([GCN](https://arxiv.org/abs/2201.02609), [SelEx](https://arxiv.org/abs/2408.14371)). 

Contrastive learning, Sparse autoencoder or older method such as [DEC (Deep Embedded Clustering)](https://arxiv.org/abs/1511.06335), SOM (Self Organizing Maps).

#### Characteristics of Time Series

[Implicit Reasoning in Deep Time Series Forecasting](https://arxiv.org/pdf/2409.10840): It is observed that certain linear, MLP-based, and patch-based Transformer models generalize effectively in carefully structured out-of-distribution scenarios, suggesting underexplored reasoning capabilities beyond simple pattern memorization.