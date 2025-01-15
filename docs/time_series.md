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
[TSML](https://arxiv.org/abs/2501.01832) crafts a bespoke encoder–decoder that fuses a 1-D CNN time-series encoder with text tokens, then trains this multimodal stack end-to-end on a synthetic-and-denoised caption corpus; the heavy language lifting during inference is off-loaded to a frozen general-purpose LLM that simply rewrites and ranks the raw captions. [TADACap](https://arxiv.org/abs/2504.11441), by contrast, skips fine-tuning altogether: it retrieves the most similar series–caption pairs from a domain-specific memory and feeds them, unchanged, to an instruction-tuned LLM (e.g. Vicuna or GPT-4) to “fill-in-the-blank”, giving domain-aware captions without a single gradient step. Together they show the spectrum from fully-trained specialist decoders to zero-training retrieval-plus-prompting for time-series narration. ([arxiv.org][1], [arxiv.org][2])

[1]: https://arxiv.org/abs/2501.01832 "TSML"
[2]: https://arxiv.org/abs/2504.11441 "TADACap"

**Chat-style time-series assistants.**
[ChatTS](https://arxiv.org/abs/2412.03104) aligns raw multivariate sequences to a 14 B-parameter Qwen model via a lightweight patch encoder and two-stage full-parameter instruction tuning on synthetic QA data, yielding a chatbot that can answer questions, spot anomalies and explain forecasts directly from numbers. [ChatTime](https://arxiv.org/abs/2412.11376) treats each numeric value as a “foreign-language” token, expanding the vocabulary of a compact 1 B LLM and tuning only the new layers (\~350 M weights); this lighter recipe still delivers zero-shot forecasting and bidirectional text/series dialogue. The pair illustrates the trade-off: full SFT for top conversational quality versus selective re-training for speed and memory savings. ([arxiv.org][3], [arxiv.org][4])


[3]: https://arxiv.org/abs/2412.03104 "ChatTS"
[4]: https://arxiv.org/abs/2412.11376 "ChatTime"

**Forecasting & reasoning with LLMs in the loop.**
[TimeXL](https://arxiv.org/abs/2503.01013) couples a tiny prototype encoder with a predict–reflect–refine trio of GPT-4-class models, gaining both accuracy and natural-language rationales while training <1 % of the system. [CAPTime](https://arxiv.org/abs/2505.10774) freezes its base LLM entirely and bolts on a mixture-of-distribution experts to deliver calibrated, multimodal probabilistic forecasts. [SMETimes](https://arxiv.org/abs/2505.10774) argues that if you must train an LLM, a sub-3 B “Small Language Model” plus lightweight adapters hits the sweet spot—fitting on a single A100 yet rivalling 7 B baselines. Finally, [TimeCMA](https://arxiv.org/abs/2406.01638) shows a dual-encoder cross-modality alignment trick: it retrieves robust prompt embeddings from a frozen GPT-2-XL to guide a slim time-series encoder, cutting MTS error without big compute. Collectively these works demonstrate that most performance gains now come from clever interfaces to largely frozen LLMs, with full fine-tuning reserved for resource-aware “small-but-mighty” models. ([arxiv.org][5], [arxiv.org][6], [arxiv.org][7], [arxiv.org][8])


[5]: https://arxiv.org/abs/2503.01013 "TimeXL"
[6]: https://arxiv.org/abs/2505.10774 "CAPTime"
[7]: https://arxiv.org/abs/2503.03594 "SMETimes"
[8]: https://arxiv.org/abs/2406.01638 "TimeCMA"


Only ChatTS, ChatTime, SMETimes weights are released.


### Discriminative Representation

The representation that can be used in `GCN (Generalized Category Discovery)` ([GCN](https://arxiv.org/abs/2201.02609), [SelEx](https://arxiv.org/abs/2408.14371)). 

Contrastive learning, Sparse autoencoder or older method such as [DEC (Deep Embedded Clustering)](https://arxiv.org/abs/1511.06335), SOM (Self Organizing Maps).

#### Characteristics of Time Series

[Implicit Reasoning in Deep Time Series Forecasting](https://arxiv.org/pdf/2409.10840): It is observed that certain linear, MLP-based, and patch-based Transformer models generalize effectively in carefully structured out-of-distribution scenarios, suggesting underexplored reasoning capabilities beyond simple pattern memorization.