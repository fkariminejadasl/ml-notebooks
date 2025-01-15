# Time Series

### Time-Series Representation Learning

- [TF-C](https://zitniklab.hms.harvard.edu/projects/TF-C): Time-Frequency Consistency (TF-C) Model (Harvard MIMS Lab, NeurIPS 2022) - A cross-domain time-series representation model that leverages contrastive learning between time-domain and frequency-domain views of the same signal​. By enforcing consistent embeddings in both domains, TF-C learns general features transferable to diverse sensors (EEG, accelerometer, etc.). Weights are available.
- [TS-TCC](https://arxiv.org/abs/2106.14112): Time-Series Representation Learning via Temporal and Contextual Contrasting: contrastive. augmentation: jitter, scale the amplitude and permutation.
- [TimesURL](https://arxiv.org/abs/2312.15709): Learning Universal Representations of Time Series via Contrastive Learning: contrastive and reconstruction based (MAE) based on **TS2Ve** model. augmentation: left-righ cropping with frequency swaping of the negative example. hard negatives are temporal and instance based. In temporal way, the augmentation mix in different time. In the instance-based augmentation, this mixing occurs in mixing the different instances.
- [TS2Vec](https://arxiv.org/abs/2106.10466): Towards universal representation of time series. Augmentation: croping
- [CPC](https://arxiv.org/abs/1807.03748): Representation Learning with Contrastive Predictive Coding: The contrastive learning works such as SimCLR, MoCo, SwAV, DINO are based on the contrastive loss introduced in this paper. The postive are the predictions.
- [TNC](https://arxiv.org/abs/2106.00750) (Temporal Neighborhood Coding): Triplet loss (neighbor positive and further negative)

### Novel Category Discovery

- [GCD](https://arxiv.org/pdf/2201.02609): Generalized Category Discovery. Use supervised contrastive and self-supervised contrastive for representation learning. Then use KMean with forcing the labeled data stays in a cluster. Hungarian matching assignment defines the accuracy of clustering on the labeled data, where this accuracy defines the number of clusters. 

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


### Discriminative Representation

The representation that can be used in `GCN (Generalized Category Discovery)` ([GCN](https://arxiv.org/abs/2201.02609), [SelEx](https://arxiv.org/abs/2408.14371)). 

Contrastive learning, Sparse autoencoder or older method such as [DEC (Deep Embedded Clustering)](https://arxiv.org/abs/1511.06335), SOM (Self Organizing Maps).

#### Characteristics of Time Series

[Implicit Reasoning in Deep Time Series Forecasting](https://arxiv.org/pdf/2409.10840): It is observed that certain linear, MLP-based, and patch-based Transformer models generalize effectively in carefully structured out-of-distribution scenarios, suggesting underexplored reasoning capabilities beyond simple pattern memorization.