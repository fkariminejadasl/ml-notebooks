# Training with Resource Limitations

Training large deep learning models is notably resource-intensive, often presenting challenges in both memory and computational demands. In contrast, smaller models, while less demanding, may lack descriptive power. A practical approach to training larger models involves starting with a pre-trained model and then fine-tuning a small subset of parameters, typically the head of the model. If your training time is limited to just a few hours, it's advisable to checkpoint the model and resume training from the last saved checkpoint. Below, we present a variety of techniques designed to either reduce memory usage or/and enhance computational efficiency.

- **Training**: Gradient accumulation and gradient checkpointing. **Gradient accumulation** involves accumulating gradients over multiple mini-batches before updating model parameters. It's useful when the available memory is insufficient for a desired batch size. **Gradient checkpointing** reduces memory usage by not saving intermediate tensors required for the backward pass. These tensors are recomputed during the backward pass, which increases computation time.
- **Fine-Tuning Tricks**: Fine-tune only a small number of parameters (PEFT), e.g., LoRA/controlNet.
- **Specific to Attention Blocks in Transformers**: FlashAttention, Flash-decoding.
- **Tricks for GPU**: Half-precision, quantization, paged optimizers (GPU to CPU transfer used in QLoRA for optimizer states). Examples are: fp32 -> fp16 -> int8 -> nf4 (normal float 4-bit).

#### N.B.

- Memory consists of parameters (weights), gradients, optimizer states, and activations (batch size x largest layer).
- QLoRA freezes and quantizes the main model and adds a low-rank adapter (LoRA).

#### References

- Fine-tuning of 7B model parameters on T4 from DeepLearning AI by Ludwig, presented by Travis Addair (watch from [here](https://youtu.be/g68qlo9Izf0?t=793) to [here](https://youtu.be/g68qlo9Izf0?t=2184).
- [LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)

