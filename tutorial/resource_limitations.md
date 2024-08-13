# Training with Resource Limitations

Training large deep learning models is notably resource-intensive, often presenting challenges in both memory and computational demands. In contrast, smaller models, while less demanding, may lack descriptive power. A practical approach to training larger models involves starting with a pre-trained model and then fine-tuning a small subset of parameters, typically the head of the model. If your training time is limited to just a few hours, it's advisable to checkpoint the model and resume training from the last saved checkpoint. Below, we present a variety of techniques designed to either reduce memory usage or/and enhance computational efficiency.

- **Training**: Gradient accumulation and gradient checkpointing. **Gradient accumulation** involves accumulating gradients over multiple mini-batches before updating model parameters. It's useful when the available memory is insufficient for a desired batch size. **Gradient checkpointing** reduces memory usage by not saving intermediate tensors required for the backward pass. These tensors are recomputed during the backward pass, which increases computation time.
- **Fine-Tuning Tricks**: Fine-tune only a small number of parameters (PEFT), e.g., LoRA/controlNet.
- **Specific to Attention Blocks in Transformers**: FlashAttention, Flash-decoding.
- **Tricks for GPU**: Half-precision, quantization, paged optimizers (GPU to CPU transfer used in QLoRA for optimizer states). Examples are: fp32 -> fp16 -> int8 -> nf4 (normal float 4-bit). [Example mixed precision training in pytroch](https://pytorch.org/docs/stable/notes/amp_examples.html)

# Inference with Resource Limitations
- **Model Parameters**: 
    - Model distillation: Distill a large model as a teacher model to a student model using distillation loss.
    - Quantization techniques: Weight clustering (aka low-bit parallelization) is a compression technique.

#### N.B.

- Memory consists of parameters (weights), gradients, optimizer states, and activations (batch size x largest layer).
- QLoRA freezes and quantizes the main model and adds a low-rank adapter (LoRA).

#### References

- Fine-tuning of 7B model parameters on T4 from DeepLearning AI by Ludwig, presented by Travis Addair (watch from [here](https://youtu.be/g68qlo9Izf0?t=793) to [here](https://youtu.be/g68qlo9Izf0?t=2184).
- [LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)


#### Example code

**Using fp16 (float16) in PyTorch:**

```python
import torch
from torch.cuda.amp import GradScaler, autocast

# Initialize model, optimizer, and other components
model = MyModel().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

scaler = GradScaler()

for inputs, labels in data_loader:
    inputs, labels = inputs.cuda(), labels.cuda()

    optimizer.zero_grad()
    
    # Casts operations to mixed precision
    with autocast():
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
    
    # Scales the loss and calls backward()
    scaler.scale(loss).backward()
    
    # Unscales gradients and calls optimizer step
    scaler.step(optimizer)
    scaler.update()
```

**Using bf16 (bfloat16) in PyTorch:**
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Check if bf16 is supported
if torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    raise RuntimeError("Bfloat16 not supported on this device")

# Initialize model, optimizer, and other components
model = MyModel().to(dtype=dtype, device='cuda')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for inputs, labels in data_loader:
    inputs, labels = inputs.to(dtype=dtype, device='cuda'), labels.to(device='cuda')

    optimizer.zero_grad()
    
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    
    loss.backward()
    optimizer.step()
```