# Training with Resource Limitations

Training large deep learning models is notably resource-intensive, often presenting challenges in both memory and computational demands. In contrast, smaller models, while less demanding, may lack descriptive power. A practical approach to training larger models involves starting with a pre-trained model and then fine-tuning a small subset of parameters, typically the head of the model. If your training time is limited to just a few hours, it's advisable to checkpoint the model and resume training from the last saved checkpoint. Below, we present a variety of techniques designed to either reduce memory usage or/and enhance computational efficiency.

- **Training**: Gradient accumulation, gradient checkpointing and CPU offloading. **Gradient accumulation** involves accumulating gradients over multiple mini-batches before updating model parameters. It's useful when the available memory is insufficient for a desired batch size. **Gradient checkpointing** (also known as activation checkpointing)) reduces memory usage by not saving intermediate tensors required for the backward pass. These tensors are recomputed during the backward pass, which increases computation time. **CPU offloading** stores weights in CPU RAM rather than on the GPU when they are not in use.
- **Fine-Tuning Tricks**: Fine-tune only a small number of parameters (PEFT), e.g., LoRA/controlNet.
- **Specific to Attention Blocks in Transformers**: FlashAttention, Flash-decoding.
- **Tricks for GPU**: Half-precision, quantization, paged optimizers (GPU to CPU transfer used in QLoRA for optimizer states). Examples are: fp32 -> fp16/bf16 -> int8 -> nf4 (normal float 4-bit).

# Inference with Resource Limitations
- **Model Parameters**: 
    - Model distillation: Distill a large model as a teacher model to a student model using distillation loss.
    - Quantization techniques: Weight clustering (aka low-bit parallelization) is a compression technique.

#### N.B.

- Memory consists of parameters (weights), gradients, optimizer states, and activations (batch size x largest layer).
- QLoRA freezes and quantizes the main model and adds a low-rank adapter (LoRA).

#### References

- Fine-tuning of 7B model parameters on T4 from DeepLearning AI by Ludwig, presented by Travis Addair (watch from [here](https://youtu.be/g68qlo9Izf0?t=793) to [here](https://youtu.be/g68qlo9Izf0?t=2184).
- [train a 70b language model on two 24GB GPUs](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html): an open source system, based on FSDP and QLoRA, that can train a 70b model on two 24GB GPUs. They also used Gradient checkpointing, CPU offloading, and Flash Attention 2.
- [LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- [SURF course on Profiling](https://github.com/sara-nl/HPML-course-materials)


#### Example code

**Using fp16 (float16) in PyTorch:**

The detail explanation is in [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html) and [Example mixed precision training in pytroch](https://pytorch.org/docs/stable/notes/amp_examples.html).

```python
import torch

# Initialize model, optimizer, and other components
model = MyModel().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

scaler = torch.GradScaler()

for inputs, labels in data_loader:
    inputs, labels = inputs.cuda(), labels.cuda()

    optimizer.zero_grad()
    
    # Casts operations to mixed precision
    with torch.autocast(device_type="cuda", dtype=torch.float16)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
    
    # Scales the loss and calls backward()
    scaler.scale(loss).backward()
    
    # Unscales gradients and calls optimizer step
    scaler.step(optimizer)
    scaler.update()
```

**Using bf16 (bfloat16) in PyTorch:**

It can be the same as float16, without using scaler, or follow the code below.

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

## Pytorch Profiling

The PyTorch Profiler is a tool that allows developers to understand and optimize their PyTorch code by analyzing its performance. Here's an example of setting up and using the PyTorch Profiler:

### Code Example with PyTorch Profiler

For more details take look at [Tensorboard Profiler Tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html).

#### Code Example

Here is a step-by-step example of setting up and using the PyTorch Profiler. 

```python
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torch.profiler import profile, record_function, ProfilerActivity

# Set up a model and input data
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generate dummy input data
input_data = torch.randn(8, 3, 224, 224).to(device)  # Batch of 8 images

# Define the profiling configuration
with profile(
    activities=[
        ProfilerActivity.CPU,  # Monitor CPU activity
        ProfilerActivity.CUDA  # Monitor CUDA activity (if applicable)
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),  # Save data for TensorBoard
    record_shapes=True,  # Record tensor shapes
    with_stack=True  # Capture stack traces
) as prof:

    # Use record_function for specific profiling scopes
    with record_function("model_inference"):
        output = model(input_data)  # Run the model inference

# Analyze the profiler output
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Visualize the profiler output using TensorBoard:
# Run `tensorboard --logdir=./log` in the terminal
```

- **`profile` Context Manager**: This manages the profiling session and specifies which activities (CPU, CUDA) to profile.
- **`record_function`**: Labels a specific code block for profiling, so you can see its performance separately.
- **`tensorboard_trace_handler`**: Saves the profiling results in a format compatible with TensorBoard.
- **`key_averages()`**: Aggregates and summarizes profiling results for analysis in the console.


You can customize the profiler to include:
- Custom intervals: Use `schedule` to specify profiling start and stop.
- Memory profiling: Set `profile_memory=True` to track memory usage.
- Exporting results: Save results to file using `prof.export_chrome_trace("trace.json")`.

Notes: Use smaller models or batches for testing, as profiling large models can generate a lot of data.

#### Visualize the profiler output

After generating a trace, simply drag the `trace.json` generated in `log` file (example above) into [Perfetto UI](https://ui.perfetto.dev) or in chrome browser by typing `chrome://tracing` to visualize your profile.

The TensorBoard integration with the PyTorch profiler is now deprecated. But if you still want to use TensorBoard you should install `pip install torch_tb_profiler` and then use `tensorboard --logdir=./log`


## References

- [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [What Every User Should Know About Mixed Precision Training in PyTorch](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)