> I want to understand diffusion models. Can you explain them with code? I want to understand both training and inference.


Yes. Here is a practical mental model first:

A diffusion model learns to reverse a gradual corruption process.

You start with a real sample \(x_0\), like an image.  
During training, you add a little Gaussian noise again and again until it becomes almost pure noise.  
The model is trained to undo that corruption.

So there are two processes:

1. **Forward process**: slowly add noise to data.
2. **Reverse process**: learn to remove noise step by step.

---

## 1. The big idea

Suppose \(x_0\) is a clean image.

At timestep \(t\), we create a noisy version \(x_t\).

A standard forward process is:

\[
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t; \sqrt{1-\beta_t}\,x_{t-1}, \beta_t I\right)
\]

where \(\beta_t\) is a small variance schedule.

Instead of simulating step by step, we can directly sample \(x_t\) from \(x_0\):

\[
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon
\]

where:

- \(\alpha_t = 1 - \beta_t\)
- \(\bar{\alpha}_t = \prod_{s=1}^t \alpha_s\)
- \(\epsilon \sim \mathcal{N}(0, I)\)

This equation is the core trick behind training.

---

## 2. What the model learns

The model usually does **not** predict the clean image directly.

Instead, it predicts the noise that was added:

\[
\epsilon_\theta(x_t, t)
\]

So during training:

- sample a clean image \(x_0\)
- sample a random timestep \(t\)
- sample random Gaussian noise \(\epsilon\)
- build \(x_t\)
- ask model to predict \(\epsilon\)

Loss:

\[
L = \mathbb{E}\left[\left\lVert \epsilon - \epsilon_\theta(x_t, t) \right\rVert^2\right]
\]

That simple MSE loss is the standard DDPM training objective.

---

## 3. Why this works

At high noise levels, \(x_t\) is almost random.  
At lower noise levels, \(x_t\) still contains structure.

By training on many noise levels, the model learns how images look at every corruption stage.  
Then, at inference time, you start from pure noise and repeatedly denoise.

So generation is:

- start with \(x_T \sim \mathcal{N}(0, I)\)
- predict noise
- remove some noise
- repeat until \(x_0\)

---

# 4. Minimal PyTorch code for training

This is a compact educational implementation, not production code.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
```

## Noise schedule

```python
T = 1000

beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T, device=device)

alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
```

## Helper to gather timestep-specific values

```python
def extract(a, t, x_shape):
    """
    a: tensor of shape [T]
    t: tensor of shape [B]
    returns values reshaped to [B, 1, 1, 1, ...] for broadcasting
    """
    batch_size = t.shape[0]
    out = a.gather(0, t)
    return out.reshape(batch_size, *([1] * (len(x_shape) - 1)))
```

## Forward noising

```python
def q_sample(x0, t, noise=None):
    """
    Sample x_t from q(x_t | x_0)
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    """
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_ab_t = extract(sqrt_alpha_bars, t, x0.shape)
    sqrt_1mab_t = extract(sqrt_one_minus_alpha_bars, t, x0.shape)

    return sqrt_ab_t * x0 + sqrt_1mab_t * noise
```

---

# 5. A very small denoiser model

Real systems use a U-Net with attention and timestep embeddings.  
For learning, here is a tiny CNN.

```python
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class SimpleDenoiser(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(base_channels, in_channels, 3, padding=1)

        self.time_proj1 = nn.Linear(time_dim, base_channels)
        self.time_proj2 = nn.Linear(time_dim, base_channels)
        self.time_proj3 = nn.Linear(time_dim, base_channels)

    def forward(self, x, t):
        temb = self.time_mlp(t)

        h = self.conv1(x)
        h = h + self.time_proj1(temb)[:, :, None, None]
        h = F.relu(h)

        h = self.conv2(h)
        h = h + self.time_proj2(temb)[:, :, None, None]
        h = F.relu(h)

        h = self.conv3(h)
        h = h + self.time_proj3(temb)[:, :, None, None]
        h = F.relu(h)

        return self.conv4(h)
```

---

# 6. Training step

This is the heart of diffusion training.

```python
model = SimpleDenoiser(in_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

```python
def training_step(x0):
    """
    x0: clean batch, assumed scaled to [-1, 1]
    """
    model.train()
    B = x0.shape[0]

    t = torch.randint(0, T, (B,), device=device)
    noise = torch.randn_like(x0)
    xt = q_sample(x0, t, noise)

    pred_noise = model(xt, t)

    loss = F.mse_loss(pred_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

If you had a dataloader:

```python
for epoch in range(10):
    for x0, _ in dataloader:
        x0 = x0.to(device) * 2 - 1  # if original images are in [0,1]
        loss = training_step(x0)
    print(f"epoch={epoch}, loss={loss:.4f}")
```

---

# 7. What inference does

At inference, we do the reverse diffusion process.

We start from random noise:

```python
x = torch.randn(batch_size, channels, height, width)
```

Then for \(t = T-1, T-2, \dots, 0\):

- predict noise with the model
- compute the mean of the reverse distribution
- optionally add a little noise except at the final step

In DDPM, one standard reverse update is:

\[
x_{t-1} =
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \,\epsilon_\theta(x_t, t)
\right)
+ \sigma_t z
\]

where \(z \sim \mathcal{N}(0, I)\) if \(t > 0\).

---

# 8. Minimal sampling code

```python
@torch.no_grad()
def p_sample(model, x, t):
    """
    One reverse step: x_t -> x_{t-1}
    t is an integer scalar timestep
    """
    betas_t = betas[t]
    alphas_t = alphas[t]
    alpha_bars_t = alpha_bars[t]

    sqrt_one_minus_alpha_bars_t = torch.sqrt(1 - alpha_bars_t)
    recip_sqrt_alpha_t = torch.rsqrt(alphas_t)

    t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
    pred_noise = model(x, t_batch)

    model_mean = recip_sqrt_alpha_t * (
        x - (betas_t / sqrt_one_minus_alpha_bars_t) * pred_noise
    )

    if t > 0:
        noise = torch.randn_like(x)
        sigma_t = torch.sqrt(betas_t)
        return model_mean + sigma_t * noise
    else:
        return model_mean
```

```python
@torch.no_grad()
def sample(model, shape):
    model.eval()
    x = torch.randn(shape, device=device)

    for t in reversed(range(T)):
        x = p_sample(model, x, t)

    return x
```

Usage:

```python
samples = sample(model, (16, 1, 28, 28))
samples = (samples.clamp(-1, 1) + 1) / 2
```

---

# 9. Training vs inference in plain language

## Training

You already have real images.

For each image:

- choose a random noise level \(t\)
- corrupt the image to get \(x_t\)
- ask the model to guess the noise that was added
- update weights to reduce prediction error

The model sees examples of denoising at every noise level.

## Inference

You do not have a real image.

Instead:

- begin with random noise
- use the model to predict what part is noise
- remove that noise a little
- repeat many times

Eventually, random noise becomes a structured image.

---

# 10. Why timestep conditioning is necessary

The same noisy image value can mean different things depending on how much noise was added.

So the model must know the timestep \(t\).

That is why the model input is not just \(x_t\), but \((x_t, t)\).

Without \(t\), the task is ambiguous.

---

# 11. What a real diffusion model changes

The minimal code above teaches the mechanics, but real systems usually add:

- **U-Net architecture**
- **attention layers**
- **better noise schedules**
- **classifier-free guidance** for text conditioning
- **latent diffusion** instead of pixel-space diffusion
- **faster samplers** such as DDIM, DPM-Solver, Euler samplers

But the core idea is still the same:  
learn to reverse gradual noising.

---

# 12. The most important formulas to remember

## Forward sample from clean image

\[
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
\]

## Model prediction

\[
\epsilon_\theta(x_t, t)
\]

## Training loss

\[
\left\lVert \epsilon - \epsilon_\theta(x_t, t) \right\rVert^2
\]

## Reverse step intuition

Use predicted noise to estimate a less noisy sample.

---

# 13. Intuition for what the model is actually learning

A useful intuition:

At each timestep, the model learns the direction from a noisy point back toward the data manifold.

So if noise gives you a point floating in image space, the model says:  
"from here, move this way to look more like a real image."

Repeated many times, this becomes generation.

---

# 14. Tiny end-to-end example on MNIST

This is still educational, but more concrete.

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

model = SimpleDenoiser(in_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    for x0, _ in dataloader:
        x0 = x0.to(device)
        x0 = x0 * 2 - 1  # scale to [-1,1]

        B = x0.shape[0]
        t = torch.randint(0, T, (B,), device=device)
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, noise)

        pred_noise = model(xt, t)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: loss={loss.item():.4f}")
```

Sampling:

```python
samples = sample(model, (16, 1, 28, 28))
samples = (samples.clamp(-1, 1) + 1) / 2
```

---

# 15. Common confusion points

## "Why predict noise instead of the image?"

Because the objective is simple and stable. In practice, predicting noise works very well.

## "Why random timestep during training?"

Because one batch can teach the model denoising at many different corruption levels.

## "Why is inference slow?"

Because generation needs many reverse steps. Faster samplers reduce the number of steps.

## "Is this like an autoencoder?"

Not exactly. Autoencoders directly compress and reconstruct. Diffusion models learn iterative denoising.

---

# 16. One compact pseudocode summary

## Training

```text
for x0 in data:
    t ~ Uniform({1,...,T})
    eps ~ Normal(0, I)
    xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
    pred = model(xt, t)
    loss = mse(pred, eps)
    update model
```

## Inference

```text
xT ~ Normal(0, I)
x = xT
for t = T, T-1, ..., 1:
    eps_pred = model(x, t)
    x = reverse_step(x, eps_pred, t)
return x
```

