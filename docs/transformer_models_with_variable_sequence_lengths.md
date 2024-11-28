# Handling Variable Sequence Lengths in Transformer Models

This guide provides code examples for handling variable sequence lengths in Transformer models using positional encodings. It includes examples of both sinusoidal (fixed) and learned positional embeddings, along with considerations when training on sequences of varying lengths.

---

## Example 1: Transformer Model with Sinusoidal Positional Encoding

The following code demonstrates a Transformer model using sinusoidal positional encoding, which can handle variable sequence lengths without modification. 

This example is adapted from [this source](https://guyuena.github.io/PyTorch-study-Tutorials/beginner/transformer_tutorial.html).

```python
import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                             (-math.log(10000.0) / d_model))  # Shape: [d_model/2]

        pe = torch.zeros(max_len, d_model)  # Shape: [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]  # Add positional encoding
        return self.dropout(x)
        
class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, 
                 d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, 
                                                 d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor of shape [seq_len, batch_size]
            src_mask: Tensor of shape [seq_len, seq_len]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

def generate_square_subsequent_mask(sz: int) -> Tensor:
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# Initialize the model
ntokens = 1000  # Vocabulary size
d_model = 512   # Embedding dimension
nhead = 8       # Number of attention heads
d_hid = 2048    # Feedforward network dimension
nlayers = 6     # Number of Transformer layers
dropout = 0.5   # Dropout rate

model = TransformerModel(ntokens, d_model, nhead, d_hid, nlayers, dropout)

# Prepare input data with variable sequence length
seq_len = 10    # Sequence length can vary
batch_size = 2  # Batch size
x = torch.randint(0, ntokens, (seq_len, batch_size))  # Random input

src_mask = generate_square_subsequent_mask(seq_len)

# Run the model
output = model(x, src_mask)
```

---

## Example 2: Transformer Model with Learned Positional Embeddings

In this example, positional embeddings are learned parameters, allowing the model to potentially capture position-specific patterns.

```python
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class LearnedPositionalEncoding(nn.Module):

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        seq_len = x.size(0)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(1).expand(seq_len, x.size(1))
        position_embeddings = self.pos_embedding(position_ids)
        return x + position_embeddings
```

---

## Example 3: Transformer Model with Learned Positional Embeddings Initialized with Sinusoidal Positional Encoding

In this example, positional embeddings are learned parameters initialized with Sinusoidal Positional Encoding.


```python
class SinusoidalInitializedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        # Create a learnable positional embedding parameter
        self.pos_embedding = nn.Parameter(torch.zeros(max_len, d_model))

        # Initialize with sinusoidal positional encoding
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))  # [d_model/2]

        sinusoidal_embedding = torch.zeros(max_len, d_model)  # [max_len, d_model]
        sinusoidal_embedding[:, 0::2] = torch.sin(position * div_term)  # Even indices
        sinusoidal_embedding[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        # Assign the sinusoidal values to the parameter without tracking gradients
        with torch.no_grad():
            self.pos_embedding.copy_(sinusoidal_embedding)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        seq_len = x.size(0)
        x = x + self.pos_embedding[:seq_len, :].unsqueeze(1)
        return x
```

---

## Considerations When Training with Learned Positional Embeddings

Training a model with learned positional embeddings (PE) where the majority of sequence lengths are significantly shorter than the specified `max_len` can indeed present challenges:

- **Underrepresentation of Longer Positions**: If most training sequences are short, embeddings corresponding to higher positions (i.e., positions near `max_len`) receive minimal updates during training. This lack of exposure can lead to poor generalization for longer sequences during inference, as the model hasn't adequately learned representations for these positions.

- **Inefficient Resource Utilization**: Allocating parameters for positions up to `max_len` consumes memory and computational resources. If these positions are seldom used during training, this allocation becomes inefficient.

### Mitigation Strategies:

- **Dynamic Positional Embeddings**: Instead of a fixed `max_len`, employ dynamic positional embeddings that adjust based on the actual sequence lengths encountered during training. This approach ensures that the model learns appropriate embeddings for the positions it processes.

- **Curriculum Learning**: Start training with shorter sequences and progressively introduce longer ones. This method helps the model gradually adapt to various sequence lengths, ensuring that embeddings for higher positions are adequately trained.

- **Data Augmentation**: Artificially increase the length of training sequences by padding or concatenating sequences. This technique exposes the model to a broader range of positions, aiding in the learning of embeddings across the entire range up to `max_len`.

- **Regularization Techniques**: Apply regularization methods to prevent overfitting to shorter sequences, encouraging the model to generalize better to longer sequences.

---

## Summary

- **Sinusoidal Positional Encoding**: Handles variable sequence lengths naturally without the need for learned parameters tied to specific positions.

- **Learned Positional Embeddings**: Require careful consideration of sequence length distribution in the training data to ensure embeddings for all positions are adequately trained.

- **Training Strategies**: Adjust your data and training process, not the model code, to handle variable sequence lengths effectively.

