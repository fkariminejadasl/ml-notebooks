## Deep Learning Project Setup

To establish a foundational deep learning project that includes data loading, model definition, training with validation, and hyperparameter tuning, consider the following structured approach. This setup is modular, facilitating scalability and maintenance.

### Folder Structure

```
project/
│
├── data/               # Data loading scripts
│   ├── __init__.py
│   └── dataset.py      # Custom dataset script
│
├── models/             # Model architectures
│   ├── __init__.py
│   └── model.py        # Model definition
│
├── configs/            # Configuration files
│   └── config.yaml     # Hyperparameter configurations
│
├── scripts/            # Training and evaluation scripts
│   └── train.py        # Training script
│
├── utils/              # Utility functions
│   ├── __init__.py
│   └── utils.py        # Helper functions (e.g., logging, metrics)
│
├── main.py             # Entry point of the project
└── requirements.txt    # List of required packages
```

### Dataset Loader

```python
# data/dataset.py
import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "data": torch.tensor(self.data[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
```

### Model Definition

```python
# models/model.py
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)
```

### Training Script with Validation

```python
# scripts/train.py
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.dataset import SimpleDataset
from models.model import SimpleModel
from sklearn.metrics import accuracy_score

def train_model(config, dataset):
    # Determine sizes for training and validation sets
    total_size = len(dataset)
    val_size = int(total_size * config['val_split'])
    train_size = total_size - val_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Model, Loss, Optimizer
    model = SimpleModel(input_size=dataset[0]['data'].shape[0], num_classes=len(set(dataset.labels)))
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])

    # TensorBoard Summary Writer
    writer = SummaryWriter(log_dir=config['log_dir'])

    # Training Loop
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            data, labels = batch['data'], batch['label']

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['epochs']}], Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                data, labels = batch['data'], batch['label']
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch [{epoch+1}/{config['epochs']}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Log metrics to TensorBoard
        writer.add_scalar('Training Loss', avg_train_loss, epoch + 1)
        writer.add_scalar('Validation Loss', avg_val_loss, epoch + 1)
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch + 1)

    writer.close()
```

### Main Entry

```python
# main.py
import yaml
import numpy as np
from data.dataset import SimpleDataset
from scripts.train import train_model

if __name__ == "__main__":
    # Load configurations
    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Dummy data (replace with real data loading)
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, 1000)

    # Create dataset
    dataset = SimpleDataset(X, y)

    # Train the model
    train_model(config, dataset)
```

### Configuration

Ensure your configuration file includes the `log_dir` for TensorBoard logs and a `val_split` parameter to define the proportion of data used for validation.

```yaml
# configs/config.yaml
batch_size: 16
learning_rate: 0.001
epochs: 10
log_dir: 'runs/experiment_1'
val_split: 0.2  # 20% of data used for validation
```

### Running the Code

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Training**:
   ```bash
   python main.py
   ```

3. **Launch TensorBoard**:
   ```bash
   tensorboard --port 4004 --logdir=runs
   ```
   Open the provided URL in your browser to access the TensorBoard dashboard.

By integrating PyTorch's `random_split` function, you can effectively partition your dataset into training and validation sets, facilitating model evaluation without the need for external libraries. 

## Simplified Hyperparameter Tuning

For a more concise approach to hyperparameter tuning, consider using a configuration file to define multiple sets of hyperparameters and iterate over them. Here's an example:

#### Define Hyperparameter Sets:

Create a YAML file listing different hyperparameter configurations:
```yaml
# configs/hyperparams.yaml
experiments:
    - batch_size: 16
    learning_rate: 0.001
    epochs: 5
    log_dir: 'runs/exp1'
    - batch_size: 32
    learning_rate: 0.01
    epochs: 10
    log_dir: 'runs/exp2'
```

#### Modify the Main Script:
Update your `main.py` to load and iterate over these configurations:
```python
# main.py
import yaml
from scripts.train import train_model

if __name__ == "__main__":
    # Load hyperparameter configurations
    with open("configs/hyperparams.yaml", "r") as file:
        experiments = yaml.safe_load(file)['experiments']

    # Iterate over each configuration
    for idx, config in enumerate(experiments):
        print(f"Running experiment {idx + 1}/{len(experiments)} with config: {config}")
        train_model(config)
```

This approach allows you to manage multiple experiments efficiently, with each configuration's results logged separately for easy comparison.

#### References:
- [How to use TensorBoard with PyTorch](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)
- [Visualizing Models, Data, and Training with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) 