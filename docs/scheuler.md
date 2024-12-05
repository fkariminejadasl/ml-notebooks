Here is a sample example to demonstrate how the scheduler changes the learning rate. You may choose to use a different [scheduler](https://pytorch.org/docs/stable/optim.html).

```python
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Define a dummy model
model = nn.Linear(10, 1)

# Define optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)

# TensorBoard writer
writer = SummaryWriter("result/tensorboard/run1")

# Dummy training loop
num_epochs = 2500
steps_per_epoch = 10

for epoch in tqdm(range(num_epochs)):
    for step in range(steps_per_epoch):
        # Dummy input and loss
        inputs = torch.randn(32, 10)
        targets = torch.randn(32, 1)
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log learning rate to TensorBoard
        current_lr = round(optimizer.param_groups[-1]["lr"], 6)
        global_step = epoch * steps_per_epoch + step
        writer.add_scalar("Learning Rate", current_lr, global_step)
        
    # Update the scheduler
    scheduler.step()

    lr_optim = round(optimizer.param_groups[-1]["lr"], 6)
    lr_sched = scheduler.get_last_lr()[0]
    writer.add_scalar("lr/optim", lr_optim, epoch)
    writer.add_scalar("lr/sched", lr_sched, epoch)


# Close the writer
writer.close()
```