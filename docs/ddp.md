# DDP (Distributed Data Parallel) in PyTorch

This manual is from the [PyTorch DDP tutorial](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html). The code can be found [here](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html). This manual summarizes the changes needed when transitioning from a single GPU to multiple GPUs on a single node, both with and without torchrun, as well as multiple GPUs across multiple nodes. You can compare the code yourself using the `diff` command:

```bash
diff --color -U 0 multigpu.py multigpu_torchrun.py
```

## Single node multiple GPU

#### Imports

```python
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
```

#### Script function
```python
def main(rank, world_size, other_args):
    ddp_setup(rank, world_size)
    # define dataset, model, optimizer, trainer
    destroy_process_group()


world_size = torch.cuda.device_count()
mp.spawn(main, args=(world_size, other_args,), nprocs=world_size)
```

#### DDP setup
```python
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
init_process_group(backend="nccl", rank=rank, world_size=world_size)
torch.cuda.set_device(rank)
```

#### model
```python
- self.model = model.to(gpu_id)
+ self.model = DDP(model, device_ids=[gpu_id])
```

#### data
```python
train_data = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=32,
-   shuffle=True,
+   shuffle=False,
+   sampler=DistributedSampler(train_dataset),
)
```

#### Shuffling across multiple epochs
Calling the set_epoch() method on the DistributedSampler at the beginning of each epoch is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be used in each epoch.
```python
for epoch in epochs:
    train_data.sampler.set_epoch(epoch)
    for source, targets in train_data:
```

#### Save checkpoints
We only need to save model checkpoints from one process.
```python
- ckp = model.state_dict()
+ ckp = model.module.state_dict()
- if epoch % save_every == 0:
+ if gpu_id == 0 and epoch % save_every == 0:
```

#### Slurm job
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=2
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH -o test_multigpu_%j.out
python python_script.py arguments
# python multigpu.py 50 10
```

## Single node multiple GPU with torchrun

#### Script function
```python
-  world_size = torch.cuda.device_count()
-  mp.spawn(main, args=(world_size, other_args,), nprocs=world_size)
+  main(other_args)
```

#### DDP setup
torchrun provided environment variables `os.environ["LOCAL_RANK"]` for the GPU id:
```python
gpu_id = int(os.environ["LOCAL_RANK"])
```

```python
- os.environ["MASTER_ADDR"] = "localhost"
- os.environ["MASTER_PORT"] = "12355"
- init_process_group(backend="nccl", rank=rank, world_size=world_size)
+ init_process_group(backend="nccl")
+ torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
```

#### Slurm job
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=2
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=00:05:00
#SBATCH -o test_multigpu_torchrun_%j.out    
torchrun --nnodes=1 --nproc_per_node=2 python_script.py arguments
# torchrun --nnodes=1 --nproc_per_node=2 multigpu_torchrun.py 50 10
```

## Multi node multi GPU

The only diffeerence with previous one:
```
+ local_rank = int(os.environ["LOCAL_RANK"])
+ global_rank = int(os.environ["RANK"])
+ model = model.to(local_rank)

- local_rank = int(os.environ["LOCAL_RANK"]) # local_rank == gpu_id
- model = model.to(local_rank)
```

#### Slurm job
```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=00:05:00
#SBATCH -o test_multinode_%j.out  
torchrun --nnodes=2 --nproc_per_node=4 python_script.py arguments
# torchrun --nnodes=2 --nproc_per_node=1 multigpu_torchrun.py 50 10
```

## References

- https://medium.com/pytorch/training-a-1-trillion-parameter-model-with-pytorch-fully-sharded-data-parallel-on-aws-3ac13aa96cff
- https://medium.com/pytorch/pytorch-data-parallel-best-practices-on-google-cloud-6c8da2be180d
- https://medium.com/pytorch/pytorch-sessions-at-nvidia-gtc-march-20-2023-b86210711c9b
