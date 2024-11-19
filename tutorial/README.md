# Access Snellius GPUs

### Large Compute
For larger amounts of compute, please refer to possible options in [NWO grants](https://servicedesk.surf.nl/wiki/display/WIKI/NWO+grants) or [Access to compute services](https://www.surf.nl/en/access-to-compute-services). 

### Small or Medium Compute
For more information, visit [RCSS: Research Capacity Computing Services](https://servicedesk.surf.nl/wiki/display/WIKI/RCCS+contract). The
[SURF services and rates 2024](https://www.surf.nl/files/2023-08/surf-services-and-rates-2024_version-aug-2023.pdf) are provided here.

### Small Compute
The FNWI institute offers small compute resources about three times a year, with each allocation providing approximately 50K-100K SBUs. NWO also provides access to small compute resources.

Create a ticket at https://servicedesk.surf.nl under "Apply for access / Direct institute contract" or "Apply for access / Small Compute applications (NWO)."

Follow the instructions in the [Setup](#setup) section to create an account.


## Setup 

**Create an account**

https://portal.cua.surf.nl : first copied public key in here (only done once)

In the case of an issue, create in https://servicedesk.surf.nl a ticket under "Servicedesk / create a ticket" or email servicedesk@surf.nl. 
**Usage**

Use the Snellius (similar for e.g. sshfs/scp):
```bash
ssh -X username@snellius.surf.nl
```

## Setup environment
The first time to setup your environment, run below script:
```bash
module purge # unload all modules
module load 2022
module load Anaconda3/2022.05 # version 4.12.0
conda init bash
```

After that, the basic virtualenv from conda can be created. See below e.g.:
```bash
conda create -n test python=3.8
conda activate test
pip install torch # version 2.0.0+cu117
```

## Schedule Tasks
SLURM is a job scheduler used by many computer clusters and supercomputer, such as Snellius. It allocates resources to users and monitors work. It is a configurable workload manager. `squeue`, `sbatch`, `srun`, `sinfo`, and `scancel` are examples of the most commonly used commands in SLURM.

## Use GPUs

**Run a job:**

NB. The run file should be executable. Make it executable with `chmod a+x runfile.sh`.
```bash
sbatch runfile.sh
```
e.g. runfile.sh:
```bash
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=14:00:00
#SBATCH -o yolo8_train4_%j.out

echo "gpus $SLURM_GPUS on node: $SLURM_GPUS_ON_NODE"
echo "nodes nnodes: $SLURM_NNODES, nodeid: $SLURM_NODEID, nodelist $SLURM_NODELIST"
echo "cpus on node: $SLURM_CPUS_ON_NODE per gpu $SLURM_CPUS_PER_GPU per task $SLURM_CPUS_PER_TASK omp num thread $OMP_NUM_THREADS"
echo "tasks per node $SLURM_TASKS_PER_NODE pid $SLURM_TASK_PID"

# activate your environment
source $HOME/.bashrc
conda activate test # your conda venv

echo "start training"
yolo detect train data=/home/username/data/data8_v1/data.yaml model=/home/username/exp/runs/detect/bgr23/weights/best.pt imgsz=1920 batch=8 epochs=100 name=bgr cache=true close_mosaic=0 augment=True rect=False mosaic=1.0 mixup=0.0
echo "end training"
```
More SBATCH options and the "output environmental variables" can be found from the [sbatch help](https://slurm.schedmd.com/sbatch.html). 

</br>

**Check job is running**

User squeue with job id or username. 
```bash
squeue -j jobid
squeue -u username
# squeue with more options
squeue -o "%.10i %.9P %.25j %.8u %.8T %.10M %.9l %.6D %.10Q %.20S %R"
```
If the job is running, it will save the result in the output file with the name specified by `SBATCH -o` option. NB. `%j` in the name replaced by job id. In the example `yolo8_train4_%j.out`, the output file will be olo8_train4_2137977.out. The job id is the id you get after running sbatch.

> **IMPORTANT**</br>
Each person has a limited budget in the unit of SBU (system billing unit). It is also listed as accounting weight factor in [Snellius partitions and accounting](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+partitions+and+accounting).
e.g. If you request for 1 A100 GPU, it is 1/4 node, which has 18 cores for 10 hours, the SBU is `128 * 10 = 1280`.
So in a `runfile.sh`, the basic slurm settings are as:
```bash
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=14:00:00
```
> NB. `--cpus-per-gpu` or `--cpus-per-task` is automatically set for 1/4 of node, which in `gpu` partition is 18. For more info, check [SBU calculating](https://servicedesk.surf.nl/wiki/display/WIKI/Estimating+SBUs).

</br>

There are more options for variables such as below. You can get the full list from the [sbatch help](https://slurm.schedmd.com/sbatch.html). 

```bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=18
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
```

> **NB**: Jobs that require more resources and or running for a long time (walltime: `SBATCH --time`) are not easily scheduled. Try to first test if everything is OK by running one or two epochs, then request for resources. Moreover, estimate the time your experiment runs by roughly calculating how long each epoch takes and multiply by epochs and then increase this time for a bit counting for caching, data transfer.

</br>

**Check finished jobs**

```bash
sacct -j jobid -o "JobID,JobName,MaxRSS,Elapsed,NNodes,NodeList"
```
More options are in the [sacct help page](https://slurm.schedmd.com/sacct.html).

</br>

**Run tensorboard from remote computer**
Connect to Snellius and map your local port to remote port:
```bash
ssh -X username@snellius.surf.nl -L your_local_port:127.0.0.1:remote_port
```
In Snellius machine, run tensorboard with the remote port:
```bash
tensorboard --logdir_spec=18:/home/username/exp18,12:/home/username/exp12 --port remote_port # remote_port = 60011
```
Now, in the local machine, run `http://localhost:local_port`, e.g. `http://localhost:36999`. 

</br>

**Interactive mode**

> **IMPORTANT**: This part is not recommended. Use it if you want to have a short check and want to use a bash.

This is same as sbatch with runfile.sh but parameters are set in srun. In this way, the bash will be active and you are in the machine.

```bash
srun --gpus=1 --partition=gpu --time=00:10:00 --pty bash -il
```

**Multigpu in single or multi node**
For every GPU there is 18 CPU no matter if specified in slurm or not. Slurm, batch scheduler, will ignore if another value for CPU is specified. Each task runs on one CPU. So `ntasks-per-node` or `ntasks` are the same here. Apparently, `with OMP_NUM_THREADS=4`, or other value, we can tell torchrun to use 4 threads per CPU.

Basically, only specifiying number of gpus and the partition is enough. Below example uses 2 GPUs on a single Node, with 1 threads per CPU. 
```bash
#!/bin/bash
#SBATCH --gpus=2
#SBATCH --partition=gpu
#changing the  OMP_NUM_THREADS env variable is the same as --cpus-per-task
export OMP_NUM_THREADS=1
torchrun --node_rank=0 --nnodes=1 --nproc_per_node=2 ~/test/multigpu_torchrun.py 50 10
```
For more information on ddp (distributed data parallel) in pytorch, look at the [tutorial](https://pytorch.org/tutorials/beginner/ddp_series_intro.html).

**Wandb in Snellius**
First run `wandb init` before sending the job via sbatch. Then run the code which has `wandb.init(project=project_name)`. `Project_name` is wandb project.

**Useful commands**

- `sbatch`: run a job
- `squeue`: show the status of the job
- `scancel`: cancel the job. 
- `scontrol`: show detailed job information. e.g. show job detail: `scontrol show j 7605565`
- `sinfo`: get information about GPUs. e.g. `sinfo -e -o  "%9P %.6D %10X %4Y %24N %24f %32G"`, `sinfo -p gpu`
- `sacct`: get statistics on completed jobs
- `accinfo`, `budget-overview -p gpu`, `accuse`: show how much credite is left (Snellius commands)
- `myquota`: show the limit of files. They are also listed in [Snellius hardware](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+hardware) and [file systems](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+filesystems).
- `gpustat -acp`: show the gpu usage. It should be installed with pip, `pip install gpustat`. It has the information from `nvidia-smi`, but one-liner. 
- `module load/unload/purge/list/display/avail`: 
    - `load/unload/purge`: e.g. `module load CUDA/11.8.0`: load this module and use `unload` to unload this module. `purge` unload all modules.
    - `list`: e.g. `module list`: list of loaded modules. 
    - `display`: e.g. `module display CUDA/11.8.0`: show information on where this module is. 
    - `avail`: e.g. `module avail`: show list of all available modules, but first load 2022/2023/or higher version if available. 
   

Some examples are given in [Convenient Slurm commands](https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands). 

</br>

**Useful links:**
- [Snellius hardware](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+hardware) 
- [file systems](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+filesystems)
- [SBU calculating](https://servicedesk.surf.nl/wiki/display/WIKI/Estimating+SBUs)
- [Example job scripts](https://servicedesk.surf.nl/wiki/display/WIKI/Example+job+scripts)
- [Convenient Slurm commands](https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands)
- [Squeue help](https://slurm.schedmd.com/squeue.html): just use `squeue --help`
- [uvadlc: Working with the Snellius cluster](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial1/Lisa_Cluster.html)
- [microhh](https://microhh.readthedocs.io/en/latest/computing_systems/snellius.html)

# Details

#### SLRUM interactive mode

In the interactive mode, you can see slurm variables. E.g. `echo $$SLURM_MEM_PER_CPU`, `echo $SLURM_CPUS_PER_TASK`.
```bash
# interactive mode: request 2 CPUs (-c,--cpus-per-task), time can be unlimited (time=UNLIMITED)
srun -c2 --mem-per-cpu=200G --pty bash -il
# interactive mode: request 1 GPU
srun --gpus=1 --partition=gpu --time=00:10:00 --pty bash -il
```
#### SLURM list of common commands
```bash
#SBATCH --job-name=result # appears in squeue
#SBATCH -o exps/test_%j.out # -o,--output, %j=job id
#SBATCH --error=test_%j.err
#SBATCH --time=00:10:00 # time can be UNLIMITED
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=2 # -c,--cpus-per-task
#SBATCH --gpus=1 # number of gpu
#SBATCH --partition=gpu # type of gpu
```

Only these values are shown on snellius:
```bash
$SLURM_CPUS_ON_NODE:  the number of CPUs allocated to your job using the following environment variable:
$SLURM_GPUS: This gives you the total number of GPUs allocated to your job.
```

These values are empty on snellius:
```bash
$SLURM_MEM_PER_NODE: This variable gives you the total amount of memory allocated per node in megabytes (MB)
$SLURM_MEM_PER_CPU * $SLURM_CPUS_ON_NODE: in the case of requesting memory using --mem-per-cpu

$SLURM_GPUS_PER_NODE: This gives the number of GPUs per node allocated to your job.
$SLURM_JOB_GPUS: This gives you a list of the GPUs allocated to your job.

echo "cpu pertask: $SLURM_CPUS_PER_TASK"
echo "cpu per gpu: $SLURM_CPUS_PER_GPU"
```

#### SLURM task array
Example using task array:

```bash
#SBATCH --output=test_%A_%a.out # -o,--output, %A=job id, %a=array number
#SBATCH --array=0-1

python your_script.py $SLURM_ARRAY_TASK_ID
```

### Get CPU and GPU information

When the job is running, you can get number of GPU, CPU, RAM.
```bash
# job_id = 7605565
`scontrol show job 7605565`
```

#### CPU

``` bash
# RAM
# ---
# python
f"{psutil.virtual_memory().total:,}" # total of the node
# command line
free -h # total of the node
htop -u -d 0 # total of the node

# disk space
# ----------
f"{psutil.disk_usage('/').total:,}" # total of the node
# command line
df -h --total # total of the node

# Number of CPUs
# --------------
# python
len(os.sched_getaffinity(0))
# N.B: total cpu (72 for A100 not divided by 4)
psutil.cpu_count(logical=True) # total of the node
os.cpu_count() # total of the node
# command line
nproc
htop -u -d 0 # total of the node
```

#### GPU
```bash
# GPU VRAM
# --------
# python
f"{torch.cuda.get_device_properties(0).total_memory:,}"
# command line
nvidia-smi --query-gpu=memory.total --format=csv

# GPU type
# --------
torch.cuda.get_device_name(0) 
# command line
nvidia-smi -L

# Number of GPUs
# --------------
# python
torch.cuda.device_count()
# command line
nvidia-smi -L
```

### Sinfo

```bash
sinfo -e -o "%9P %.6D %10X %4Y %24N %24f %32G" | grep gpu_a
gpu_a100      36 4          18   gcn[37-72]               hwperf,scratch-node      gpu:a100:4(S:0-1),cpu:72
```

The `-e` option in sinfo stands for "extend" and is used to show information about all nodes, even those that are in states like down, drain, fail, or unknown. Without -e, sinfo typically only shows nodes that are available or idle.

Breaking down each field:
```
    %9P: Partition name, gpu_a100.
    %6D: Number of nodes, 36.
    %10X: Number of GPUs (sockets) per node, 4.
    %4Y:  Number of cores per GPU (socket), 18.
    %24N: List of nodes in this partition (in this case, gcn[37-72]).
    %24f: Features of the nodes (e.g., hwperf,scratch-node).
    %32G: Shows detailed information about GPU and CPU availability (e.g., gpu:a100:4(S:0-1),cpu:72).
```

```bash
sinfo | grep gpu_a
```
```
gpu_a100     up 5-00:00:00      5   resv gcn[47-49,69,71]
    gpu_a100 is the partition name.
    up 5-00:00:00 means the partition is up and available for 5 days.
    5 represents the number of nodes available.
    resv indicates that the nodes are reserved.
    gcn[47-49,69,71] specifies the nodes within the gpu_a100 partition.
```

# External GPUs

- [Cloud GPU comparison](https://cloud-gpus.com)
- [Vast.ai](https://vast.ai/pricing)
- Lambda Labs: [On demand](https://lambdalabs.com/service/gpu-cloud#pricing), [One, two & three year contracts](https://lambdalabs.com/service/gpu-cloud/reserved-cloud-pricing).
- [runpod.io](https://www.runpod.io/pricing) 
- [Paperspace](https://www.paperspace.com/pricing)
- [Jarvislabs](https://jarvislabs.ai/pricing)
- [HPC-AI](https://hpc-ai.com/)
- GCP
- AWS
