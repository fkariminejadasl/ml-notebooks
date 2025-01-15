## MinkowskiEngine Installation

### On Snellius

`conda` should be installed first. This part doesn't require GPU machine and only done once.

```bash
module load 2023
module load Anaconda3/2023.07-2
conda init
```

After `conda init`, the terminal should be closed. So be careful, if you allocated the GPU, you will loss your GPUs, unless you do `ssh` to the allocated machine. E.g. `ssh gcn14`. 

> `MinkowskiEngine` requires GPU. You can ask for GPU by `salloc`.

#### CUDA 11

```bash
salloc --gpus=1 --partition=gpu --time=02:00:00

# MinkowskiEngine==0.5.4
conda create -n mink2 python=3.8
python activate mink2
module load 2022
module load CUDA/11.7.0 # CUDA/11.6.0 was giving error
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install openblas-devel -c anaconda

git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

Note that CUDA is backwards compatible, so `CUDA 11.1` can be also used for all packages.

```bash
pip install numpy==1.19.5
conda install pytorch=1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c nvidia
```

#### CUDA 12

Only runs on H100. `MinkowskiEngine` doesn't work before some changes in the `MinkowskiEngine` code. This fix is described here.

**Fix**: There is an issue described in [here](https://github.com/NVIDIA/MinkowskiEngine/issues/594#issuecomment-2294860926). This fix is based on [this link](https://github.com/NVIDIA/MinkowskiEngine/issues/543#issuecomment-1773458776).

- Get the code
```bash
git clone https://github.com/NVIDIA/MinkowskiEngine.git 
cd MinkowskiCu12
```

- Fix the code
```bash
1 - .../MinkowskiEngine/src/convolution_kernel.cuh

Add header:
#include <thrust/execution_policy.h>

2 - .../MinkowskiEngine/src/coordinate_map_gpu.cu

Add headers:

#include <thrust/unique.h>
#include <thrust/remove.h>

3 - .../MinkowskiEngine/src/spmm.cu

Add headers:

#include <thrust/execution_policy.h>
#include <thrust/reduce.h> 
#include <thrust/sort.h>

4 - .../MinkowskiEngine/src/3rdparty/concurrent_unordered_map.cuh

Add header:
#include <thrust/execution_policy.h>
```

- Installation

```bash
salloc --gpus=1 --partition=gpu_h100 --time=02:00:00

module load 2023
module load CUDA/12.1.1
conda create -n mink-cu12 python=3.8
conda activate mink-cu12
# https://pytorch.org/get-started/previous-versions/
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install openblas-devel -c anaconda

cd ~/dev/MinkowskiCu12
# Before this command apply the Fix. 
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

#### Docker CUDA 12 with the Fix 

This information is from [here](https://github.com/NVIDIA/MinkowskiEngine/issues/543#issuecomment-2371637255).

The base image is `nvidia/cuda:12.1.1-devel-ubuntu20.04` with `PyTorch 2.3.1` and `CUDA 12.1`.

```bash
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.9"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git /tmp/MinkowskiEngine \
    && cd /tmp/MinkowskiEngine \
    && sed -i '31i #include <thrust/execution_policy.h>' ./src/convolution_kernel.cuh \
    && sed -i '39i #include <thrust/unique.h>\n#include <thrust/remove.h>' ./src/coordinate_map_gpu.cu \
    && sed -i '38i #include <thrust/execution_policy.h>\n#include <thrust/reduce.h>\n#include <thrust/sort.h>' ./src/spmm.cu \
    && sed -i '38i #include <thrust/execution_policy.h>' ./src/3rdparty/concurrent_unordered_map.cuh \
    && python setup.py install --force_cuda --blas=openblas \
    && cd - \
    && rm -rf /tmp/MinkowskiEngine
```

#### Test MinkowskiEngine

```python
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import sys
sys.path.insert(0,"~/dev/MinkowskiEngine/tests/python")
from common import data_loader

class ExampleNetwork(ME.MinkowskiNetwork):
    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU())
        self.pooling = ME.MinkowskiGlobalPooling()
        self.linear = ME.MinkowskiLinear(128, out_feat)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pooling(out)
        return self.linear(out)

criterion = nn.CrossEntropyLoss()
net = ExampleNetwork(in_feat=3, out_feat=5, D=2)
print(net)

# a data loader must return a tuple of coords, features, and labels.
coords, feat, label = data_loader()
input = ME.SparseTensor(feat, coordinates=coords)
# Forward
output = net(input)

# Loss
loss = criterion(output.F, label)
```

## Install other packages For SegmentAnyTree

```bash
# open3d, torch-sparse, torch-spline-conv: very slow
# torch-points-kernels: different version of numpy, sklearn
# Issue with pip install torch-scatter, torch-sparse, torch-cluster, torch-geometric. 
# I use it from: https://github.com/prs-eth/PanopticSegForLargeScalePointCloud
pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-cluster==1.2.1 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html 
pip install torch-geometric==1.7.2

pip install -U pip
pip install hydra-core wandb tqdm gdown types-six types-requests h5py
pip install numpy scikit-image numba 
pip install matplotlib
pip install plyfile
pip install torchnet tensorboard
pip install open3d
pip install torch-scatter torch-sparse torch-cluster torch-geometric pytorch_metric_learning torch-points-kernels
TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6;9.0" FORCE_CUDA=1
pip install cython
pip install torch-spline-conv
pip install git+https://github.com/mit-han-lab/torchsparse.git # failed to install
```

## Run SegmentAnyTree From Docker

#### On a local computer

You need to install the NVIDIA Container Toolkit. Follow the instructions [here](docker.md#installing-the-nvidia-container-toolkit).

```bash
docker pull donaldmaen/segment-any-tree

docker run -it --gpus all --name SAT     --mount type=bind,source=/home/fatemeh/Downloads/tree/SAT/,target=/home/nibio/mutable-outside-world/bucket_in_folder     --mount type=bind,source=/home/fatemeh/Downloads/tree/SAT/out,target=/home/nibio/mutable-outside-world/bucket_out_folder   --mount type=bind,source=/home/fatemeh/Downloads/tree/ds,target=/home/datascience  donaldmaen/segment-any-tree
```

#### On Snellius

Since Snellius uses `apptainer` instead of Docker, follow these steps:

```bash
# On Snellius, the GPU container option is enabled. Run this command inside Apptainer to check if it detects the GPU:
# First, request a GPU with the following command:
salloc --gpus=1 --partition=gpu --time=01:00:00

# Test GPU visibility inside Apptainer:
apptainer run --nv docker://nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Pull the container image:
apptainer pull docker://donaldmaen/segment-any-tree

# Start an interactive shell session with Apptainer:
apptainer shell --nv  --bind /home/user/data/tree/sat:/home/nibio/mutable-outside-world/bucket_in_folder     --bind /home/user/data/tree/sat/output:/home/nibio/mutable-outside-world/bucket_out_folder  --bind /home/user/data/tree/ds:/home/datascience    /home/user/segment-any-tree_latest.sif

# Once inside Apptainer, run the following commands:
Apptainer> cd dev/SegmentAnyTree/
Apptainer> bash run_oracle_pipeline.sh
```

Note that if you use apptainer run instead of apptainer shell, it runs everything automatically.

## Install TreeLearn

This content is adapted from the [TreeLearn Pipeline Notebook](https://github.com/ecker-lab/TreeLearn/blob/main/TreeLearn_Pipeline.ipynb).

```bash
# Note that newer version laspy[lazrs] gives error 
# spconv-cu120 comes with warnings and then Floating point exception (core dumped).
conda create -n treelearn python=3.9
conda activate treelearn
pip install torch torch-scatter timm laspy[lazrs]==2.5.1 munch pandas plyfile pyyaml scikit-learn six tqdm open3d-cpu jakteristics shapely geopandas alphashape spconv-cu114 tensorboard tensorboardX
cd ~/dev
git clone https://github.com/ecker-lab/TreeLearn.git
cd TreeLearn
pip install -e .

mkdir -p ~/Downloads/tree/treelearn/checkpoints
cd ~/Downloads/tree/treelearn
mkdir -p pipeline/forests

wget https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/VPMPID/1JMEQV -O checkpoints/model_weights.pth
# https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/VPMPID/C1AHQW for model_weights_diverse_training_data.pth

wget https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/VPMPID/0WDXL6 -O pipeline/forests/plot_7_cut.laz

cd ~/dev/TreeLearn
python
```

For Snellius with CUDA11: 
```bash
module load 2022
module load Anaconda3/2022.05
conda init
# exit and ssh again then conda will be in .bashrc

module load 2022
module load CUDA/11.7.0

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
# pytorch 1.11 has issue with new numpy
pip install numpy==1.26
# I install torch-scatter first to see if there is an issue. 
pip install torch-scatter
# Install the rest as above
```

```python
# for python path
sys.path.append("/home/user/dev/TreeLearn/tools/pipeline")
config_path = "/home/user/dev/TreeLearn/configs/pipeline/pipeline.yaml"
config.forest_path = "/home/user/data/tree/treelearn/pipeline/forests/plot_7_cut.laz"
config.dataset_test.data_root = "/home/user/data/tree/treelearn/pipeline/forests/tiles"
config.pretrain = "/home/user/data/tree/treelearn/checkpoints/model_weights.pth"
```

```python
import sys
sys.path.append("/home/fatemeh/dev/TreeLearn/tools/pipeline")
from pipeline import run_treelearn_pipeline
import argparse, pprint
from tree_learn.util import get_config

config_path = "/home/fatemeh/dev/TreeLearn/configs/pipeline/pipeline.yaml"
config = get_config(config_path)

# adjust config
config.forest_path = "/home/fatemeh/Downloads/tree/treelearn/pipeline/forests/plot_7_cut.laz"
config.dataset_test.data_root = "/home/fatemeh/Downloads/tree/treelearn/pipeline/forests/tiles"
config.pretrain = "/home/fatemeh/Downloads/tree/treelearn/checkpoints/model_weights.pth"
config.tile_generation = True
config.sample_generation.stride = 0.9 # small overlap of tiles
config.shape_cfg.outer_remove = False # default value = 13.5
config.save_cfg.save_treewise = False
config.save_cfg.return_type = "voxelized_and_filtered"
print(pprint.pformat(config.toDict(), indent=2))

import logging
logger = logging.getLogger("TreeLearn")
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
logging.basicConfig()
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

run_treelearn_pipeline(config)
```

## Install ForAINet

Follow the rest of the [installation guide](https://github.com/prs-eth/PanopticSegForLargeScalePointCloud?tab=readme-ov-file#example-1-of-installation). However, a few changes need to be made. Follow the steps below:

- The guide mentions that `requirements.txt` should be installed, but the file is not provided. Additionally, `torch-cluster` should be installed from a separate file.
- `MinkowskiEngine` should be installed locally.  

```bash
# make requirements.txt from https://github.com/nicolas-chaulet/torch-points3d/blob/master/requirements.txt
# remove: omegaconf, hydra-core and replace by hydra-core>=1.0.0,<1.1.0
# remove: torch-scatter, torch-sparse, torch-cluster, torch-geometric, torch, torchvision, numpy
pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-geometric==1.7.2

# Don't do: export CUDA_HOME=/usr/local/cuda-11
# Don't do: pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" –install-option="--blas=openblas"
module load 2022
module load CUDA/11.7.0 # CUDA/11.6.0 was giving error
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

## Tools

[labelCloud](https://github.com/ch-sa/labelCloud) is installed both on python 3.9, 3.10 but run with errors. 

point cloud annotation tools:
https://chatgpt.com/share/67adeacf-bef8-8011-ba02-c59ae671126c

- cloudcompare for visualization: `sudo snap install cloudcompare`. Note that `sudo apt install cloudcompare` can't open `.laz` file.

## Miscellaneous 

#### GPU Compute Capability

You can define from the below code or from [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus).

```python
import torch
torch.cuda.get_device_capability()
```

H100: 9.0, A100: 8.6

#### CUDA modules available on Snellius

```bash
# module load 2024
CUDA/12.6.0
cuDNN/9.5.0.50-CUDA-12.6.0

# module load 2023
CUDA/12.1.1 
CUDA/12.4.0
cuDNN/8.9.2.26-CUDA-12.1.1

# module load 2022
CUDA/11.6.0    
CUDA/11.7.0
CUDA/11.8.0
cuDNN/8.4.1.50-CUDA-11.7.0
cuDNN/8.6.0.163-CUDA-11.8.0
```

#### CUDA modules paths on Snellius

To get the path information such as `CUDA_HOME`, `CUDA_PATH`, `PAHT`, `LIBRARY_PATH`, `LD_LIBRARY_PATH` use: 

```bash
module load CUDA/12.6.0
module display CUDA/12.6.0
```


### Tried but failed

These are my previos attemps to solve issues encountered. But resolvinig issues didn't install MinkowskiEngine. But it is worth mentioning them here.

#### pip DEPRECATION: --build-option and --global-option

This option doesn't work, even with the `--no-build-isolation` fix.

```bash
export CXX=g++
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-build-isolation -v --config-settings=build_option=--force_cuda --config-settings=blas_include_dirs=${CONDA_PREFIX}/include --config-settings=blas=openblas
```

MinkowskiEngine requires torch at build time, pip’s default build isolation can cause “Pytorch not found” errors. `--no-build-isolation` tells pip not to create a temporary/isolated environment at build time and instead to use the packages in your current conda/pip environment (where you have PyTorch installed).

#### Distutil issue with Numpy

`numpy.distutils` removed after numpy=1.24.

```bash
conda install "numpy=1.23.*" setuptools pip wheel
```

#### Find openbals

```bash
OPENBLAS=$(find ~/.conda/envs/test -name "libopenblas.so" | head -n 1)
export LD_LIBRARY_PATH=$(dirname $OPENBLAS):$LD_LIBRARY_PATH
pip install --no-cache-dir --force-reinstall numpy
```

```bash
# >>> np.__config__.show()
  "Build Dependencies": {
    "blas": {
      "name": "scipy-openblas",
      "found": true,
      "version": "0.3.28",
      "detection method": "pkgconfig",
      "include directory": "/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas64/include",
      "lib directory": "/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas64/lib",
      "openblas configuration": "OpenBLAS 0.3.28  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64",
      "pc file directory": "/project/.openblas"
    },
    "lapack": {
      "name": "scipy-openblas",
      "found": true,
      "version": "0.3.28",
      "detection method": "pkgconfig",
      "include directory": "/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas64/include",
      "lib directory": "/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas64/lib",
      "openblas configuration": "OpenBLAS 0.3.28  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64",
      "pc file directory": "/project/.openblas"
    }
  }
```

#### CUDA installation in Docker

```bash
# docker pull or From
# nvidia/cuda:12.6.0-runtime-ubuntu22.04
# nvidia/cuda:12.6.0-devel-ubuntu22.04
# nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# https://developer.nvidia.com/cuda-downloads
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
sh cuda_11.1.1_455.32.00_linux.run --silent --toolkit

export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.1
```

### References

- TreeLearn: ~30M parameters [github](https://github.com/ecker-lab/TreeLearn/tree/main), [paper](https://www.sciencedirect.com/science/article/pii/S1574954124004308)
- SegmentAnyTree: ~12M parameters, [paper](https://github.com/SmartForest-no/SegmentAnyTree)
- ForAINet: [paper](https://arxiv.org/pdf/2312.15084), [github](https://github.com/prs-eth/ForAINet/tree/main)

TreeLearn, SegmentAnyTree, and ForAINet are based on [PointGroup](https://arxiv.org/abs/2004.01658) model. NB [SoftGroup](https://arxiv.org/abs/2203.01509) is improved version of the [PointGroup](https://arxiv.org/abs/2004.01658).

[Point2Tree](https://arxiv.org/pdf/2305.02651) utilizes PointNet++ for semantic segmentation and employs a "Bayesian flow approach," a non-deep-learning method, for instance segmentation.

[FSCT](https://www.mdpi.com/2072-4292/13/8/1413)([code](https://github.com/SKrisanski/FSCT)) is a semantic segmentation model, seemingly based on PointNet++.

[TLS2Tree](https://github.com/philwilkes) performs instance segmentation of individual trees, leveraging the FSCT semantic segmentation model as a preprocessing step. Notably, it does not rely on deep learning.