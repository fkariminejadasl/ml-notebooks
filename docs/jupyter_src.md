# Shared Jupyter Notebook in SRC (SURF Research Cloud)

In order to make a notebook available to all users, `conda`, a `virtual environment`, and a `Jupyter kernel` should be installed in a shared location, such as `/opt`, where all users have access.

#### Create a Workspace

Create a "Jupyter Notebook" workspace from "Create new workspace" and attach an external storage. The storage is mounted under "/data/storage_name". You can transfer data for example by scp:

```bash
scp -r your_data username@ip_address:/data/storage_name
```

#### Setup Conda for All Users

```bash
# Create a Directory for Conda in a System-Wide Location:
sudo mkdir -p /opt/miniconda3

# Download the Miniconda Installer:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

# Install Miniconda to the System-Wide Location:
sudo bash /tmp/miniconda.sh -b -u -p /opt/miniconda3

# Remove the Installer Script:
rm /tmp/miniconda.sh

# Set Permissions for All Users:
sudo chmod -R 755 /opt/miniconda3
sudo chown -R root:root /opt/miniconda3

# Add Conda to the PATH for All Users:
echo 'export PATH="/opt/miniconda3/bin:$PATH"' | sudo tee -a /etc/profile.d/conda.sh
# /opt/miniconda3/etc/profile.d/conda.sh

# Initialize Conda for All Users' Shells:
/opt/miniconda3/bin/conda init --all

# Verify the Installation:
conda --version
```

#### Make Kernel for All Users

```bash
# Activate Conda
source /opt/miniconda3/bin/activate

# Run Conda as Root Using Full Path
sudo /opt/miniconda3/bin/conda create -y -p /opt/miniconda3/envs/p310 python=3.10

# Set Permissions
sudo chmod -R 755 /opt/miniconda3/envs/p310
sudo chown -R root:root /opt/miniconda3/envs/p310

# Install ipykernel in the New Environment
conda activate /opt/miniconda3/envs/p310
sudo /opt/miniconda3/envs/p310/bin/pip install ipykernel

# Register a System-Wide Jupyter Kernel: Create a new kernel specification that all users can access
# This places the kernel spec in a system directory (e.g., /usr/local/share/jupyter/kernels/p310).
sudo /opt/miniconda3/envs/p310/bin/python -m ipykernel install --name p310 --display-name "bird"

# Permissions for Kernel Directory
sudo chmod -R 755 /usr/local/share/jupyter/kernels/p310

# Confirm the Installation: You should see p310 listed.
jupyter kernelspec list
```

#### Install Your Package for All Users

```bash
# Navigate to your desired directory
cd /scratch

# Clone the GitHub repo
git clone https://github.com/fkariminejadasl/bird-behavior.git
cd bird-behavior

# Checkout the specific branch
git checkout cleanup1

# Install the package into the p310 environment
sudo /opt/miniconda3/envs/p310/bin/pip install -e .
```


