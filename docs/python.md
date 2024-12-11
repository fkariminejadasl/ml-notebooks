## Setup Python

Look at the guide in [here](https://docs.anaconda.com/miniconda/#quick-command-line-install) or follow bellow steps.

Install conda:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

After installing, initialize your newly-installed Miniconda.
```bash
~/miniconda3/bin/conda init bash
```

Setup your python:
```bash
conda create -n some_name python=3.10
conda activate some_name
```
You can put `conda activate some_name` in your `~/.bashrc` to activate it when open a new terminal.

Remove your conda environment:
```bash
conda remove -n some_name --all
```

## Remove Conda
To completely remove Miniconda from your Ubuntu system, including all associated configurations and files, follow these steps:

#### Jupyter Kernel
```bash
# This process installs a new Jupyter kernel named test with the display name "bird".
pip install ipykernel
sudo python -m ipykernel install --name test --display-name bird

# Location of the Installed Kernel
jupyter kernelspec list
# This command will display a list of all installed kernels along with their corresponding paths. 
#  test       /home/your_username/.local/share/jupyter/kernels/test

# Removing the Installed Kernel
jupyter kernelspec uninstall test
```

#### Deactivate Any Active Conda Environments

Ensure that no Conda environments are active by running:

```bash
conda deactivate
```

#### Remove the Miniconda Installation Directory

Delete the directory where Miniconda is installed. By default, this is `~/miniconda3`. Use the following command, adjusting the path if necessary:

```bash
rm -rf ~/miniconda3/
```

*Note: Replace `~/miniconda3/` with the correct path if you installed Miniconda elsewhere.*

#### Remove Conda-Related Hidden Files and Directories

Remove hidden files and directories in your home directory that Conda uses:

```bash
rm -rf ~/.conda ~/.condarc ~/.continuum
```

*These directories store Conda environments and settings.*

#### Remove Conda Initialization from Shell Configuration

Conda adds initialization code to your shell's configuration file (e.g., `.bashrc`). To remove these lines:

- Open the configuration file in a text editor:

  ```bash
  nano ~/.bashrc
  ```

- Scroll to the section managed by 'conda init', which looks like:

  ```
  # >>> conda initialize >>>
  # !! Contents within this block are managed by 'conda init' !!
  __conda_setup="$('/home/username/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
  if [ $? -eq 0 ]; then
      eval "$__conda_setup"
  else
      if [ -f "/home/username/miniconda3/etc/profile.d/conda.sh" ]; then
          . "/home/username/miniconda3/etc/profile.d/conda.sh"
      else
          export PATH="/home/username/miniconda3/bin:$PATH"
      fi
  fi
  unset __conda_setup
  # <<< conda initialize <<<
  ```

- Delete this entire block.

- Save the file and exit the editor (in nano, press `CTRL+O` to save and `CTRL+X` to exit).

- Apply the changes by sourcing the file:

  ```bash
  source ~/.bashrc
  ```

#### Remove Any Remaining Conda-Related Cache

Check for and remove any remaining Conda-related cache files:

```bash
rm -rf ~/.cache/conda
```

 ## Python Courses

From https://software-carpentry.org/lessons, below courses are offered. 
- [Programming with Python](https://swcarpentry.github.io/python-novice-inflammation/)
- [Plotting and Programming in Python](https://swcarpentry.github.io/python-novice-gapminder)

For more on software engineering side, you could also attend this course:
- [Intermediate Research Software Development with Python](https://www.esciencecenter.nl/event/intermediate-research-software-development-with-python-3)



