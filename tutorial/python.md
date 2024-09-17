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


## Python Courses

From https://software-carpentry.org/lessons, below courses are offered. 
- [Programming with Python](https://swcarpentry.github.io/python-novice-inflammation/)
- [Plotting and Programming in Python](https://swcarpentry.github.io/python-novice-gapminder)

For more on software engineering side, you could also attend this course:
- [Intermediate Research Software Development with Python](https://www.esciencecenter.nl/event/intermediate-research-software-development-with-python-3)



