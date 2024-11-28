# Gradio in SRC (SURF Research Cloud)

#### Create a Workspace
In the SRC Dashboard, create an `Ubuntu 2204 - SUDO enabled` machine. 

#### Install Miniconda
Follow the instructions [here](https://github.com/fkariminejadasl/ml-notebooks/blob/main/tutorial/python.md#setup-python) or below:

```bash
mkdir /scratch/venv
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /scratch/venv/miniconda.sh
bash /scratch/venv/miniconda.sh -b -u -p /scratch/venv/
rm /scratch/venv/miniconda.sh
/scratch/venv/bin/conda init bash
```

Open a new terminal.

#### Setup Virtual Environment

```bash
conda create -n p310 python=3.10
conda activate p310
```

Add the `conda activate p310` command to `~/.bashrc`.

#### Install Bird Classification App

In `/scratch`:

```bash
git clone https://github.com/fkariminejadasl/bird-behavior.git
pip install .[app]
```

#### Setup Nginx

```bash
sudo apt install nginx
```

Create a `testgradio` file with the content below by using `sudo vi /etc/nginx/sites-available/testgradio`. For more detailed information, refer to [Running Gradio on Your Web Server with Nginx](https://www.gradio.app/guides/running-gradio-on-your-web-server-with-nginx).

> Find the domain name in `/etc/hosts`.  

```nginx
server {
    listen 80;
    server_name your_domain_name;  # This is in /etc/hosts

    location / {  # Change this if you'd like to serve your Gradio app on a different path
        proxy_pass http://127.0.0.1:7860/; # Change this if your Gradio app will be running on a different port
        proxy_buffering off;
        proxy_redirect off;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Since `include /etc/nginx/sites-enabled/*;` is added in the `/etc/nginx/nginx.conf` file, we need to create a symbolic link:

```bash
sudo ln -s /etc/nginx/sites-available/testgradio /etc/nginx/sites-enabled
```

#### Setup Gradio App as a Service
Create the service file using `sudo vi /etc/systemd/system/gradio.service`:

```ini
[Unit]
Description=Service to launch Gradio app
After=network.target

[Service]
User=fkariminej
Group=www-data
Environment="CONDA_BASE=/scratch/venv"
Environment="CONDA_ENV_NAME=p310"
WorkingDirectory=/scratch/bird-behavior/app
ExecStart=/bin/bash -c "source $CONDA_BASE/bin/activate $CONDA_ENV_NAME && python testgradio.py"

[Install]
WantedBy=multi-user.target
```

#### Enable Services
Handy commands in `systemctl`: status, start, enable, disable, stop.

```bash
sudo systemctl status nginx
sudo systemctl status gradio
```

#### Acknowledgement
This document is modified from the original document by Berend Wijers and loosely follows the [Serve Python App on Nginx](https://blog.devgenius.io/serve-python-app-on-nginx-6bc57ceaed4c).
