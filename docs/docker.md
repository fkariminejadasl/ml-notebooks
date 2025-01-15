# DOCKER

## Installation for Ubuntu

More information can be found on the ["engine install ubuntu"](https://docs.docker.com/engine/install/ubuntu).

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install the latest version
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

More information can be found on the [linux postinstall](https://docs.docker.com/engine/install/linux-postinstall).

```bash
# Create the `docker` group. The Docker group already exists, so there is no need to run this command.
sudo groupadd docker

# Add your user to the docker group.
sudo usermod -aG docker $USER

# Activate the changes to groups. 
# I think it is better to restart the system than to run this command. I encountered a weird issue when I installed Docker Compose.
newgrp docker

# Verify docker
docker run hello-world
```

## Uninstall Docker Engine

More information can be found on the [official website](https://docs.docker.com/engine/install/ubuntu).

```bash
sudo apt-get purge docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras

sudo rm -rf /var/lib/docker
sudo rm -rf /var/lib/containerd

sudo rm /etc/apt/sources.list.d/docker.list
sudo rm /etc/apt/keyrings/docker.asc
```

## Basic Commands

Here's a guide to essential Docker commands, followed by a cheat sheet for quick reference.

### Listing Docker Images and Containers

#### List Images: 

To view all Docker images on your system:

```bash
docker images
```

This displays a table of images with details like repository, tag, and image ID.

#### List Running Containers: 

To see all currently running containers:

```bash
docker ps
```

This shows a list of active containers with their IDs, names, and statuses.

#### List All Containers

To list all containers, including those that are stopped:

```bash
docker ps -a
```
This provides a comprehensive list of all containers, regardless of their state.

To display the full command without truncation:

```
docker ps -a --no-trunc
```

### Managing Docker Images

#### Pull an Image

To download an image from Docker Hub:

```bash
docker pull <image_name>
```

Replace `<image_name>` with the desired image, e.g., `ubuntu`.

#### Remove an Image

To delete a specific image:

```bash
docker rmi <image_name_or_id>
```

Use the image name or ID from the `docker images` list.

### Building Docker Images

#### Build an Image from a Dockerfile

To create a Docker image from a Dockerfile:

```bash
docker build -t <image_name> <path_to_dockerfile_directory>
```

Replace `<image_name>` with your desired image name and `<path_to_dockerfile_directory>` with the path to the directory containing your Dockerfile. The `-t` flag tags the image with a name.

Example:

```bash
docker build -t myapp:latest .
```
  
This builds an image named `myapp` with the tag `latest` from the Dockerfile in the current directory.

### Running and Managing Containers

#### Run a Container

To create and start a new container:
  
```bash
docker run [OPTIONS] <image_name>
```

Common options include:
- `-d`: Run the container in detached mode (in the background).
- `-it`: Run the container in interactive mode with a terminal.
- `--name <container_name>`: Assign a name to the container.
- `-p <host_port>:<container_port>`: Map host port to container port.
- `-v <host_directory>:<container_directory>`: Mount a host directory as a volume in the container.

Example:

```bash
docker run -it bird-behavior bash
```
  
Example:

This runs a container named `bird-behavior` and opens an interactive bash shell inside the `bird-behavior` container.

```bash
docker run -d -p 8080:80 -v /host/data:/container/data --name mynginx nginx
```

This runs an Nginx container named `mynginx` in detached mode, mapping port 8080 on the host to port 80 in the container, and mounts the host directory `/host/data` to `/container/data` in the container.

**Resource constraints**: For more details check [here](https://docs.docker.com/engine/containers/resource_constraints).

```bash
docker run --cpus="4" --memory="8g" ...
```

#### Execute Commands in a Running Container

To run a command inside a running container:
  
```bash
docker exec [OPTIONS] <container_name_or_id> <command>
 ```

Common options:
- `-it`: Run in interactive mode with a terminal.

Example:

```bash
docker exec -it mynginx /bin/bash
```

This opens an interactive bash shell inside the `mynginx` container.

#### Stop a Running Container

To stop a container:

```bash
docker stop <container_name_or_id>
```

#### Start a Stopped Container

To start a container that has been stopped:
  
```bash
docker start <container_name_or_id>
```

#### Remove a Container

To delete a container:
  
```bash
docker rm <container_name_or_id>
```
Note: Ensure the container is stopped before removing it.

### Viewing Logs and Inspecting Containers

#### View Container Logs

To see the logs of a container:
  
```bash
docker logs <container_name_or_id>
```

#### Inspect Container Details

To get detailed information about a container:
  
```bash
docker inspect <container_name_or_id>
```

## Docker Compose

Docker Compose is a tool that simplifies running applications with multiple containers. By defining services, networks, and volumes in a single YAML file, you can start and manage all components of your application with one command. 

You can find an example of a Docker Compose file [here](https://github.com/fkariminejadasl/prev_homepage_academicpages/blob/main/docker-compose.yml). The description is provided below.

### Creating a `docker-compose.yml` File

Create a `docker-compose.yml` file with the following content:

```yaml
version: '3'
services:
  jekyll:
    image: jekyll/jekyll:latest
    command: jekyll serve --watch --incremental
    ports:
      - "4000:4000"
    volumes:
      - .:/srv/jekyll
```

### Running the Docker Compose File

To start the services defined in your `docker-compose.yml` file, run:

```bash
docker-compose up
```

## Cheat Sheet

| Command                                      | Description                                      |
|----------------------------------------------|--------------------------------------------------|
| `docker images`                              | List all Docker images                           |
| `docker ps`                                  | List running containers                          |
| `docker ps -a`                               | List all containers (running and stopped)        |
| `docker pull <image_name>`                   | Pull an image from Docker Hub                    |
| `docker rmi <image_name_or_id>`              | Remove a Docker image                            |
| `docker build -t <image_name> <path>`        | Build an image from a Dockerfile                 |
| `docker run [OPTIONS] <image_name>`          | Run a new container                              |
| `docker exec [OPTIONS] <container> <command>`| Execute a command in a running container         |
| `docker stop <container_name_or_id>`         | Stop a running container                         |
| `docker start <container_name_or_id>`        | Start a stopped container                        |
| `docker rm <container_name_or_id>`           | Remove a container                               |
| `docker logs <container_name_or_id>`         | View logs of a container                         |
| `docker inspect <container_name_or_id>`      | Inspect detailed information of a container      |


## Installing the NVIDIA Container Toolkit

Follow these instructions from [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to enable GPU support in Docker containers:

First, back up your NVIDIA packages in case anything breaks during installation:

```bash
dpkg --get-selections | grep -i nvidia > ~/nvidia-packages-backup.txt
```

Then proceed with the installation steps below.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
```

#### Configuring Docker with NVIDIA Runtime

1. After installing the NVIDIA Container Toolkit, configure Docker to use the NVIDIA runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

This command updates `/etc/docker/daemon.json` to enable NVIDIA GPU support in containers. The file will contain:

```json
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```

2. Restart the Docker daemon:

```bash
sudo systemctl restart docker
```

3. Verify the installation:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-sm
```

#### Rollback Installation

```bash
# Rollback Installation
# Remove the Nvidia GPG Key
sudo rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# Remove the Nvidia Container Source List
sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list
# Refresh the Package List
sudo apt update
sudo apt remove --purge nvidia-container-toolkit
sudo apt autoremove

# Reinstall Your Nvidia Drivers
# If your GPU stops working, reinstall the driver listed in your backup (e.g., nvidia-driver-535):
sudo apt install --reinstall nvidia-driver-535  # adjust version to match your backup

# If you had issues with missing dependencies after reinstalling the driver, running the full 
# reinstall command ensures everything is restored.
cat ~/nvidia-packages-backup.txt | awk '{print $1}' | xargs sudo apt install --reinstall -y

# Reboot and Check
sudo reboot
nvidia-smi
```

## Apptainer

In Snellius, there is Apptainer instead of Docker.

Apptainer (formerly Singularity) is a containerization tool designed for high-performance computing (HPC), scientific workloads, and secure application deployment. Unlike Docker, Apptainer focuses on security, reproducibility, and portability, allowing users to run containers without requiring root privileges. It uses single-file SIF (Singularity Image Format) images, making it ideal for environments like HPC clusters and supercomputers.

The equivalent commands for the Docker commands you mentioned are:

| Command                                      | Description                                      |
|----------------------------------------------|--------------------------------------------------|
| `apptainer cache list`                       | `docker images`                                  |
| `apptainer instance list`                    | `docker ps -a`                                   |
| `apptainer cache clean`                      | `docker rmi imageid`. See example below.         |
| `docker rm containerid`                      | `apptainer instance stop <instance_name>`        |

These directories store the downloaded images.

```bash
ls ~/.apptainer/cache/library
ls ~/.apptainer/cache/oci-tmp
```

E.g. If you pulled docker using apptainer pull, it should have been saved as a `.sif` file in your working directory 

```bash
apptainer pull docker://godlovedc/lolcow 
apptainer inspect lolcow_latest.sif
# Since Apptainer doesn’t use a centralized image store like Docker, you typically just remove the .sif file:
rm lolcow_latest.sif
# If you want to clean the cache (which includes OCI blobs and temporary files), use:
apptainer cache clean
```

