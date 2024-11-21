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

**1. Listing Docker Images and Containers**

- **List Images**: To view all Docker images on your system:
  ```bash
  docker images
  ```
  This displays a table of images with details like repository, tag, and image ID.

- **List Running Containers**: To see all currently running containers:
  ```bash
  docker ps
  ```
  This shows a list of active containers with their IDs, names, and statuses.

- **List All Containers**: To list all containers, including those that are stopped:
  ```bash
  docker ps -a
  ```
  This provides a comprehensive list of all containers, regardless of their state.

**2. Managing Docker Images**

- **Pull an Image**: To download an image from Docker Hub:
  ```bash
  docker pull <image_name>
  ```
  Replace `<image_name>` with the desired image, e.g., `ubuntu`.

- **Remove an Image**: To delete a specific image:
  ```bash
  docker rmi <image_name_or_id>
  ```
  Use the image name or ID from the `docker images` list.

**3. Building Docker Images**

- **Build an Image from a Dockerfile**: To create a Docker image from a Dockerfile:
  ```bash
  docker build -t <image_name> <path_to_dockerfile_directory>
  ```
  Replace `<image_name>` with your desired image name and `<path_to_dockerfile_directory>` with the path to the directory containing your Dockerfile. The `-t` flag tags the image with a name.

  Example:
  ```bash
  docker build -t myapp:latest .
  ```
  This builds an image named `myapp` with the tag `latest` from the Dockerfile in the current directory.

**4. Running and Managing Containers**

- **Run a Container**: To create and start a new container:
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
  docker run -d -p 8080:80 -v /host/data:/container/data --name mynginx nginx
  ```
  This runs an Nginx container named `mynginx` in detached mode, mapping port 8080 on the host to port 80 in the container, and mounts the host directory `/host/data` to `/container/data` in the container.

- **Execute Commands in a Running Container**: To run a command inside a running container:
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

- **Stop a Running Container**: To stop a container:
  ```bash
  docker stop <container_name_or_id>
  ```

- **Start a Stopped Container**: To start a container that has been stopped:
  ```bash
  docker start <container_name_or_id>
  ```

- **Remove a Container**: To delete a container:
  ```bash
  docker rm <container_name_or_id>
  ```
  Note: Ensure the container is stopped before removing it.

**5. Viewing Logs and Inspecting Containers**

- **View Container Logs**: To see the logs of a container:
  ```bash
  docker logs <container_name_or_id>
  ```

- **Inspect Container Details**: To get detailed information about a container:
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





