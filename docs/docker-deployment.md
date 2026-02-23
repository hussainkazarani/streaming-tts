## Deployment Guide

This document outlines the exact steps required to provision the host environment, configure credentials, and run the different project milestones (Reviews 2, 3, and 4) using Docker.

## 0. Prerequisites
- An AWS EC2 Instance with an NVIDIA GPU (e.g., G4dn or G5 series) running Ubuntu.
- A Hugging Face account with access granted to the `meta-llama/Llama-3.2-1B` and `sesame/csm-1b` model.
- An SSH key (`.pem` file) to access your EC2 instance.

## 1. First-Time Host Setup (EC2)
Before running the Docker containers, the EC2 host must have the NVIDIA drivers and Docker Container Toolkit installed. You only need to do this **once** per fresh EC2 instance.

  1. SSH into your EC2 instance.
  2. Navigate to the root of the cloned project repository.
  3. Run the included setup script and then reboot the instance:
     ```bash
     bash setup_host.sh
     sudo reboot
     ```

## 2. Environment Configuration
The project uses environment variables for secure credential management.
```bash
# Copy the provided example env file
cp .env.example .env
# Open the file to add your Hugging Face token
vim .env
# Update the HF_TOKEN value, save, and exit
```

## 3. Running the Project Reviews
**IMPORTANT:** All docker build and docker run commands must be executed from the root directory of the project(`tts-streaming`) so the build context includes all necessary folders.

### Review 2: File-Based Generation (Offline)
This stage demonstrates the core TTS engine generating a static `.wav` file locally. We use a bind mount so the generated file saves directly to the EC2 host.

### Build
```
sudo docker build -t tts-r2 -f deployment/Dockerfile.R2 .
```
### Run
```bash
# The -v flag maps your current host directory to /app inside the container
sudo docker run --rm \
  --name tts-r2-container \
  --gpus all \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd):/app \
  tts-r2 \
  /bin/bash # only add if you want to go into bash
```
### Result
Check your current directory for final_conversation.wav.

## Review 3: Basic Streaming Architecture
This stage demonstrates the high-performance WebSockets streaming engine using a default voice.

### Build
```bash
sudo docker build -t tts-r3 -f deployment/Dockerfile.R3 .
```

### Run
```bash
sudo docker run -d \
  --name tts-r3-container \
  --gpus all \
  -p 8000:8000 \
  --env-file .env \
  --cap-add=SYS_NICE \
  -v $(pwd):/app \
  tts-r3
```

## Review 4: Full-Stack Real-Time Streaming & Voice Cloning
The final production architecture. Includes custom voice loading, modular components, and the polished UI.

### Build
```bash
sudo docker build -t tts-r4 -f deployment/Dockerfile.R4 .
```

### Run
```bash
sudo docker run -d \
  --name tts-r4-container \
  --gpus all \
  -p 8000:8000 \
  --env-file .env \
  --cap-add=SYS_NICE \
  -v $(pwd):/app \
  tts-r4
```

## 4. Accessing the Web Interfaces (R3 & R4)
Because the EC2 instance is remote, we use SSH Local Port Forwarding to securely map the server's port 8000 to your local machine.

1. Open a new terminal on your local Mac/PC.
2. Run the SSH Tunnel command:
    ```bash
    ssh -L 8000:localhost:8000 -i "path/to/your-key.pem" ubuntu@<your-ec2-ip-address>
    ```
3. Open your browser and navigate to:
    ```
    http://localhost:8000
    ```

## 5. Remote GUI Management
Instead of managing containers strictly via the terminal, you can link your local Mac's Docker Desktop app directly to the EC2 engine over SSH. This allows you to view logs, start/stop containers, and drag-and-drop generated .wav files directly from the UI.

### Create the Context (Run on your Mac terminal)
```bash
Replace <ip> with your EC2 IP address
docker context create ec2-tts --docker "host=ssh://ubuntu@<ip>"
```

### Switch to the Remote Engine
```bash
docker context use ec2-tts
```

### Open Docker Desktop
You will now see your remote EC2 containers. To switch back to your local machine later, run `docker context use default`

## 6. Useful Docker Commands (Cheatsheet)

### Containers & Monitoring
1. View active containers: `docker ps`
2. View all containers (including stopped): `docker ps -a`
3. View live application logs: `docker logs -f <container_name>`
4. View logs (non-streaming): `docker logs <container_name>`

---

### Container Management
5. Stop a container: `docker stop <container_name>`
6. Remove a container: `docker rm <container_name>`

---

### Access & Debugging
7. Open bash inside a running container: `docker exec -it <container_name> /bin/bash`
8. Check GPU usage inside container: `docker exec -it <container_name> nvidia-smi`

---

### Images Management
9. List Docker images: `sudo docker images`
10. Remove a Docker image: `sudo docker rmi <image_name>:<tag>`