#!/bin/bash
# 1. Update and install basic dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y ca-certificates curl gnupg lsb-release

# 2. Install NVIDIA Drivers (535 is stable for CUDA 12.1)
sudo apt install -y nvidia-driver-535 nvidia-utils-535
# Note: A reboot is usually required here for the driver to load.

# 3. Install Docker
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io

# 4. Install NVIDIA Container Toolkit (The bridge for GPU in Docker)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit

# 5. Configure Docker to use NVIDIA and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "Setup complete. PLEASE REBOOT YOUR EC2 NOW: sudo reboot"