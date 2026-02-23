## EC2 Deployment Guide (No Docker)

This guide outlines the exact steps to provision an Ubuntu EC2 instance, install NVIDIA drivers, set up Python 3.10, and run the streaming TTS engine directly on the host machine.

## 1. Initial System & GPU Setup
First, update the system and install the required NVIDIA drivers and CUDA toolkit.

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y nvidia-driver-535 nvidia-utils-535 nvidia-cuda-toolkit

# Reboot the instance to apply driver changes
sudo reboot
(Wait about 10 seconds for the instance to reboot, then reconnect via SSH.)
```

### 2. Python Environment Setup
Install Python 3.10 via the deadsnakes PPA and set up an isolated virtual environment.

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-pip

# Create and activate the virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Application Installation & Authentication
Install the project dependencies and authenticate with Hugging Face to download the required models.

```bash
pip install -r requirements.txt

# Authenticate Hugging Face CLI
huggingface-cli login --token <TOKEN_ID>
```

### 4. Running the Application (Admin/Sudo)
If your specific setup requires running the application with administrative privileges (e.g., binding to specific system ports or modifying priority), use the following commands to ensure sudo uses the correct virtual environment binaries.

```bash
# 1. Ensure you are authenticated in the sudo context
sudo $(which huggingface-cli) login --token <TOKEN_ID>

# 2. Activate the environment (if not already active)
source .venv/bin/activate

# 3. Run the server using the virtual environment's Python executable
sudo $(which python) backend/web_api/server.py
```

### 5. Accessing the Web UI (Port Forwarding)
Because the EC2 instance is remote, use SSH Local Port Forwarding to securely map the server's port 8000 to your local machine.

1. Open a new terminal on your local Mac/PC and run:
    ```bash
    ssh -L 8000:localhost:8000 -i "stream-aws.pem" ubuntu@54.166.150.162
    ```

2. Open your browser and navigate to:
    ```bash
    http://localhost:8000
    ````

### 6. Useful Monitoring Commands
Keep an eye on system resources using these commands (best run in a separate SSH window while the server is running):

#### Watch GPU Usage (Updates every second):
```bash
watch -n 1 nvidia-smi
```

#### Watch Disk Space (Updates every 5 seconds):
```bash
watch -n 5 df -h
```

#### Check Total CPU Threads:
```bash
python -c "import os; print(f'Total CPU Threads: {os.cpu_count()}')"
```