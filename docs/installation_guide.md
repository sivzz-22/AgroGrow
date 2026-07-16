# AgroGrow Installation Guide

Follow these steps to set up the environment and install dependencies for the AgroGrow system.

## System Requirements
- **Operating System**: Windows (tested on Windows 10/11)
- **Python**: version 3.10 or higher
- **GPU**: CUDA-compatible Nvidia Graphics Card with PyTorch drivers (highly recommended for deep learning training; falls back to CPU automatically for inference and dashboard tasks)

## Environment Preparation

### 1. Open Terminal or PowerShell
Navigate to the root directory where the workspace resides:
```bash
cd D:/Sem_7/"research paper 1"
```

### 2. Create the Virtual Environment
Create a clean Python virtual environment named `venv` inside the project space:
```bash
python -m venv venv
```

### 3. Activate the Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**:
  ```cmd
  .\venv\Scripts\activate.bat
  ```
- **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies
Upgrade pip and install all required modules using the requirements file:
```bash
python -m pip install --upgrade pip
pip install -r AgroGrow/requirements.txt
```

## Troubleshooting Dependency Conflicts

- **PyTorch & CUDA**: If you encounter issues with PyTorch not detecting your GPU, verify your CUDA version and install the matching PyTorch version from [pytorch.org](https://pytorch.org/).
  Example for CUDA 11.8:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- **OpenCV Errors**: If OpenCV complains about missing DLLs on Windows Server environments, install the media feature pack or switch to the headless version:
  ```bash
  pip install opencv-python-headless
  ```
