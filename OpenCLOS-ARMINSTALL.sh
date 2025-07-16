#!/bin/bash

# OpenCLOS Auto-start Installation for Armbian/Orange Pi 3
# THIS IS INTRUSIVE AND WILL INSTALL ALL THE NEEDED DEPENDENCIES! You have been warned!

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
section() {
    echo -e "\n${BLUE}===[ $1 ]===${NC}"
    if [ -n "$2" ]; then
        echo -e "${YELLOW}$2${NC}"
    fi
}

# Display purple ASCII art
echo -e "${PURPLE}"
cat << "EOF"
 _____ ______       ________      ________      ________      ________      _____ ______       ___           ________      ________     
|\   _ \  _   \    |\   __  \    |\   __  \    |\   ____\    |\   __  \    |\   _ \  _   \    |\  \         |\   ___ \    |\   ___ \    
\ \  \\\__\ \  \   \ \  \|\  \   \ \  \|\  \   \ \  \___|    \ \  \|\  \   \ \  \\\__\ \  \   \ \  \        \ \  \_|\ \   \ \  \_|\ \   
 \ \  \\|__| \  \   \ \   __  \   \ \   _  _\   \ \  \        \ \  \\\  \   \ \  \\|__| \  \   \ \  \        \ \  \ \\ \   \ \  \ \\ \  
  \ \  \    \ \  \   \ \  \ \  \   \ \  \\  \|   \ \  \____    \ \  \\\  \   \ \  \    \ \  \   \ \  \____    \ \  \_\\ \   \ \  \_\\ \ 
   \ \__\    \ \__\   \ \__\ \__\   \ \__\\ _\    \ \_______\   \ \_______\   \ \__\    \ \__\   \ \_______\   \ \_______\   \ \_______\
    \|__|     \|__|    \|__|\|__|    \|__|\|__|    \|_______|    \|_______|    \|__|     \|__|    \|_______|    \|_______|    \|_______|
EOF
echo -e "${NC}"

# Display comprehensive warning and confirmation
section "IMPORTANT WARNING" "THIS SCRIPT WILL MAKE SIGNIFICANT CHANGES TO YOUR SYSTEM"
echo -e "${RED}BY CONTINUING, YOU ACKNOWLEDGE THAT THIS SCRIPT WILL:${NC}"
echo -e "1. ${YELLOW}Download and install packages from the internet${NC} (requires active internet connection)"
echo -e "2. ${YELLOW}Modify system configurations${NC} (including creating a 2GB swap file)"
echo -e "3. ${YELLOW}Install system dependencies${NC} (Python, build tools, media libraries)"
echo -e "4. ${YELLOW}Compile dlib from source${NC} (optimized for ARM, may take 15-30 minutes)"
echo -e "5. ${YELLOW}Install Python packages${NC} (including OpenCV, numpy, dlib, pygame)"
echo -e "6. ${YELLOW}Download facial recognition models${NC} (68MB shape predictor)"
echo -e "7. ${YELLOW}Install OpenCLOS in /opt/openclos${NC}"
echo -e "8. ${YELLOW}Create a systemd service${NC} (to auto-start the application)"
echo -e "\n${RED}REQUIREMENTS:${NC}"
echo -e "- Active internet connection"
echo -e "- At least 3GB of free disk space"
echo -e "- 30-60 minutes of time (depending on hardware)"
echo -e "\n${RED}THIS PROCESS CANNOT BE EASILY UNDONE!${NC}\n"

read -p "Do you understand and accept these changes? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Installation aborted by user${NC}"
    exit 1
fi

# ARM-specific variables
ARM_ARCH=$(uname -m)
CPU_CORES=$(nproc)
SERVICE_USER=$(whoami)
INSTALL_DIR="/opt/openclos"
SERVICE_FILE="/etc/systemd/system/openclos.service"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to setup swap
setup_swap() {
    section "Configuring Swap Space" "Creating 2GB swap file to improve performance on memory-constrained ARM devices"
    if [ ! -f /swapfile ]; then
        sudo fallocate -l 2G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        echo -e "${GREEN}2GB swap file created${NC}"
    else
        echo -e "${YELLOW}Swap file already exists${NC}"
    fi
}

# Function to install dependencies
install_dependencies() {
    section "Installing Required Packages" "Installing system packages including Python, build tools, and media libraries"
    sudo apt-get update
    sudo apt-get install -y \
        python3 python3-dev python3-pip python3-venv \
        build-essential cmake git wget unzip \
        libjpeg-dev libpng-dev libatlas-base-dev \
        libopenblas-dev libgtk-3-dev libcanberra-gtk-module \
        libv4l-dev libxvidcore-dev libx264-dev \
        ffmpeg libsm6 libxext6 \
        python3-tk python3-dev-tk
}

# Function to build optimized dlib
build_dlib() {
    section "Building ARM-Optimized dlib" "Compiling dlib from source with ARM-specific optimizations (this may take 15-30 minutes)"
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"
    
    git clone https://github.com/davisking/dlib.git
    cd dlib
    mkdir build
    cd build
    
    cmake .. -DDLIB_USE_CUDA=0 -DUSE_AVX_INSTRUCTIONS=0 -DUSE_SSE4_INSTRUCTIONS=0
    cmake --build . --config Release -- -j$CPU_CORES
    sudo make install
    sudo ldconfig
    
    cd ~
    rm -rf "$temp_dir"
}

# Function to install Python requirements
install_python_deps() {
    section "Installing Python Dependencies" "Installing optimized Python packages including NumPy with OpenBLAS support"
    pip3 install --upgrade pip setuptools wheel
    
    # Install numpy with OpenBLAS support
    OPENBLAS="$(ls /usr/lib/arm-linux-gnueabihf/libopenblas.so.* | head -n1)"
    [ -n "$OPENBLAS" ] && {
        echo "[openblas]
libraries = openblas
library_dirs = /usr/lib/arm-linux-gnueabihf
include_dirs = /usr/include/openblas" > ~/.numpy-site.cfg
        pip3 install numpy==1.21.6 --no-cache-dir --force-reinstall --no-binary numpy
    }
    
    pip3 install \
        opencv-python-headless==4.5.5.64 \
        dlib==19.24.0 \
        scipy==1.7.3 \
        psutil==5.9.0 \
        pygame==2.1.2 \
        Pillow==9.2.0 \
        tk==0.1.0
}

# Function to download dlib model
download_dlib_model() {
    section "Downloading Face Landmark Model" "Downloading pre-trained facial landmark detection model (68MB)"
    local model_file="$INSTALL_DIR/shape_predictor_68_face_landmarks.dat"
    
    if [ ! -f "$model_file" ]; then
        wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O model.bz2
        bunzip2 model.bz2
        sudo mv shape_predictor_68_face_landmarks.dat "$model_file"
        rm -f model.bz2
        echo -e "${GREEN}Model downloaded to $model_file${NC}"
    else
        echo -e "${YELLOW}Model already exists${NC}"
    fi
}

# Function to install OpenCLOS
install_openclos() {
    section "Installing OpenCLOS" "Cloning OpenCLOS repository and setting up Python virtual environment"
    sudo mkdir -p "$INSTALL_DIR"
    sudo chown -R $SERVICE_USER:$SERVICE_USER "$INSTALL_DIR"
    
    # Clone repository
    git clone --depth 1 --branch "OpenCLOSv.1.1" \
        https://github.com/MarcoMLDD/OpenCLOS.git "$INSTALL_DIR"
    
    # Apply ARM optimizations
    sed -i "s/self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)/self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)/" \
        "$INSTALL_DIR/OpenCLOS_v1.1.py"
    sed -i "s/self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)/self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)/" \
        "$INSTALL_DIR/OpenCLOS_v1.1.py"
    
    # Create virtual environment
    python3 -m venv "$INSTALL_DIR/venv"
    source "$INSTALL_DIR/venv/bin/activate"
    pip install -r "$INSTALL_DIR/requirements.txt"
    
    # Download alert sound
    wget -q https://assets.mixkit.co/sfx/preview/mixkit-alarm-digital-clock-beep-989.mp3 \
        -O "$INSTALL_DIR/alert.wav"
}

# Function to create systemd service
create_systemd_service() {
    section "Creating Systemd Service" "Setting up OpenCLOS to run automatically at startup"
    sudo bash -c "cat > $SERVICE_FILE" <<EOL
[Unit]
Description=OpenCLOS Drowsiness Detection
After=graphical.target

[Service]
User=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python $INSTALL_DIR/OpenCLOS_v1.1.py --arm
Restart=always
RestartSec=30
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/$SERVICE_USER/.Xauthority

[Install]
WantedBy=multi-user.target
EOL

    sudo systemctl daemon-reload
    sudo systemctl enable openclos.service
    echo -e "${GREEN}Systemd service created and enabled${NC}"
}

# Main installation process
section "Starting OpenCLOS Auto-start Installation" "This process may take 30-60 minutes depending on your hardware"
setup_swap
install_dependencies
build_dlib
install_python_deps
download_dlib_model
install_openclos
create_systemd_service

# Final instructions
section "Installation Complete" "OpenCLOS is now configured to start automatically"
echo -e "${GREEN}OpenCLOS will now start automatically on boot!${NC}"
echo -e "\nService commands:"
echo -e "  Start now: ${YELLOW}sudo systemctl start openclos${NC}"
echo -e "  Check status: ${YELLOW}sudo systemctl status openclos${NC}"
echo -e "  View logs: ${YELLOW}journalctl -u openclos -f${NC}"
echo -e "\nConfiguration:"
echo -e "  Edit service file: ${YELLOW}sudo nano $SERVICE_FILE${NC}"
echo -e "  After editing: ${YELLOW}sudo systemctl daemon-reload${NC}"
echo -e "\nRunning with ARM optimizations (320x240 resolution)"
