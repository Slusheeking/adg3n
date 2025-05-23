#!/bin/bash

# Enhanced Environment Setup Script with ARM64 Optimizations
#
# This script provides comprehensive environment setup for the Deep Momentum Trading System
# with ARM64-specific optimizations, dependency management, and system configuration.
#
# Features:
# - ARM64 architecture detection and optimization
# - Python environment setup with virtual environments
# - System dependency installation
# - Database initialization and configuration
# - Security configuration and API key management
# - Performance optimization and tuning
# - Development and production environment support

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SCRIPT_NAME="$(basename "$0")"
LOG_FILE="${PROJECT_ROOT}/logs/setup_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# System information
ARCH=$(uname -m)
OS=$(uname -s)
DISTRO=""
IS_ARM64=false
IS_MACOS=false
IS_LINUX=false

# Environment configuration
ENVIRONMENT="${TRADING_ENV:-development}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
VENV_NAME="${VENV_NAME:-deep_momentum_env}"
INSTALL_CUDA="${INSTALL_CUDA:-false}"
INSTALL_DOCKER="${INSTALL_DOCKER:-false}"
INSTALL_K8S="${INSTALL_K8S:-false}"

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Create logs directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Log to file
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    # Log to console with colors
    case "$level" in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} $message"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
        "DEBUG")
            echo -e "${BLUE}[DEBUG]${NC} $message"
            ;;
        *)
            echo -e "${CYAN}[$level]${NC} $message"
            ;;
    esac
}

# Error handling
error_exit() {
    log "ERROR" "$1"
    exit 1
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log "WARN" "Running as root. Some operations may behave differently."
    fi
}

# Detect system information
detect_system() {
    log "INFO" "Detecting system information..."
    
    # Architecture detection
    case "$ARCH" in
        "arm64"|"aarch64")
            IS_ARM64=true
            log "INFO" "ARM64 architecture detected"
            ;;
        "x86_64"|"amd64")
            IS_ARM64=false
            log "INFO" "x86_64 architecture detected"
            ;;
        *)
            log "WARN" "Unknown architecture: $ARCH"
            ;;
    esac
    
    # OS detection
    case "$OS" in
        "Darwin")
            IS_MACOS=true
            log "INFO" "macOS detected"
            ;;
        "Linux")
            IS_LINUX=true
            log "INFO" "Linux detected"
            
            # Detect Linux distribution
            if [[ -f /etc/os-release ]]; then
                . /etc/os-release
                DISTRO="$ID"
                log "INFO" "Linux distribution: $DISTRO"
            fi
            ;;
        *)
            error_exit "Unsupported operating system: $OS"
            ;;
    esac
    
    log "INFO" "System: $OS $ARCH, Environment: $ENVIRONMENT"
}

# Check system requirements
check_requirements() {
    log "INFO" "Checking system requirements..."
    
    local missing_deps=()
    
    # Check essential tools
    local essential_tools=("curl" "wget" "git" "make" "gcc")
    
    for tool in "${essential_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_deps+=("$tool")
        fi
    done
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        missing_deps+=("python3-pip")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log "WARN" "Missing dependencies: ${missing_deps[*]}"
        log "INFO" "Will attempt to install missing dependencies..."
    else
        log "INFO" "All essential requirements satisfied"
    fi
}

# Install system dependencies
install_system_dependencies() {
    log "INFO" "Installing system dependencies..."
    
    if [[ "$IS_MACOS" == true ]]; then
        install_macos_dependencies
    elif [[ "$IS_LINUX" == true ]]; then
        install_linux_dependencies
    fi
}

# Install macOS dependencies
install_macos_dependencies() {
    log "INFO" "Installing macOS dependencies..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        log "INFO" "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Update Homebrew
    brew update
    
    # Install essential packages
    local packages=(
        "python@${PYTHON_VERSION}"
        "git"
        "wget"
        "curl"
        "cmake"
        "pkg-config"
        "openssl"
        "readline"
        "sqlite3"
        "xz"
        "zlib"
        "libyaml"
        "hdf5"
        "postgresql"
        "redis"
    )
    
    # ARM64-specific packages
    if [[ "$IS_ARM64" == true ]]; then
        packages+=("llvm" "libomp")
    fi
    
    for package in "${packages[@]}"; do
        log "INFO" "Installing $package..."
        brew install "$package" || log "WARN" "Failed to install $package"
    done
    
    # Install Docker if requested
    if [[ "$INSTALL_DOCKER" == true ]]; then
        log "INFO" "Installing Docker Desktop..."
        brew install --cask docker
    fi
}

# Install Linux dependencies
install_linux_dependencies() {
    log "INFO" "Installing Linux dependencies..."
    
    case "$DISTRO" in
        "ubuntu"|"debian")
            install_debian_dependencies
            ;;
        "centos"|"rhel"|"fedora")
            install_redhat_dependencies
            ;;
        "arch")
            install_arch_dependencies
            ;;
        *)
            log "WARN" "Unsupported Linux distribution: $DISTRO"
            log "INFO" "Attempting generic installation..."
            install_generic_linux_dependencies
            ;;
    esac
}

# Install Debian/Ubuntu dependencies
install_debian_dependencies() {
    log "INFO" "Installing Debian/Ubuntu dependencies..."
    
    # Update package list
    sudo apt-get update
    
    # Install essential packages
    local packages=(
        "python${PYTHON_VERSION}"
        "python${PYTHON_VERSION}-dev"
        "python3-pip"
        "python3-venv"
        "build-essential"
        "git"
        "wget"
        "curl"
        "cmake"
        "pkg-config"
        "libssl-dev"
        "libffi-dev"
        "libhdf5-dev"
        "libpq-dev"
        "redis-server"
        "postgresql"
        "postgresql-contrib"
        "libyaml-dev"
        "zlib1g-dev"
        "libbz2-dev"
        "libreadline-dev"
        "libsqlite3-dev"
        "libncurses5-dev"
        "libncursesw5-dev"
        "xz-utils"
        "tk-dev"
        "libxml2-dev"
        "libxmlsec1-dev"
        "libffi-dev"
        "liblzma-dev"
    )
    
    # ARM64-specific packages
    if [[ "$IS_ARM64" == true ]]; then
        packages+=(
            "libomp-dev"
            "libopenblas-dev"
            "liblapack-dev"
            "gfortran"
        )
    fi
    
    # Install packages
    sudo apt-get install -y "${packages[@]}"
    
    # Install Docker if requested
    if [[ "$INSTALL_DOCKER" == true ]]; then
        install_docker_debian
    fi
    
    # Install Kubernetes tools if requested
    if [[ "$INSTALL_K8S" == true ]]; then
        install_k8s_tools_debian
    fi
}

# Install Red Hat/CentOS/Fedora dependencies
install_redhat_dependencies() {
    log "INFO" "Installing Red Hat/CentOS/Fedora dependencies..."
    
    local package_manager="yum"
    if command -v dnf &> /dev/null; then
        package_manager="dnf"
    fi
    
    # Install EPEL repository for CentOS/RHEL
    if [[ "$DISTRO" == "centos" || "$DISTRO" == "rhel" ]]; then
        sudo $package_manager install -y epel-release
    fi
    
    # Install essential packages
    local packages=(
        "python${PYTHON_VERSION//.}"
        "python3-devel"
        "python3-pip"
        "gcc"
        "gcc-c++"
        "make"
        "git"
        "wget"
        "curl"
        "cmake"
        "pkgconfig"
        "openssl-devel"
        "libffi-devel"
        "hdf5-devel"
        "postgresql-devel"
        "redis"
        "postgresql-server"
        "libyaml-devel"
        "zlib-devel"
        "bzip2-devel"
        "readline-devel"
        "sqlite-devel"
        "ncurses-devel"
        "xz-devel"
        "tk-devel"
        "libxml2-devel"
        "xmlsec1-devel"
    )
    
    # ARM64-specific packages
    if [[ "$IS_ARM64" == true ]]; then
        packages+=(
            "openblas-devel"
            "lapack-devel"
            "gcc-gfortran"
        )
    fi
    
    # Install packages
    sudo $package_manager install -y "${packages[@]}"
    
    # Initialize PostgreSQL
    if [[ "$DISTRO" == "centos" || "$DISTRO" == "rhel" ]]; then
        sudo postgresql-setup initdb
        sudo systemctl enable postgresql
        sudo systemctl start postgresql
    fi
}

# Install Arch Linux dependencies
install_arch_dependencies() {
    log "INFO" "Installing Arch Linux dependencies..."
    
    # Update package database
    sudo pacman -Sy
    
    # Install essential packages
    local packages=(
        "python"
        "python-pip"
        "base-devel"
        "git"
        "wget"
        "curl"
        "cmake"
        "pkgconf"
        "openssl"
        "libffi"
        "hdf5"
        "postgresql"
        "redis"
        "libyaml"
        "zlib"
        "bzip2"
        "readline"
        "sqlite"
        "ncurses"
        "xz"
        "tk"
        "libxml2"
        "xmlsec"
    )
    
    # ARM64-specific packages
    if [[ "$IS_ARM64" == true ]]; then
        packages+=(
            "openblas"
            "lapack"
            "gcc-fortran"
        )
    fi
    
    # Install packages
    sudo pacman -S --noconfirm "${packages[@]}"
}

# Generic Linux installation
install_generic_linux_dependencies() {
    log "WARN" "Using generic Linux installation - some features may not work"
    
    # Try to install Python and pip
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-venv build-essential
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3 python3-pip gcc gcc-c++ make
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm python python-pip base-devel
    else
        error_exit "No supported package manager found"
    fi
}

# Install Docker on Debian/Ubuntu
install_docker_debian() {
    log "INFO" "Installing Docker on Debian/Ubuntu..."
    
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Add Docker repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Add user to docker group
    sudo usermod -aG docker "$USER"
    
    log "INFO" "Docker installed. Please log out and back in to use Docker without sudo."
}

# Install Kubernetes tools on Debian/Ubuntu
install_k8s_tools_debian() {
    log "INFO" "Installing Kubernetes tools on Debian/Ubuntu..."
    
    # Install kubectl
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/$(dpkg --print-architecture)/kubectl"
    sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
    rm kubectl
    
    # Install helm
    curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
    sudo apt-get update
    sudo apt-get install -y helm
}

# Setup Python environment
setup_python_environment() {
    log "INFO" "Setting up Python environment..."
    
    # Check Python version
    local python_cmd="python3"
    if command -v "python${PYTHON_VERSION}" &> /dev/null; then
        python_cmd="python${PYTHON_VERSION}"
    fi
    
    local python_version=$($python_cmd --version 2>&1 | cut -d' ' -f2)
    log "INFO" "Using Python version: $python_version"
    
    # Create virtual environment
    log "INFO" "Creating virtual environment: $VENV_NAME"
    $python_cmd -m venv "$PROJECT_ROOT/$VENV_NAME"
    
    # Activate virtual environment
    source "$PROJECT_ROOT/$VENV_NAME/bin/activate"
    
    # Upgrade pip
    log "INFO" "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    # Install Python dependencies
    install_python_dependencies
}

# Install Python dependencies
install_python_dependencies() {
    log "INFO" "Installing Python dependencies..."
    
    # Check if requirements.txt exists
    local requirements_file="$PROJECT_ROOT/requirements.txt"
    if [[ ! -f "$requirements_file" ]]; then
        log "WARN" "requirements.txt not found, creating basic requirements..."
        create_requirements_file
    fi
    
    # Install requirements
    log "INFO" "Installing from requirements.txt..."
    pip install -r "$requirements_file"
    
    # ARM64-specific optimizations
    if [[ "$IS_ARM64" == true ]]; then
        install_arm64_optimizations
    fi
    
    # Install development dependencies if in development environment
    if [[ "$ENVIRONMENT" == "development" ]]; then
        install_development_dependencies
    fi
}

# Create basic requirements file
create_requirements_file() {
    log "INFO" "Creating basic requirements.txt..."
    
    cat > "$PROJECT_ROOT/requirements.txt" << EOF
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0

# Deep learning
torch>=1.12.0
torchvision>=0.13.0
transformers>=4.20.0

# Data processing
h5py>=3.7.0
tables>=3.7.0
pyarrow>=8.0.0
fastparquet>=0.8.0

# Financial data
yfinance>=0.1.70
alpha-vantage>=2.3.1

# Database
sqlalchemy>=1.4.0
psycopg2-binary>=2.9.0
redis>=4.3.0

# API and networking
requests>=2.28.0
aiohttp>=3.8.0
websockets>=10.3
zmq>=0.0.0

# Configuration and utilities
pyyaml>=6.0
python-dotenv>=0.20.0
click>=8.1.0
tqdm>=4.64.0

# Monitoring and logging
prometheus-client>=0.14.0
structlog>=22.1.0

# Testing (development)
pytest>=7.1.0
pytest-asyncio>=0.19.0
pytest-cov>=3.0.0

# Code quality (development)
black>=22.6.0
flake8>=5.0.0
mypy>=0.971
pre-commit>=2.20.0
EOF
}

# Install ARM64 optimizations
install_arm64_optimizations() {
    log "INFO" "Installing ARM64-specific optimizations..."
    
    # Install optimized BLAS libraries
    if [[ "$IS_MACOS" == true ]]; then
        # Use Accelerate framework on macOS
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
    elif [[ "$IS_LINUX" == true ]]; then
        # Install OpenBLAS optimizations
        pip install --no-cache-dir numpy --config-settings=setup-args="-Dblas=openblas"
    fi
    
    # Install ARM64-optimized PyTorch
    if [[ "$IS_ARM64" == true ]]; then
        log "INFO" "Installing ARM64-optimized PyTorch..."
        if [[ "$IS_MACOS" == true ]]; then
            # macOS ARM64 PyTorch with Metal Performance Shaders
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        else
            # Linux ARM64 PyTorch
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    fi
}

# Install development dependencies
install_development_dependencies() {
    log "INFO" "Installing development dependencies..."
    
    local dev_packages=(
        "jupyter"
        "jupyterlab"
        "ipython"
        "matplotlib"
        "seaborn"
        "plotly"
        "dash"
        "streamlit"
        "notebook"
    )
    
    pip install "${dev_packages[@]}"
}

# Setup databases
setup_databases() {
    log "INFO" "Setting up databases..."
    
    # Setup PostgreSQL
    setup_postgresql
    
    # Setup Redis
    setup_redis
    
    # Create database directories
    mkdir -p "$PROJECT_ROOT/data/storage"
    mkdir -p "$PROJECT_ROOT/data/cache"
    mkdir -p "$PROJECT_ROOT/data/logs"
}

# Setup PostgreSQL
setup_postgresql() {
    log "INFO" "Setting up PostgreSQL..."
    
    # Start PostgreSQL service
    if [[ "$IS_MACOS" == true ]]; then
        brew services start postgresql
    elif [[ "$IS_LINUX" == true ]]; then
        sudo systemctl enable postgresql
        sudo systemctl start postgresql
    fi
    
    # Create database and user
    local db_name="deep_momentum_trading"
    local db_user="trading_user"
    local db_password="trading_password_$(date +%s)"
    
    # Create user and database
    if command -v psql &> /dev/null; then
        sudo -u postgres psql -c "CREATE USER $db_user WITH PASSWORD '$db_password';" || true
        sudo -u postgres psql -c "CREATE DATABASE $db_name OWNER $db_user;" || true
        sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $db_name TO $db_user;" || true
        
        log "INFO" "PostgreSQL database created: $db_name"
        log "INFO" "Database credentials saved to .env file"
        
        # Save credentials to .env file
        echo "DATABASE_URL=postgresql://$db_user:$db_password@localhost:5432/$db_name" >> "$PROJECT_ROOT/.env"
    else
        log "WARN" "PostgreSQL not available, skipping database setup"
    fi
}

# Setup Redis
setup_redis() {
    log "INFO" "Setting up Redis..."
    
    # Start Redis service
    if [[ "$IS_MACOS" == true ]]; then
        brew services start redis
    elif [[ "$IS_LINUX" == true ]]; then
        sudo systemctl enable redis
        sudo systemctl start redis
    fi
    
    # Test Redis connection
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping &> /dev/null; then
            log "INFO" "Redis is running and accessible"
            echo "REDIS_URL=redis://localhost:6379/0" >> "$PROJECT_ROOT/.env"
        else
            log "WARN" "Redis is not responding"
        fi
    else
        log "WARN" "Redis CLI not available"
    fi
}

# Setup environment files
setup_environment_files() {
    log "INFO" "Setting up environment files..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
        log "INFO" "Creating .env file..."
        cat > "$PROJECT_ROOT/.env" << EOF
# Deep Momentum Trading System Environment Configuration
# Generated on $(date)

# Environment
TRADING_ENV=$ENVIRONMENT
PYTHONPATH=$PROJECT_ROOT

# API Keys (replace with actual values)
POLYGON_API_KEY=your_polygon_api_key_here
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key_here
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here

# Alpaca Configuration
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets

# Logging
LOG_LEVEL=INFO
LOG_FILE=$PROJECT_ROOT/logs/trading.log

# Performance
MAX_WORKERS=4
BATCH_SIZE=100

# Security
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET_KEY=$(openssl rand -hex 32)

# ARM64 Optimizations
ARM64_OPTIMIZED=$IS_ARM64
USE_ACCELERATE=$IS_MACOS
EOF
    else
        log "INFO" ".env file already exists, skipping creation"
    fi
    
    # Create .env.example file
    cp "$PROJECT_ROOT/.env" "$PROJECT_ROOT/.env.example"
    
    # Replace sensitive values in example file
    sed -i.bak 's/=your_.*_here/=your_api_key_here/g' "$PROJECT_ROOT/.env.example"
    sed -i.bak 's/SECRET_KEY=.*/SECRET_KEY=your_secret_key_here/g' "$PROJECT_ROOT/.env.example"
    sed -i.bak 's/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=your_jwt_secret_key_here/g' "$PROJECT_ROOT/.env.example"
    rm -f "$PROJECT_ROOT/.env.example.bak"
}

# Setup logging
setup_logging() {
    log "INFO" "Setting up logging..."
    
    # Create log directories
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/logs/trading"
    mkdir -p "$PROJECT_ROOT/logs/models"
    mkdir -p "$PROJECT_ROOT/logs/data"
    mkdir -p "$PROJECT_ROOT/logs/system"
    
    # Create log configuration
    cat > "$PROJECT_ROOT/config/logging.yaml" << EOF
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    datefmt: '%Y-%m-%d %H:%M:%S'
  
  detailed:
    format: '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
    datefmt: '%Y-%m-%d %H:%M:%S'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/trading.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
  
  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: logs/errors.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  deep_momentum_trading:
    level: DEBUG
    handlers: [console, file, error_file]
    propagate: false

root:
  level: INFO
  handlers: [console, file]
EOF
}

# Setup system services
setup_system_services() {
    log "INFO" "Setting up system services..."
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        setup_production_services
    else
        log "INFO" "Development environment - skipping system services setup"
    fi
}

# Setup production services
setup_production_services() {
    log "INFO" "Setting up production services..."
    
    # Create systemd service files for Linux
    if [[ "$IS_LINUX" == true ]] && command -v systemctl &> /dev/null; then
        create_systemd_services
    fi
    
    # Setup log rotation
    setup_log_rotation
    
    # Setup monitoring
    setup_monitoring
}

# Create systemd services
create_systemd_services() {
    log "INFO" "Creating systemd services..."
    
    # Trading service
    sudo tee /etc/systemd/system/deep-momentum-trading.service > /dev/null << EOF
[Unit]
Description=Deep Momentum Trading System
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_ROOT
Environment=PATH=$PROJECT_ROOT/$VENV_NAME/bin
ExecStart=$PROJECT_ROOT/$VENV_NAME/bin/python -m scripts.start_trading
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Data feed service
    sudo tee /etc/systemd/system/deep-momentum-data-feed.service > /dev/null << EOF
[Unit]
Description=Deep Momentum Data Feed Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_ROOT
Environment=PATH=$PROJECT_ROOT/$VENV_NAME/bin
ExecStart=$PROJECT_ROOT/$VENV_NAME/bin/python -m scripts.start_data_feed
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Risk monitor service
    sudo tee /etc/systemd/system/deep-momentum-risk-monitor.service > /dev/null << EOF
[Unit]
Description=Deep Momentum Risk Monitor Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_ROOT
Environment=PATH=$PROJECT_ROOT/$VENV_NAME/bin
ExecStart=$PROJECT_ROOT/$VENV_NAME/bin/python -m scripts.start_risk_monitor
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    log "INFO" "Systemd services created. Enable with: sudo systemctl enable <service-name>"
}

# Setup log rotation
setup_log_rotation() {
    log "INFO" "Setting up log rotation..."
    
    if [[ "$IS_LINUX" == true ]]; then
        sudo tee /etc/logrotate.d/deep-momentum-trading > /dev/null << EOF
$PROJECT_ROOT/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        systemctl reload deep-momentum-trading || true
    endscript
}
EOF
    fi
}

# Setup monitoring
setup_monitoring() {
    log "INFO" "Setting up monitoring..."
    
    # Create monitoring directories
    mkdir -p "$PROJECT_ROOT/monitoring/prometheus"
    mkdir -p "$PROJECT_ROOT/monitoring/grafana"
    
    # Create basic Prometheus configuration
    cat > "$PROJECT_ROOT/monitoring/prometheus/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'deep-momentum-trading'
    static_configs:
      - targets: ['localhost:8080']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'system'
    static_configs:
      - targets: ['localhost:9100']
    scrape_interval: 15s
EOF
}

# Optimize system performance
optimize_system_performance() {
    log "INFO" "Optimizing system performance..."
    
    # ARM64-specific optimizations
    if [[ "$IS_ARM64" == true ]]; then
        optimize_arm64_performance
    fi
    
    # General optimizations
    optimize_general_performance
}

# ARM64 performance optimizations
optimize_arm64_performance() {
    log "INFO" "Applying ARM64 performance optimizations..."
    
    # Set environment variables for ARM64 optimization
    cat >> "$PROJECT_ROOT/.env" << EOF

# ARM64 Performance Optimizations
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_MAX_THREADS=4
OMP_NUM_THREADS=4
VECLIB_MAXIMUM_THREADS=4

# ARM64 SIMD optimizations
USE_NEON=1
ARM_COMPUTE_LIBRARY=1
EOF
    
    # macOS-specific optimizations
    if [[ "$IS_MACOS" == true ]]; then
        cat >> "$PROJECT_ROOT/.env" << EOF

# macOS ARM64 optimizations
PYTORCH_ENABLE_MPS_FALLBACK=1
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
EOF
    fi
}

# General performance optimizations
optimize_general_performance() {
    log "INFO" "Applying general performance optimizations..."
    
    # Python optimizations
    cat >> "$PROJECT_ROOT/.env" << EOF

# Python performance optimizations
PYTHONOPTIMIZE=1
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1

# Memory optimizations
MALLOC_ARENA_MAX=2
MALLOC_MMAP_THRESHOLD_=131072
MALLOC_TRIM_THRESHOLD_=131072
MALLOC_TOP_PAD_=131072
MALLOC_MMAP_MAX_=65536
EOF
}

# Run tests
run_tests() {
    log "INFO" "Running system tests..."
    
    # Activate virtual environment
    source "$PROJECT_ROOT/$VENV_NAME/bin/activate"
    
    # Test Python imports
    log "INFO" "Testing Python imports..."
    python3 -c "
import sys
import numpy as np
import pandas as pd
import torch
import yaml
import redis
import psycopg2
print('✅ All core imports successful')
print(f'Python: {sys.version}')
print(f'NumPy: {np.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'PyTorch: {torch.__version__}')
"
    
    # Test database connections
    test_database_connections
    
    # Test ARM64 optimizations
    if [[ "$IS_ARM64" == true ]]; then
        test_arm64_optimizations
    fi
}

# Test database connections
test_database_connections() {
    log "INFO" "Testing database connections..."
    
    # Test PostgreSQL
    if command -v psql &> /dev/null; then
        if psql -h localhost -U postgres -c "SELECT version();" &> /dev/null; then
            log "INFO" "✅ PostgreSQL connection successful"
        else
            log "WARN" "❌ PostgreSQL connection failed"
        fi
    fi
    
    # Test Redis
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping &> /dev/null; then
            log "INFO" "✅ Redis connection successful"
        else
            log "WARN" "❌ Redis connection failed"
        fi
    fi
}

# Test ARM64 optimizations
test_arm64_optimizations() {
    log "INFO" "Testing ARM64 optimizations..."
    
    python3 -c "
import platform
import numpy as np
import torch

print(f'Architecture: {platform.machine()}')
print(f'Platform: {platform.platform()}')

# Test NumPy BLAS
print(f'NumPy BLAS info: {np.show_config()}')

# Test PyTorch
print(f'PyTorch version: {torch.__version__}')
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✅ MPS (Metal Performance Shaders) available')
else:
    print('ℹ️  MPS not available')

print('✅ ARM64 optimization tests completed')
"
}

# Generate setup report
generate_setup_report() {
    log "INFO" "Generating setup report..."
    
    local report_file="$PROJECT_ROOT/setup_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
Deep Momentum Trading System - Setup Report
==========================================
Generated: $(date)

System Information:
------------------
OS: $OS
Architecture: $ARCH
Distribution: $DISTRO
ARM64: $IS_ARM64
Environment: $ENVIRONMENT

Python Environment:
------------------
Virtual Environment: $VENV_NAME
Python Version: $(python3 --version 2>&1)
Pip Version: $(pip --version 2>&1)

Installed Services:
------------------
PostgreSQL: $(command -v psql &> /dev/null && echo "✅ Installed" || echo "❌ Not installed")
Redis: $(command -v redis-cli &> /dev/null && echo "✅ Installed" || echo "❌ Not installed")
Docker: $(command -v docker &> /dev/null && echo "✅ Installed" || echo "❌ Not installed")
Kubectl: $(command -v kubectl &> /dev/null && echo "✅ Installed" || echo "❌ Not installed")

Configuration Files:
-------------------
.env: $(test -f "$PROJECT_ROOT/.env" && echo "✅ Created" || echo "❌ Missing")
.env.example: $(test -f "$PROJECT_ROOT/.env.example" && echo "✅ Created" || echo "❌ Missing")
logging.yaml: $(test -f "$PROJECT_ROOT/config/logging.yaml" && echo "✅ Created" || echo "❌ Missing")

Directories Created:
-------------------
logs/: $(test -d "$PROJECT_ROOT/logs" && echo "✅ Created" || echo "❌ Missing")
data/storage/: $(test -d "$PROJECT_ROOT/data/storage" && echo "✅ Created" || echo "❌ Missing")
monitoring/: $(test -d "$PROJECT_ROOT/monitoring" && echo "✅ Created" || echo "❌ Missing")

Next Steps:
----------
1. Update API keys in .env file
2. Activate virtual environment: source $VENV_NAME/bin/activate
3. Run tests: python -m pytest tests/
4. Start services: python -m scripts.start_trading

For production deployment:
1. Enable systemd services: sudo systemctl enable deep-momentum-*
2. Configure firewall and security settings
3. Set up SSL certificates
4. Configure monitoring and alerting

Setup Log: $LOG_FILE
EOF
    
    log "INFO" "Setup report generated: $report_file"
    
    # Display summary
    echo -e "\n${GREEN}=== Setup Complete ===${NC}"
    echo -e "Report: ${CYAN}$report_file${NC}"
    echo -e "Log: ${CYAN}$LOG_FILE${NC}"
    echo -e "\n${YELLOW}Next steps:${NC}"
    echo -e "1. Update API keys in ${CYAN}.env${NC} file"
    echo -e "2. Activate environment: ${CYAN}source $VENV_NAME/bin/activate${NC}"
    echo -e "3. Run tests: ${CYAN}python -m pytest tests/${NC}"
    echo -e "4. Start trading: ${CYAN}python -m scripts.start_trading${NC}"
}

# Main setup function
main() {
    echo -e "${PURPLE}"
    echo "========================================"
    echo "Deep Momentum Trading System Setup"
    echo "ARM64 Optimized Environment Setup"
    echo "========================================"
    echo -e "${NC}"
    
    # Check if running as root
    check_root
    
    # Detect system
    detect_system
    
    # Check requirements
    check_requirements
    
    # Install system dependencies
    install_system_dependencies
    
    # Setup Python environment
    setup_python_environment
    
    # Setup databases
    setup_databases
    
    # Setup environment files
    setup_environment_files
    
    # Setup logging
    setup_logging
    
    # Setup system services
    setup_system_services
    
    # Optimize performance
    optimize_system_performance
    
    # Run tests
    run_tests
    
    # Generate report
    generate_setup_report
    
    log "INFO" "Setup completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment|-e)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --python-version|-p)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --venv-name|-v)
            VENV_NAME="$2"
            shift 2
            ;;
        --install-docker)
            INSTALL_DOCKER=true
            shift
            ;;
        --install-k8s)
            INSTALL_K8S=true
            shift
            ;;
        --install-cuda)
            INSTALL_CUDA=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -e, --environment ENV     Set environment (development|staging|production)"
            echo "  -p, --python-version VER  Set Python version (default: 3.9)"
            echo "  -v, --venv-name NAME      Set virtual environment name"
            echo "      --install-docker      Install Docker"
            echo "      --install-k8s         Install Kubernetes tools"
            echo "      --install-cuda        Install CUDA support"
            echo "  -h, --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"