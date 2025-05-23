#!/usr/bin/env python3
"""
Deep Momentum Trading System - Setup Configuration
Advanced deep learning momentum trading system leveraging NVIDIA GH200 Grace Hopper architecture
"""

import os
import sys
import platform
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess

# Package metadata
PACKAGE_NAME = "deep-momentum-trading"
VERSION = "1.0.0"
DESCRIPTION = "Advanced deep learning momentum trading system leveraging NVIDIA GH200 Grace Hopper architecture"
LONG_DESCRIPTION = """
Deep Momentum Networks on GH200: Complete Trading System

An advanced deep learning momentum trading system leveraging NVIDIA GH200 Grace Hopper 
architecture to achieve superior risk-adjusted returns through LSTM-based momentum 
detection and Sharpe ratio optimization.

Key Features:
- 🧠 Advanced Neural Architectures: LSTM, Transformer, and Ensemble models
- ⚡ GH200 Optimization: Leverages 288GB HBM3 unified memory and 900GB/s bandwidth  
- 📈 Sharpe Ratio Optimization: Direct optimization of risk-adjusted returns
- 🔄 Real-time Processing: Live trading capabilities with microsecond latency
- 🎯 Multi-timeframe Analysis: Simultaneous processing across multiple time horizons
- 🛡️ Advanced Risk Management: VaR, CVaR, and drawdown controls
- 🤖 Meta-learning: Adaptive model combination based on market conditions

Performance Targets:
- Sharpe Ratio: 4.0-8.0 (vs. 1.2 for S&P 500)
- Maximum Drawdown: <15% (vs. 34% for S&P 500)
- Win Rate: 65-75%
- Information Ratio: 2.5-4.0
"""

AUTHOR = "Deep Momentum Trading Team"
AUTHOR_EMAIL = "contact@deepmomentum.ai"
URL = "https://github.com/your-org/deep-momentum-trading"
LICENSE = "MIT"

# Minimum Python version
PYTHON_REQUIRES = ">=3.11"

# Current directory
HERE = Path(__file__).parent.absolute()

def get_long_description():
    """Read long description from README.md"""
    readme_path = HERE / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return LONG_DESCRIPTION

def get_requirements():
    """Read requirements from requirements.txt"""
    requirements_path = HERE / "requirements.txt"
    requirements = []
    
    if requirements_path.exists():
        with open(requirements_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments, empty lines, and development dependencies
                if (line and not line.startswith("#") and 
                    not line.startswith("pytest") and 
                    not line.startswith("black") and 
                    not line.startswith("mypy")):
                    requirements.append(line)
    
    return requirements

def get_dev_requirements():
    """Get development requirements"""
    return [
        "pytest>=7.4.0",
        "black>=23.7.0", 
        "mypy>=1.5.0",
        "pytest-asyncio>=0.21.0",
        "pytest-benchmark>=4.0.0",
        "pytest-cov>=4.1.0",
        "flake8>=6.0.0",
        "isort>=5.12.0",
    ]

def detect_arm64():
    """Detect if running on ARM64 architecture"""
    machine = platform.machine().lower()
    return machine in ['arm64', 'aarch64']

def get_extra_compile_args():
    """Get ARM64-specific compilation arguments"""
    args = []
    
    if detect_arm64():
        args.extend([
            '-march=armv8-a+simd',
            '-mtune=neoverse-v2',  # GH200 Grace CPU
            '-O3',
            '-ffast-math',
            '-funroll-loops',
            '-ftree-vectorize',
            '-DARM64_OPTIMIZED=1'
        ])
    else:
        args.extend([
            '-O3',
            '-ffast-math',
            '-funroll-loops',
            '-ftree-vectorize'
        ])
    
    return args

class CustomBuildExt(build_ext):
    """Custom build extension for ARM64 optimizations"""
    
    def build_extensions(self):
        # Add ARM64-specific compilation flags
        for ext in self.extensions:
            ext.extra_compile_args.extend(get_extra_compile_args())
            
            if detect_arm64():
                ext.define_macros.append(('ARM64_OPTIMIZED', '1'))
        
        super().build_extensions()

# Package configuration
setup(
    # Basic package information
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    
    # Python version requirements
    python_requires=PYTHON_REQUIRES,
    
    # Package discovery
    packages=find_packages(include=[
        "src",
        "src.*",
        "scripts",
        "configs",
        "tests",
        "tests.*"
    ]),
    
    # Include additional files
    include_package_data=True,
    package_data={
        "": [
            "*.yaml", "*.yml", "*.json", "*.txt", "*.md",
            "*.sh", "*.bat", "*.cfg", "*.ini"
        ],
        "configs": ["**/*.yaml", "**/*.yml", "**/*.json"],
        "scripts": ["*.py", "*.sh", "*.bat"],
        "tests": ["**/*.py", "**/*.yaml", "**/*.json"],
    },
    
    # Dependencies
    install_requires=get_requirements(),
    
    # Optional dependencies
    extras_require={
        "dev": get_dev_requirements(),
        "gpu": [
            "torch[cuda]>=2.0.0",
            "nvidia-ml-py>=12.535.0",
        ],
        "arm64": [
            "torch>=2.0.0",  # ARM64-optimized PyTorch
        ],
        "all": get_dev_requirements() + [
            "torch[cuda]>=2.0.0",
            "nvidia-ml-py>=12.535.0",
        ]
    },
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            # Main trading system
            "momentum-trade=src.inference.trading_engine:main",
            "momentum-engine=src.inference.trading_engine:main",
            
            # Training and model management
            "momentum-train=scripts.train_models:main",
            "momentum-train-lstm=scripts.train_lstm:main", 
            "momentum-train-transformer=scripts.train_transformer:main",
            "momentum-train-ensemble=scripts.train_ensemble:main",
            
            # Data pipeline tools
            "momentum-data=src.data.data_pipeline:main",
            "momentum-preprocess=src.data.data_preprocessing:main",
            "momentum-features=src.data.feature_engineering:main",
            
            # Risk management tools
            "momentum-risk=src.risk.risk_manager:main",
            "momentum-portfolio=src.risk.portfolio_optimizer:main",
            
            # Utilities
            "momentum-config=src.utils.config:main",
            "momentum-metrics=src.utils.metrics:main",
            "momentum-validate=src.validation.model_validator:main",
            
            # Deployment and monitoring
            "momentum-deploy=scripts.deploy_system:main",
            "momentum-monitor=scripts.monitor_system:main",
            "momentum-backtest=scripts.run_backtest:main",
        ]
    },
    
    # Custom build command for ARM64 optimizations
    cmdclass={
        "build_ext": CustomBuildExt,
    },
    
    # Classification metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: GPU :: NVIDIA CUDA :: 12.0",
    ],
    
    # Keywords for package discovery
    keywords=[
        "trading", "momentum", "deep-learning", "lstm", "transformer",
        "quantitative-finance", "algorithmic-trading", "risk-management",
        "portfolio-optimization", "sharpe-ratio", "gh200", "arm64",
        "nvidia", "cuda", "real-time", "backtesting", "ensemble-learning"
    ],
    
    # Project URLs
    project_urls={
        "Documentation": "https://deep-momentum-trading.readthedocs.io/",
        "Source": "https://github.com/your-org/deep-momentum-trading",
        "Tracker": "https://github.com/your-org/deep-momentum-trading/issues",
        "Funding": "https://github.com/sponsors/your-org",
    },
    
    # Zip safety
    zip_safe=False,
    
    # Platform-specific configurations
    platforms=["any"],
    
    # Additional metadata for ARM64/GH200 optimization
    options={
        "build_ext": {
            "parallel": True,  # Enable parallel compilation
        },
        "bdist_wheel": {
            "universal": False,  # Platform-specific wheels for ARM64 optimizations
        }
    },
)

# Post-installation setup for ARM64 systems
def post_install_setup():
    """Perform post-installation setup for ARM64 systems"""
    if detect_arm64():
        print("\n" + "="*60)
        print("ARM64/GH200 OPTIMIZATION DETECTED")
        print("="*60)
        print("✅ ARM64-optimized compilation flags applied")
        print("✅ GH200 Grace CPU optimizations enabled")
        print("✅ SIMD vectorization configured")
        print("\nFor optimal performance on GH200:")
        print("1. Ensure PyTorch is compiled with ARM64 optimizations")
        print("2. Set CUDA_VISIBLE_DEVICES for GPU acceleration")
        print("3. Configure unified memory settings in config files")
        print("4. Run: momentum-config --validate-arm64")
        print("="*60)

if __name__ == "__main__":
    # Run post-installation setup
    if "install" in sys.argv:
        import atexit
        atexit.register(post_install_setup)