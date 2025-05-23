# memory-bank/productContext.md

## Project Overview
Deep Momentum Trading System - A comprehensive algorithmic trading platform that leverages deep learning models to identify and capitalize on momentum patterns in financial markets.

## Core Architecture

### Data Layer
- **Polygon Client**: Real-time market data ingestion
- **Feature Engineering**: Technical indicators and momentum features
- **Data Preprocessing**: Normalization and cleaning pipelines
- **Memory Cache**: High-performance data caching
- **Market Universe**: Asset selection and filtering

### Storage Layer
- **HDF5 Storage**: High-performance numerical data storage
- **Parquet Storage**: Columnar data format for analytics
- **SQLite Storage**: Lightweight relational database
- **Memory Storage**: In-memory data structures for real-time processing

### Model Layer
- **Deep Momentum LSTM**: Primary momentum detection model
- **Transformer Momentum**: Advanced attention-based model
- **Ensemble System**: Model combination and voting
- **Meta Learner**: Model selection and adaptation
- **ARM64 Optimizations**: Platform-specific performance enhancements

### Trading Layer
- **Alpaca Client**: Brokerage integration
- **Execution Engine**: Order execution and management
- **Order Manager**: Order lifecycle management
- **Position Manager**: Portfolio position tracking
- **Trading Engine**: Core trading logic coordination

### Risk Management
- **Risk Manager**: Overall risk oversight
- **Portfolio Optimizer**: Position sizing and allocation
- **VAR Calculator**: Value at Risk calculations
- **Correlation Monitor**: Asset correlation tracking
- **Liquidity Monitor**: Market liquidity assessment
- **Stress Testing**: Scenario analysis

### Infrastructure
- **Communication**: ZMQ-based messaging system
- **Monitoring**: Performance and system health tracking
- **Training**: Distributed model training pipeline
- **Scheduling**: Task automation and coordination

## Key Technologies
- **Python 3.8+**: Primary development language
- **PyTorch**: Deep learning framework
- **NumPy/Pandas**: Data manipulation and analysis
- **ZeroMQ**: High-performance messaging
- **HDF5**: Scientific data storage
- **Alpaca API**: Trading execution
- **Polygon API**: Market data feed

## Technical Standards
- **Code Quality**: Type hints, docstrings, comprehensive testing
- **Testing**: Unit tests, integration tests, end-to-end testing
- **Configuration**: YAML-based configuration management
- **Logging**: Structured logging with multiple output formats
- **Documentation**: Comprehensive API and architecture documentation

## Key Dependencies
- torch>=1.9.0
- numpy>=1.21.0
- pandas>=1.3.0
- pyzmq>=22.0.0
- h5py>=3.1.0
- alpaca-trade-api>=2.0.0
- polygon-api-client>=1.0.0

## Development Workflow
- **Environment**: Python virtual environment with requirements.txt
- **Testing**: pytest with fixtures and comprehensive test coverage
- **Configuration**: Environment-specific YAML configurations
- **Deployment**: Script-based deployment with health monitoring

## Project Goals
1. **Performance**: Low-latency trade execution and real-time data processing
2. **Reliability**: Robust error handling and system resilience
3. **Scalability**: Distributed processing and horizontal scaling
4. **Risk Management**: Comprehensive risk controls and monitoring
5. **Maintainability**: Clean code architecture and comprehensive testing