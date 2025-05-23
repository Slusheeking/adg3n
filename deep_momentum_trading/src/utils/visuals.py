"""
Training Visualization Utilities

This module provides comprehensive visualization tools for training metrics,
model performance, and analysis charts for the deep momentum trading system.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings
from dataclasses import asdict

from deep_momentum_trading.src.utils.logger import get_logger

logger = get_logger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingVisualizer:
    """
    Comprehensive visualization tool for training metrics and model performance.
    
    Creates interactive and static charts for:
    - Training/validation loss curves
    - Learning rate schedules
    - Performance metrics (accuracy, Sharpe ratio, etc.)
    - System metrics (latency, memory usage, GPU utilization)
    - Model predictions vs actual returns
    - Portfolio performance analysis
    """
    
    def __init__(self, 
                 output_dir: str = "training_visuals",
                 save_format: str = "both",  # "matplotlib", "plotly", or "both"
                 interactive: bool = True,
                 theme: str = "plotly_white"):
        """
        Initialize the TrainingVisualizer.
        
        Args:
            output_dir: Directory to save visualization files
            save_format: Format to save charts ("matplotlib", "plotly", or "both")
            interactive: Whether to create interactive plotly charts
            theme: Plotly theme to use
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_format = save_format
        self.interactive = interactive
        self.theme = theme
        
        # Create subdirectories
        (self.output_dir / "static").mkdir(exist_ok=True)
        (self.output_dir / "interactive").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        logger.info(f"TrainingVisualizer initialized. Output directory: {self.output_dir}")
    
    def create_training_dashboard(self, 
                                training_history: Dict[str, List[float]],
                                model_name: str = "model",
                                training_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Create a comprehensive training dashboard with all key metrics.
        
        Args:
            training_history: Dictionary containing training metrics over epochs
            model_name: Name of the model being trained
            training_config: Training configuration parameters
        """
        logger.info(f"Creating training dashboard for {model_name}")
        
        try:
            # Create loss curves
            self.plot_loss_curves(training_history, model_name)
            
            # Create learning rate schedule
            if 'learning_rate' in training_history:
                self.plot_learning_rate_schedule(training_history, model_name)
            
            # Create performance metrics
            self.plot_performance_metrics(training_history, model_name)
            
            # Create system metrics
            self.plot_system_metrics(training_history, model_name)
            
            # Create convergence analysis
            self.plot_convergence_analysis(training_history, model_name)
            
            # Create combined dashboard
            if self.interactive and self.save_format in ["plotly", "both"]:
                self.create_interactive_dashboard(training_history, model_name, training_config)
            
            # Save training data
            self.save_training_data(training_history, model_name, training_config)
            
            logger.info(f"Training dashboard created successfully for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to create training dashboard: {e}", exc_info=True)
            raise
    
    def plot_loss_curves(self, 
                        training_history: Dict[str, List[float]], 
                        model_name: str) -> None:
        """Plot training and validation loss curves."""
        epochs = list(range(1, len(training_history.get('train_loss', [])) + 1))
        train_loss = training_history.get('train_loss', [])
        val_loss = training_history.get('val_loss', [])
        
        if not train_loss:
            logger.warning("No training loss data available for plotting")
            return
        
        # Matplotlib version
        if self.save_format in ["matplotlib", "both"]:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            ax.plot(epochs, train_loss, label='Training Loss', linewidth=2, marker='o', markersize=4)
            if val_loss:
                ax.plot(epochs, val_loss, label='Validation Loss', linewidth=2, marker='s', markersize=4)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'Training and Validation Loss - {model_name}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add best epoch marker
            if val_loss:
                best_epoch = np.argmin(val_loss) + 1
                best_loss = min(val_loss)
                ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch: {best_epoch}')
                ax.plot(best_epoch, best_loss, 'ro', markersize=8, label=f'Best Loss: {best_loss:.6f}')
                ax.legend(fontsize=11)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "static" / f"{model_name}_loss_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plotly version
        if self.save_format in ["plotly", "both"]:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=epochs, y=train_loss,
                mode='lines+markers',
                name='Training Loss',
                line=dict(width=3),
                marker=dict(size=6)
            ))
            
            if val_loss:
                fig.add_trace(go.Scatter(
                    x=epochs, y=val_loss,
                    mode='lines+markers',
                    name='Validation Loss',
                    line=dict(width=3),
                    marker=dict(size=6, symbol='square')
                ))
                
                # Add best epoch marker
                best_epoch = np.argmin(val_loss) + 1
                best_loss = min(val_loss)
                fig.add_vline(x=best_epoch, line_dash="dash", line_color="red", 
                             annotation_text=f"Best Epoch: {best_epoch}")
                fig.add_trace(go.Scatter(
                    x=[best_epoch], y=[best_loss],
                    mode='markers',
                    name=f'Best Loss: {best_loss:.6f}',
                    marker=dict(size=12, color='red', symbol='star')
                ))
            
            fig.update_layout(
                title=f'Training and Validation Loss - {model_name}',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                template=self.theme,
                hovermode='x unified',
                width=1000,
                height=600
            )
            
            fig.write_html(self.output_dir / "interactive" / f"{model_name}_loss_curves.html")
    
    def plot_learning_rate_schedule(self, 
                                   training_history: Dict[str, List[float]], 
                                   model_name: str) -> None:
        """Plot learning rate schedule over training."""
        epochs = list(range(1, len(training_history.get('learning_rate', [])) + 1))
        learning_rates = training_history.get('learning_rate', [])
        
        if not learning_rates:
            logger.warning("No learning rate data available for plotting")
            return
        
        # Matplotlib version
        if self.save_format in ["matplotlib", "both"]:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(epochs, learning_rates, linewidth=2, marker='o', markersize=4, color='orange')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Learning Rate', fontsize=12)
            ax.set_title(f'Learning Rate Schedule - {model_name}', fontsize=14, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "static" / f"{model_name}_learning_rate.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plotly version
        if self.save_format in ["plotly", "both"]:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=epochs, y=learning_rates,
                mode='lines+markers',
                name='Learning Rate',
                line=dict(width=3, color='orange'),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title=f'Learning Rate Schedule - {model_name}',
                xaxis_title='Epoch',
                yaxis_title='Learning Rate',
                yaxis_type="log",
                template=self.theme,
                width=1000,
                height=500
            )
            
            fig.write_html(self.output_dir / "interactive" / f"{model_name}_learning_rate.html")
    
    def plot_performance_metrics(self, 
                               training_history: Dict[str, List[float]], 
                               model_name: str) -> None:
        """Plot performance metrics like Sharpe ratio, accuracy, etc."""
        epochs = list(range(1, len(training_history.get('train_loss', [])) + 1))
        
        # Define metrics to plot
        performance_metrics = {
            'sharpe_ratio': 'Sharpe Ratio',
            'max_drawdown': 'Max Drawdown',
            'volatility': 'Volatility',
            'information_ratio': 'Information Ratio',
            'total_return': 'Total Return'
        }
        
        available_metrics = {k: v for k, v in performance_metrics.items() 
                           if k in training_history and training_history[k]}
        
        if not available_metrics:
            logger.warning("No performance metrics available for plotting")
            return
        
        # Matplotlib version
        if self.save_format in ["matplotlib", "both"]:
            n_metrics = len(available_metrics)
            fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 10))
            if n_metrics == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, (metric_key, metric_name) in enumerate(available_metrics.items()):
                values = training_history[metric_key]
                if len(values) == len(epochs):
                    axes[i].plot(epochs, values, linewidth=2, marker='o', markersize=4)
                    axes[i].set_xlabel('Epoch')
                    axes[i].set_ylabel(metric_name)
                    axes[i].set_title(f'{metric_name} - {model_name}')
                    axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(available_metrics), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "static" / f"{model_name}_performance_metrics.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plotly version
        if self.save_format in ["plotly", "both"]:
            fig = make_subplots(
                rows=2, cols=(len(available_metrics) + 1) // 2,
                subplot_titles=[v for v in available_metrics.values()],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            for i, (metric_key, metric_name) in enumerate(available_metrics.items()):
                values = training_history[metric_key]
                if len(values) == len(epochs):
                    row = i // ((len(available_metrics) + 1) // 2) + 1
                    col = i % ((len(available_metrics) + 1) // 2) + 1
                    
                    fig.add_trace(
                        go.Scatter(x=epochs, y=values, mode='lines+markers', 
                                 name=metric_name, showlegend=False),
                        row=row, col=col
                    )
            
            fig.update_layout(
                title=f'Performance Metrics - {model_name}',
                template=self.theme,
                width=1200,
                height=800
            )
            
            fig.write_html(self.output_dir / "interactive" / f"{model_name}_performance_metrics.html")
    
    def plot_system_metrics(self, 
                          training_history: Dict[str, List[float]], 
                          model_name: str) -> None:
        """Plot system metrics like latency, memory usage, GPU utilization."""
        epochs = list(range(1, len(training_history.get('train_loss', [])) + 1))
        
        # Define system metrics to plot
        system_metrics = {
            'batch_time': ('Batch Time (s)', 'Time'),
            'memory_usage': ('Memory Usage (GB)', 'Memory'),
            'gpu_utilization': ('GPU Utilization (%)', 'Utilization'),
            'throughput': ('Throughput (samples/s)', 'Throughput'),
            'gradient_norm': ('Gradient Norm', 'Norm')
        }
        
        available_metrics = {k: v for k, v in system_metrics.items() 
                           if k in training_history and training_history[k]}
        
        if not available_metrics:
            logger.warning("No system metrics available for plotting")
            return
        
        # Matplotlib version
        if self.save_format in ["matplotlib", "both"]:
            n_metrics = len(available_metrics)
            fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 10))
            if n_metrics == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, (metric_key, (metric_name, ylabel)) in enumerate(available_metrics.items()):
                values = training_history[metric_key]
                if len(values) == len(epochs):
                    axes[i].plot(epochs, values, linewidth=2, marker='o', markersize=4)
                    axes[i].set_xlabel('Epoch')
                    axes[i].set_ylabel(ylabel)
                    axes[i].set_title(f'{metric_name} - {model_name}')
                    axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(available_metrics), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "static" / f"{model_name}_system_metrics.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plotly version
        if self.save_format in ["plotly", "both"]:
            fig = make_subplots(
                rows=2, cols=(len(available_metrics) + 1) // 2,
                subplot_titles=[v[0] for v in available_metrics.values()],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            for i, (metric_key, (metric_name, ylabel)) in enumerate(available_metrics.items()):
                values = training_history[metric_key]
                if len(values) == len(epochs):
                    row = i // ((len(available_metrics) + 1) // 2) + 1
                    col = i % ((len(available_metrics) + 1) // 2) + 1
                    
                    fig.add_trace(
                        go.Scatter(x=epochs, y=values, mode='lines+markers', 
                                 name=metric_name, showlegend=False),
                        row=row, col=col
                    )
            
            fig.update_layout(
                title=f'System Metrics - {model_name}',
                template=self.theme,
                width=1200,
                height=800
            )
            
            fig.write_html(self.output_dir / "interactive" / f"{model_name}_system_metrics.html")
    
    def plot_convergence_analysis(self, 
                                training_history: Dict[str, List[float]], 
                                model_name: str) -> None:
        """Plot convergence analysis including loss smoothing and trends."""
        train_loss = training_history.get('train_loss', [])
        val_loss = training_history.get('val_loss', [])
        
        if not train_loss:
            logger.warning("No loss data available for convergence analysis")
            return
        
        epochs = list(range(1, len(train_loss) + 1))
        
        # Calculate smoothed losses (moving average)
        window_size = min(5, len(train_loss) // 4) if len(train_loss) > 10 else 1
        train_loss_smooth = pd.Series(train_loss).rolling(window=window_size, center=True).mean()
        val_loss_smooth = pd.Series(val_loss).rolling(window=window_size, center=True).mean() if val_loss else None
        
        # Matplotlib version
        if self.save_format in ["matplotlib", "both"]:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Raw and smoothed losses
            ax1.plot(epochs, train_loss, alpha=0.3, color='blue', label='Training Loss (Raw)')
            ax1.plot(epochs, train_loss_smooth, color='blue', linewidth=2, label='Training Loss (Smoothed)')
            
            if val_loss:
                ax1.plot(epochs, val_loss, alpha=0.3, color='red', label='Validation Loss (Raw)')
                ax1.plot(epochs, val_loss_smooth, color='red', linewidth=2, label='Validation Loss (Smoothed)')
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'Loss Convergence Analysis - {model_name}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Loss improvement rate
            if len(train_loss) > 1:
                train_improvement = np.diff(train_loss)
                ax2.plot(epochs[1:], train_improvement, color='blue', linewidth=2, label='Training Loss Change')
                
                if val_loss and len(val_loss) > 1:
                    val_improvement = np.diff(val_loss)
                    ax2.plot(epochs[1:], val_improvement, color='red', linewidth=2, label='Validation Loss Change')
                
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss Change')
                ax2.set_title('Loss Improvement Rate')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "static" / f"{model_name}_convergence_analysis.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plotly version
        if self.save_format in ["plotly", "both"]:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Loss Convergence', 'Loss Improvement Rate'],
                vertical_spacing=0.1
            )
            
            # Raw and smoothed losses
            fig.add_trace(
                go.Scatter(x=epochs, y=train_loss, mode='lines', name='Training Loss (Raw)',
                          line=dict(color='blue', width=1), opacity=0.3),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs, y=train_loss_smooth, mode='lines', name='Training Loss (Smoothed)',
                          line=dict(color='blue', width=3)),
                row=1, col=1
            )
            
            if val_loss:
                fig.add_trace(
                    go.Scatter(x=epochs, y=val_loss, mode='lines', name='Validation Loss (Raw)',
                              line=dict(color='red', width=1), opacity=0.3),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=epochs, y=val_loss_smooth, mode='lines', name='Validation Loss (Smoothed)',
                              line=dict(color='red', width=3)),
                    row=1, col=1
                )
            
            # Loss improvement rate
            if len(train_loss) > 1:
                train_improvement = np.diff(train_loss)
                fig.add_trace(
                    go.Scatter(x=epochs[1:], y=train_improvement, mode='lines', 
                              name='Training Loss Change', line=dict(color='blue', width=2)),
                    row=2, col=1
                )
                
                if val_loss and len(val_loss) > 1:
                    val_improvement = np.diff(val_loss)
                    fig.add_trace(
                        go.Scatter(x=epochs[1:], y=val_improvement, mode='lines', 
                                  name='Validation Loss Change', line=dict(color='red', width=2)),
                        row=2, col=1
                    )
                
                fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
            
            fig.update_layout(
                title=f'Convergence Analysis - {model_name}',
                template=self.theme,
                width=1000,
                height=800
            )
            
            fig.write_html(self.output_dir / "interactive" / f"{model_name}_convergence_analysis.html")
    
    def create_interactive_dashboard(self, 
                                   training_history: Dict[str, List[float]], 
                                   model_name: str,
                                   training_config: Optional[Dict[str, Any]] = None) -> None:
        """Create a comprehensive interactive dashboard with all metrics."""
        epochs = list(range(1, len(training_history.get('train_loss', [])) + 1))
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Training & Validation Loss',
                'Learning Rate Schedule',
                'Performance Metrics',
                'System Metrics',
                'Convergence Analysis',
                'Training Summary'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"type": "table"}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Loss curves
        if 'train_loss' in training_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=training_history['train_loss'], 
                          mode='lines+markers', name='Training Loss',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
        
        if 'val_loss' in training_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=training_history['val_loss'], 
                          mode='lines+markers', name='Validation Loss',
                          line=dict(color='red', width=2)),
                row=1, col=1
            )
        
        # 2. Learning rate
        if 'learning_rate' in training_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=training_history['learning_rate'], 
                          mode='lines+markers', name='Learning Rate',
                          line=dict(color='orange', width=2)),
                row=1, col=2
            )
        
        # 3. Performance metrics
        performance_metrics = ['sharpe_ratio', 'max_drawdown', 'volatility']
        colors = ['green', 'purple', 'brown']
        for i, metric in enumerate(performance_metrics):
            if metric in training_history and training_history[metric]:
                fig.add_trace(
                    go.Scatter(x=epochs, y=training_history[metric], 
                              mode='lines+markers', name=metric.replace('_', ' ').title(),
                              line=dict(color=colors[i % len(colors)], width=2)),
                    row=2, col=1
                )
        
        # 4. System metrics
        system_metrics = ['batch_time', 'memory_usage', 'throughput']
        for i, metric in enumerate(system_metrics):
            if metric in training_history and training_history[metric]:
                fig.add_trace(
                    go.Scatter(x=epochs, y=training_history[metric], 
                              mode='lines+markers', name=metric.replace('_', ' ').title(),
                              line=dict(color=colors[i % len(colors)], width=2)),
                    row=2, col=2
                )
        
        # 5. Convergence analysis
        if 'train_loss' in training_history and len(training_history['train_loss']) > 1:
            train_improvement = np.diff(training_history['train_loss'])
            fig.add_trace(
                go.Scatter(x=epochs[1:], y=train_improvement, 
                          mode='lines', name='Training Loss Change',
                          line=dict(color='blue', width=2)),
                row=3, col=1
            )
        
        # 6. Training summary table
        summary_data = self._create_training_summary(training_history, training_config)
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'], fill_color='lightblue'),
                cells=dict(values=[list(summary_data.keys()), list(summary_data.values())],
                          fill_color='white')
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Training Dashboard - {model_name}',
            template=self.theme,
            width=1400,
            height=1200,
            showlegend=True
        )
        
        # Update y-axis for learning rate to log scale
        fig.update_yaxes(type="log", row=1, col=2)
        
        fig.write_html(self.output_dir / "interactive" / f"{model_name}_dashboard.html")
    
    def _create_training_summary(self, 
                               training_history: Dict[str, List[float]], 
                               training_config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Create a summary of training metrics."""
        summary = {}
        
        # Basic metrics
        if 'train_loss' in training_history:
            summary['Total Epochs'] = str(len(training_history['train_loss']))
            summary['Final Train Loss'] = f"{training_history['train_loss'][-1]:.6f}"
        
        if 'val_loss' in training_history:
            summary['Final Val Loss'] = f"{training_history['val_loss'][-1]:.6f}"
            summary['Best Val Loss'] = f"{min(training_history['val_loss']):.6f}"
            summary['Best Epoch'] = str(np.argmin(training_history['val_loss']) + 1)
        
        # Performance metrics
        if 'sharpe_ratio' in training_history and training_history['sharpe_ratio']:
            summary['Final Sharpe Ratio'] = f"{training_history['sharpe_ratio'][-1]:.4f}"
        
        if 'max_drawdown' in training_history and training_history['max_drawdown']:
            summary['Max Drawdown'] = f"{training_history['max_drawdown'][-1]:.4f}"
        
        # System metrics
        if 'batch_time' in training_history and training_history['batch_time']:
            summary['Avg Batch Time'] = f"{np.mean(training_history['batch_time']):.3f}s"
        
        if 'memory_usage' in training_history and training_history['memory_usage']:
            summary['Peak Memory'] = f"{max(training_history['memory_usage']):.2f}GB"
        
        if 'throughput' in training_history and training_history['throughput']:
            summary['Avg Throughput'] = f"{np.mean(training_history['throughput']):.1f} samples/s"
        
        # Training config
        if training_config:
            summary['Learning Rate'] = str(training_config.get('learning_rate', 'N/A'))
            summary['Batch Size'] = str(training_config.get('batch_size', 'N/A'))
            summary['Optimizer'] = str(training_config.get('optimizer', {}).get('type', 'N/A'))
        
        return summary
    
    def save_training_data(self, 
                         training_history: Dict[str, List[float]], 
                         model_name: str,
                         training_config: Optional[Dict[str, Any]] = None) -> None:
        """Save training data to JSON for future analysis."""
        data = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'training_history': training_history,
            'training_config': training_config,
            'summary': self._create_training_summary(training_history, training_config)
        }
        
        output_file = self.output_dir / "data" / f"{model_name}_training_data.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Training data saved to {output_file}")
    
    def plot_model_predictions(self, 
                             predictions: np.ndarray, 
                             actual_returns: np.ndarray,
                             model_name: str,
                             asset_names: Optional[List[str]] = None) -> None:
        """
        Plot model predictions vs actual returns.
        
        Args:
            predictions: Model predictions array
            actual_returns: Actual returns array
            model_name: Name of the model
            asset_names: Optional list of asset names
        """
        if predictions.shape != actual_returns.shape:
            logger.warning("Predictions and actual returns have different shapes")
            return
        
        # Select a subset of assets for visualization if too many
        n_assets = min(10, predictions.shape[1]) if len(predictions.shape) > 1 else 1
        
        if len(predictions.shape) > 1:
            pred_subset = predictions[:, :n_assets]
            actual_subset = actual_returns[:, :n_assets]
        else:
            pred_subset = predictions.reshape(-1, 1)
            actual_subset = actual_returns.reshape(-1, 1)
        
        # Matplotlib version
        if self.save_format in ["matplotlib", "both"]:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Scatter plot
            axes[0, 0].scatter(actual_subset.flatten(), pred_subset.flatten(), alpha=0.6)
            axes[0, 0].plot([actual_subset.min(), actual_subset.max()], 
                           [actual_subset.min(), actual_subset.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual Returns')
            axes[0, 0].set_ylabel('Predicted Returns')
            axes[0, 0].set_title('Predictions vs Actual Returns')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Time series for first asset
            time_steps = range(len(pred_subset))
            axes[0, 1].plot(time_steps, actual_subset[:, 0], label='Actual', linewidth=2)
            axes[0, 1].plot(time_steps, pred_subset[:, 0], label='Predicted', linewidth=2)
            axes[0, 1].set_xlabel('Time Step')
            axes[0, 1].set_ylabel('Returns')
            axes[0, 1].set_title('Time Series Comparison (Asset 1)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Residuals
            residuals = pred_subset - actual_subset
            axes[1, 0].hist(residuals.flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Residuals Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Correlation heatmap
            if n_assets > 1:
                correlation_matrix = np.corrcoef(pred_subset.T, actual_subset.T)
                im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                axes[1, 1].set_title('Prediction-Actual Correlation Matrix')
                plt.colorbar(im, ax=axes[1, 1])
            else:
                axes[1, 1].text(0.5, 0.5, 'Single Asset\nCorrelation Analysis', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Correlation Analysis')
            
            plt.suptitle(f'Model Predictions Analysis - {model_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / "static" / f"{model_name}_predictions.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plotly version
        if self.save_format in ["plotly", "both"]:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Predictions vs Actual', 'Time Series (Asset 1)', 
                               'Residuals Distribution', 'Correlation Analysis'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Scatter plot
            fig.add_trace(
                go.Scatter(x=actual_subset.flatten(), y=pred_subset.flatten(),
                          mode='markers', name='Predictions',
                          marker=dict(size=4, opacity=0.6)),
                row=1, col=1
            )
            
            # Perfect prediction line
            min_val, max_val = actual_subset.min(), actual_subset.max()
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                          mode='lines', name='Perfect Prediction',
                          line=dict(color='red', dash='dash', width=2)),
                row=1, col=1
            )
            
            # Time series
            time_steps = list(range(len(pred_subset)))
            fig.add_trace(
                go.Scatter(x=time_steps, y=actual_subset[:, 0],
                          mode='lines', name='Actual',
                          line=dict(width=2)),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=time_steps, y=pred_subset[:, 0],
                          mode='lines', name='Predicted',
                          line=dict(width=2)),
                row=1, col=2
            )
            
            # Residuals histogram
            residuals = pred_subset - actual_subset
            fig.add_trace(
                go.Histogram(x=residuals.flatten(), name='Residuals',
                           nbinsx=50, opacity=0.7),
                row=2, col=1
            )
            
            # Correlation heatmap
            if n_assets > 1:
                correlation_matrix = np.corrcoef(pred_subset.T, actual_subset.T)
                fig.add_trace(
                    go.Heatmap(z=correlation_matrix, colorscale='RdBu',
                              zmid=0, name='Correlation'),
                    row=2, col=2
                )
            
            fig.update_layout(
                title=f'Model Predictions Analysis - {model_name}',
                template=self.theme,
                width=1200,
                height=900
            )
            
            fig.write_html(self.output_dir / "interactive" / f"{model_name}_predictions.html")


def create_training_visualizations(training_metrics: List[Dict[str, Any]], 
                                 model_name: str,
                                 output_dir: str = "training_visuals",
                                 training_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Convenience function to create all training visualizations.
    
    Args:
        training_metrics: List of training metrics dictionaries from each epoch
        model_name: Name of the model
        output_dir: Directory to save visualizations
        training_config: Training configuration parameters
    """
    # Convert list of metrics to history format
    training_history = {}
    
    if training_metrics:
        # Get all metric keys from the first entry
        metric_keys = set()
        for metrics in training_metrics:
            metric_keys.update(metrics.keys())
        
        # Convert to history format
        for key in metric_keys:
            training_history[key] = []
            for metrics in training_metrics:
                if key in metrics:
                    training_history[key].append(metrics[key])
    
    # Create visualizer and generate charts
    visualizer = TrainingVisualizer(output_dir=output_dir)
    visualizer.create_training_dashboard(training_history, model_name, training_config)
    
    logger.info(f"Training visualizations created for {model_name} in {output_dir}")


# Example usage
if __name__ == "__main__":
    # Example training history data
    example_history = {
        'train_loss': [0.1, 0.08, 0.06, 0.05, 0.04, 0.035, 0.03, 0.028, 0.025, 0.023],
        'val_loss': [0.12, 0.09, 0.07, 0.055, 0.045, 0.04, 0.038, 0.036, 0.034, 0.032],
        'learning_rate': [0.001, 0.001, 0.001, 0.0005, 0.0005, 0.0005, 0.00025, 0.00025, 0.00025, 0.00025],
        'sharpe_ratio': [0.5, 0.7, 0.9, 1.1, 1.3, 1.4, 1.5, 1.6, 1.65, 1.7],
        'max_drawdown': [0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.055, 0.05, 0.048, 0.045],
        'batch_time': [0.5, 0.48, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40, 0.39],
        'memory_usage': [2.1, 2.2, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6],
        'throughput': [128, 132, 135, 138, 140, 142, 145, 148, 150, 152]
    }
    
    example_config = {
        'learning_rate': 0.001,
        'batch_size': 64,
        'optimizer': {'type': 'AdamW'},
        'epochs': 10
    }
    
    # Create visualizations
    visualizer = TrainingVisualizer(output_dir="example_visuals")
    visualizer.create_training_dashboard(example_history, "example_model", example_config)
    
    print("Example visualizations created in 'example_visuals' directory")