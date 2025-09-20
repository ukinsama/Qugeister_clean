"""
Enhanced logging configuration for Qugeister system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime
import json


class QuantumFormatter(logging.Formatter):
    """Custom formatter for quantum computing logs"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = datetime.now()

    def format(self, record):
        # Add quantum-specific context
        if hasattr(record, "qubits"):
            record.msg = f"[{record.qubits}Q] {record.msg}"
        if hasattr(record, "circuit_depth"):
            record.msg = f"[D{record.circuit_depth}] {record.msg}"

        return super().format(record)


class MetricsFilter(logging.Filter):
    """Filter for performance metrics"""

    def filter(self, record):
        # Only log metrics if they contain performance data
        return hasattr(record, "metrics") or record.levelno >= logging.WARNING


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
    enable_metrics: bool = False,
    json_logs: bool = False,
) -> logging.Logger:
    """Setup enhanced logging configuration for Qugeister system

    Args:
        level: Logging level (INFO, DEBUG, etc.)
        log_file: Optional file path for log output
        log_format: Custom log format string
        enable_metrics: Enable performance metrics logging
        json_logs: Output logs in JSON format
    """

    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Default formats
    if json_logs:
        log_format = None  # JSON formatter handles this
    elif log_format is None:
        log_format = "%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s"

    # Create main logger
    logger = logging.getLogger("qugeister")
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler with color support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if json_logs:
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(QuantumFormatter(log_format))

    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        if json_logs:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(QuantumFormatter(log_format))

        logger.addHandler(file_handler)

    # Metrics handler for performance tracking
    if enable_metrics:
        metrics_handler = MetricsHandler()
        metrics_handler.addFilter(MetricsFilter())
        logger.addHandler(metrics_handler)

    # Set up specialized loggers
    setup_quantum_logger(level)
    setup_training_logger(level)

    logger.info("Logging system initialized")
    return logger


def setup_quantum_logger(level: int) -> logging.Logger:
    """Setup specialized logger for quantum operations"""
    logger = logging.getLogger("qugeister.quantum")
    logger.setLevel(level)
    return logger


def setup_training_logger(level: int) -> logging.Logger:
    """Setup specialized logger for training operations"""
    logger = logging.getLogger("qugeister.training")
    logger.setLevel(level)
    return logger


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logs"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add quantum-specific fields
        for attr in ["qubits", "circuit_depth", "metrics", "episode", "loss"]:
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)

        return json.dumps(log_entry)


class MetricsHandler(logging.Handler):
    """Handler for collecting performance metrics"""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def emit(self, record):
        if hasattr(record, "metrics"):
            self.metrics.append(
                {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "metrics": record.metrics,
                }
            )

    def get_metrics(self) -> list:
        """Get collected metrics"""
        return self.metrics

    def clear_metrics(self):
        """Clear collected metrics"""
        self.metrics.clear()


# Convenience functions for quantum logging
def log_quantum_info(qubits: int, circuit_depth: int, message: str):
    """Log quantum circuit information"""
    logger = logging.getLogger("qugeister.quantum")
    logger.info(message, extra={"qubits": qubits, "circuit_depth": circuit_depth})


def log_training_metrics(episode: int, loss: float, metrics: Dict[str, Any]):
    """Log training metrics"""
    logger = logging.getLogger("qugeister.training")
    logger.info(
        f"Episode {episode} completed",
        extra={"episode": episode, "loss": loss, "metrics": metrics},
    )
