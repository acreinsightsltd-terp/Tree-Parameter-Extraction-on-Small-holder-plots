# logging_config.py
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(log_dir="logs", level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    # File handler (rotating = production safe)
    file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, "pipeline.log"),
        maxBytes=5_000_000,
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=level,
        handlers=[console, file_handler]
    )
