from __future__ import annotations
import logging, os, sys

def get_logger(name: str = "spectra"):
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger
