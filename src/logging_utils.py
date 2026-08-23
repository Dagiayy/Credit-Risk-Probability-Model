"""Structured logging setup shared by every entry point (CLI scripts and API).

Replaces the ad-hoc `print("emoji ...")` statements that made it impossible to
tell, in a container's stdout, what ran, when, and whether it failed.
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    if json_format:
        fmt = (
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        )
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers = [handler]
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
