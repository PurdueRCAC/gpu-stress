# SPDX-FileCopyrightText: 2025 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Base configuration and core utilities."""


# Type annotations
from __future__ import annotations
from typing import Final, List

# Standard libs
import os
import re
import sys
from subprocess import check_output
from datetime import timedelta

# External libs
from cmdkit.app import exit_status
from cmdkit.config import Configuration, Namespace
from cmdkit.logging import Logger, level_by_name, logging_styles

# Public interface
__all__ = [
    'default_config', 'DEFAULT_SIZE', 'DEFAULT_DEVICE', 'DEFAULT_TIME',
    'ctx', 'cfg', 'log', 
    'handle_exception', 'parse_time', 'get_gpu_info',
]


default_config: Final[Namespace] = Namespace({
	'size': 10_000,      # Size of matrix to use
	'device': 0,         # Which device ID to use
	'time': '60s',       # Time to run
    'log': {
        'level': 'info',
        'style': 'default',
    },
})


DEFAULT_SIZE: Final[int] = default_config.size
DEFAULT_DEVICE: Final[int] = default_config.device
DEFAULT_TIME: Final[str] = default_config.time


try:
    ctx, cfg = Configuration.from_context('GPUStress', create_dirs=True, config_format='toml',
                                          default_config=default_config)
    log = Logger.default('gpu-stress',
                         level=level_by_name[cfg.log.level.upper()],
                         **logging_styles[cfg.log.style.lower()])
except Exception as exc:
    print(f'Error [{exc.__class__.__name__}] {exc}')
    sys.exit(exit_status.bad_config)


def handle_exception(exc: Exception, status: int) -> int:
    """Handle exceptions raised by the application."""
    log.critical(f'{exc.__class__.__name__}: {exc}')
    return status


def parse_time(time_spec: str) -> timedelta:
    """Solve user input time_spec."""
    if match := re.match(r'(?P<value>[0-9.]+)(?P<unit>s|m)?', time_spec.lower()):
        value = float(match.group('value'))
        unit = match.group('unit')
        match unit:
            case 'm':
                return timedelta(minutes=value)
            case 's':
                return timedelta(seconds=value)
    raise ValueError(f'Invalid time specification: {time_spec!r}')


def get_gpu_info() -> List[str]:
    """Get GPU device information."""
    try:
        return check_output(['nvidia-smi', '--list-gpus']).decode('utf-8').strip().splitlines()
    except Exception as exc:
        raise RuntimeError(f"Failed to retrieve GPU information: {exc}") from exc

