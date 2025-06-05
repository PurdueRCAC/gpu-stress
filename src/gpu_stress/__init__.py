# SPDX-FileCopyrightText: 2025 Purdue RCAC
# SPDX-License-Identifier: MIT

"""A simple GPU stress test application."""


# Type annotations
from __future__ import annotations
from typing import Final, List

# Standard libs
import sys
from importlib.metadata import version as get_version
from functools import partial
from datetime import datetime, timedelta
from platform import python_implementation, python_version

# External libs
from cmdkit.cli import Interface
from cmdkit.app import Application, exit_status
from cmdkit.logging import level_by_name
import torch

# Internal libs
from gpu_stress.core import (
    DEFAULT_SIZE, DEFAULT_DEVICE, DEFAULT_TIME, 
    cfg, log, handle_exception, parse_time, get_gpu_info
)

# Public interface
__all__ = [
    'GPUStress', 'main', 'PROGNAME', 'VERSION', 'USAGE', 'HELP',
]

# Metadata
__version__ = get_version('gpu-stress')


PROGNAME: Final[str] = 'gpu-stress'
VERSION: Final[str] = f'{PROGNAME} v{__version__} ({python_implementation()} {python_version()})'
USAGE: Final[str] = f"""\
Usage:
  {PROGNAME} [-vh] [-t TIME] [-s SIZE] [-d DEVICE]
  Stress Nvidia device for some period of time.\
"""
HELP: Final[str] = f"""\
{USAGE}

Options:
  -t, --time      TIME     Minimum runtime for program (default: '{DEFAULT_TIME}').
  -s, --size      SIZE     Size of matrix to compute (default: {DEFAULT_SIZE}).
  -d, --device    DEVICE   Device to use (default: {DEFAULT_DEVICE}).
      --debug              Set logging level to debug.
  -v, --version            Show version info and exit.
  -h, --help               Show this message and exit.\
"""




class GPUStress(Application):
    """Application interface."""

    interface = Interface(PROGNAME, USAGE, HELP)
    interface.add_argument('-v', '--version', action='version', version=VERSION)

    size: int = DEFAULT_SIZE
    interface.add_argument('-s', '--size', type=int, default=size)

    device: int = DEFAULT_DEVICE
    interface.add_argument('-d', '--device', type=int, default=device)

    runtime: timedelta = parse_time(DEFAULT_TIME)
    interface.add_argument('-t', '--time', type=parse_time, default=runtime, dest='runtime')

    log_level: str = cfg.log.level
    interface.add_argument('--debug', action='store_const', const='debug',
                           default=log_level, dest='log_level')

    exceptions = {
        RuntimeError: partial(handle_exception, status=exit_status.runtime_error),
        ValueError: partial(handle_exception, status=exit_status.bad_config),
        Exception: partial(handle_exception, status=exit_status.uncaught_exception),
    }

    def run(self: GPUStress) -> None:
        """Run program."""
        log.setLevel(level_by_name[self.log_level.upper()])
        for info in get_gpu_info():
            log.info(info)
        log.info(f'Solving matrix (size={self.size}) for approx {self.runtime} on device {self.device}')
        device = torch.device(f'cuda:{self.device}')
        start_time = datetime.now()
        random1 = torch.randn([self.size, self.size]).to(device)
        random2 = torch.randn([self.size, self.size]).to(device)
        while datetime.now() - start_time < self.runtime:
            random1 = random1 * random2
            random2 = random2 * random1
        log.info(f'Elapsed {datetime.now() - start_time}')


def main(argv: List[str] | None = None) -> int:
    """Main entry-point for the program."""
    return GPUStress.main(argv or sys.argv[1:])
