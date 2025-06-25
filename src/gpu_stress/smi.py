# SPDX-FileCopyrightText: 2025 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Query device information with external smi interface."""


# Type annotations
from __future__ import annotations
from typing import List, Tuple, Callable, Final, Type

# Standard libs
from collections import defaultdict
from subprocess import check_output
import re

# Public interface
__all__ = ['parse_nvidia_smi', 'parse_rocm_smi', 'parse_xpu_smi', 'gpu_info']


def parse_nvidia_smi(output: str) -> List[str]:
    """Parse output of nvidia-smi."""
    return output.splitlines()


def parse_rocm_smi(output: str) -> List[str]:
    """Parse output of rocm-smi."""
    info = defaultdict(dict)
    for line in output.splitlines():
        if not line.startswith('GPU'):
            continue
        gpu_id, info_key, info_value = line.split(':')
        gpu_id = gpu_id.replace('[', ' ').replace(']', '').strip()
        info[gpu_id][info_key.strip()] = info_value.strip()
    return [
        f'{k}: AMD {v["Device Name"]} {v["Device ID"]} (GUID: {v["GUID"]})'
        for k, v in info.items()
    ]


def parse_xpu_smi(output: str) -> List[str]:
    """Parse output of xpu-smi."""
    section_boundary = re.compile(f'^\+(-)+\+(-)+\+$')
    section_id = -1
    info = defaultdict(dict)
    for line in output.splitlines():
        if section_boundary.match(line):
            continue
        if section_id == -1:
            assert 'Device ID' in line
            assert 'Device Information' in line
            section_id = 0
            continue
        if 'Device Name' in line:
            parts = line.split()
            section_id = int(parts[1])
            info[section_id]['name'] = ' '.join(parts[6:-1])
        if 'UUID' in line:
            parts = line.split()
            info[section_id]['uuid'] = parts[4]
    return [
        f'GPU {k}: {v["name"]} (UUID: {v["uuid"]})'
        for k, v in info.items()
    ]


SMI_FUNCTION: Type = Callable[[str, ], List[str]]
SMI_INFO: Type = Tuple[str, str, SMI_FUNCTION]
SMI_MAP: Final[List[SMI_INFO]] = [
    ('nvidia-smi', 'nvidia-smi --list-gpus', parse_nvidia_smi),
    ('rocm-smi', 'rocm-smi -i', parse_rocm_smi),
    ('xpu-smi', 'xpu-smi discovery', parse_xpu_smi),
]


def gpu_info() -> List[str]:
    """Get GPU device information."""
    for name, command, parser in SMI_MAP:
        try:
            return parser(check_output(command.split()).decode('utf-8').strip())
        except FileNotFoundError:
            pass
        except Exception as exc:
            raise RuntimeError(f'Failed to retrieve GPU information: {exc}') from exc
    else:
        program_names = ', '.join([info[0] for info in SMI_MAP])
        raise RuntimeError(f'GPU interface not found ({program_names})')
