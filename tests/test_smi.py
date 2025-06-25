# SPDX-FileCopyrightText: 2025 Purdue RCAC
# SPDX-License-Identifier: MIT

"""Test gpu information parsing."""


# Type annotations
from __future__ import annotations
from typing import Final, List

# External libs
from pytest import mark

# Internal libs
from gpu_stress.smi import parse_nvidia_smi, parse_rocm_smi, parse_xpu_smi


NVIDIA_EXAMPLE: Final[str] = """\
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-deb02430-f8f0-88a9-2871-7cc1acc99eee)
GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-13e10222-7b1d-85b6-0a60-46b35bc724dc)
GPU 2: NVIDIA H100 80GB HBM3 (UUID: GPU-ba212763-9fab-3158-972e-de212ed315b4)
GPU 3: NVIDIA H100 80GB HBM3 (UUID: GPU-056c1c5b-5002-c6f3-7c52-19b80090d8fd)
GPU 4: NVIDIA H100 80GB HBM3 (UUID: GPU-09290c1c-6b1e-f85f-295f-fb6d97907e25)
GPU 5: NVIDIA H100 80GB HBM3 (UUID: GPU-5d39a166-54d9-aaf0-616d-9cf0b9a1ca10)
GPU 6: NVIDIA H100 80GB HBM3 (UUID: GPU-ac34c587-c949-7949-fc5d-3f62199b38aa)
GPU 7: NVIDIA H100 80GB HBM3 (UUID: GPU-c674bee4-9599-7385-18a5-ee64430a859a)
"""


NVIDIA_EXAMPLE_RESULT: Final[List[str]] = [
    'GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-deb02430-f8f0-88a9-2871-7cc1acc99eee)',
    'GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-13e10222-7b1d-85b6-0a60-46b35bc724dc)',
    'GPU 2: NVIDIA H100 80GB HBM3 (UUID: GPU-ba212763-9fab-3158-972e-de212ed315b4)',
    'GPU 3: NVIDIA H100 80GB HBM3 (UUID: GPU-056c1c5b-5002-c6f3-7c52-19b80090d8fd)',
    'GPU 4: NVIDIA H100 80GB HBM3 (UUID: GPU-09290c1c-6b1e-f85f-295f-fb6d97907e25)',
    'GPU 5: NVIDIA H100 80GB HBM3 (UUID: GPU-5d39a166-54d9-aaf0-616d-9cf0b9a1ca10)',
    'GPU 6: NVIDIA H100 80GB HBM3 (UUID: GPU-ac34c587-c949-7949-fc5d-3f62199b38aa)',
    'GPU 7: NVIDIA H100 80GB HBM3 (UUID: GPU-c674bee4-9599-7385-18a5-ee64430a859a)',
]


@mark.unit
def test_nvidia_smi() -> None:
    """Test nvidia-smi parser."""
    assert NVIDIA_EXAMPLE_RESULT == parse_nvidia_smi(NVIDIA_EXAMPLE)


ROCM_EXAMPLE: Final[str] = """\
============================ ROCm System Management Interface ============================
=========================================== ID ===========================================
GPU[0]          : Device Name:          Instinct MI210
GPU[0]          : Device ID:            0x740f
GPU[0]          : Device Rev:           0x02
GPU[0]          : Subsystem ID:         0x0c34
GPU[0]          : GUID:                 13566
GPU[1]          : Device Name:          Instinct MI210
GPU[1]          : Device ID:            0x740f
GPU[1]          : Device Rev:           0x02
GPU[1]          : Subsystem ID:         0x0c34
GPU[1]          : GUID:                 36740
GPU[2]          : Device Name:          Instinct MI210
GPU[2]          : Device ID:            0x740f
GPU[2]          : Device Rev:           0x02
GPU[2]          : Subsystem ID:         0x0c34
GPU[2]          : GUID:                 38382
==========================================================================================
================================== End of ROCm SMI Log ===================================
"""


ROCM_EXAMPLE_RESULT: Final[List[str]] = [
    'GPU 0: AMD Instinct MI210 0x740f (GUID: 13566)',
    'GPU 1: AMD Instinct MI210 0x740f (GUID: 36740)',
    'GPU 2: AMD Instinct MI210 0x740f (GUID: 38382)',
]


@mark.unit
def test_rocm_smi() -> None:
    """Test rocm-smi parser."""
    assert ROCM_EXAMPLE_RESULT == parse_rocm_smi(ROCM_EXAMPLE)


XPU_EXAMPLE: Final[str] = """\
+-----------+--------------------------------------------------------------------------------------+
| Device ID | Device Information                                                                   |
+-----------+--------------------------------------------------------------------------------------+
| 0         | Device Name: Intel(R) Data Center GPU Max 1550                                       |
|           | Vendor Name: Intel(R) Corporation                                                    |
|           | SOC UUID: 00000000-0000-0000-51ac-359875c9cdb7                                       |
|           | PCI BDF Address: 0000:18:00.0                                                        |
|           | DRM Device: /dev/dri/card0                                                           |
|           | Function Type: physical                                                              |
+-----------+--------------------------------------------------------------------------------------+
| 1         | Device Name: Intel(R) Data Center GPU Max 1550                                       |
|           | Vendor Name: Intel(R) Corporation                                                    |
|           | SOC UUID: 00000000-0000-0000-c920-f846ca881f43                                       |
|           | PCI BDF Address: 0000:42:00.0                                                        |
|           | DRM Device: /dev/dri/card1                                                           |
|           | Function Type: physical                                                              |
+-----------+--------------------------------------------------------------------------------------+
| 2         | Device Name: Intel(R) Data Center GPU Max 1550                                       |
|           | Vendor Name: Intel(R) Corporation                                                    |
|           | SOC UUID: 00000000-0000-0000-3433-296571c957dd                                       |
|           | PCI BDF Address: 0000:6c:00.0                                                        |
|           | DRM Device: /dev/dri/card2                                                           |
|           | Function Type: physical                                                              |
+-----------+--------------------------------------------------------------------------------------+
| 3         | Device Name: Intel(R) Data Center GPU Max 1550                                       |
|           | Vendor Name: Intel(R) Corporation                                                    |
|           | SOC UUID: 00000000-0000-0000-e12e-5bce745481f2                                       |
|           | PCI BDF Address: 0001:18:00.0                                                        |
|           | DRM Device: /dev/dri/card3                                                           |
|           | Function Type: physical                                                              |
+-----------+--------------------------------------------------------------------------------------+
| 4         | Device Name: Intel(R) Data Center GPU Max 1550                                       |
|           | Vendor Name: Intel(R) Corporation                                                    |
|           | SOC UUID: 00000000-0000-0000-584d-2fffd73f8b78                                       |
|           | PCI BDF Address: 0001:42:00.0                                                        |
|           | DRM Device: /dev/dri/card4                                                           |
|           | Function Type: physical                                                              |
+-----------+--------------------------------------------------------------------------------------+
| 5         | Device Name: Intel(R) Data Center GPU Max 1550                                       |
|           | Vendor Name: Intel(R) Corporation                                                    |
|           | SOC UUID: 00000000-0000-0000-9a16-c2b794b6fad7                                       |
|           | PCI BDF Address: 0001:6c:00.0                                                        |
|           | DRM Device: /dev/dri/card5                                                           |
|           | Function Type: physical                                                              |
+-----------+--------------------------------------------------------------------------------------+
"""


XPU_EXAMPLE_RESULT: Final[List[str]] = [
    'GPU 0: Data Center GPU Max 1550 (UUID: 00000000-0000-0000-51ac-359875c9cdb7)',
    'GPU 1: Data Center GPU Max 1550 (UUID: 00000000-0000-0000-c920-f846ca881f43)',
    'GPU 2: Data Center GPU Max 1550 (UUID: 00000000-0000-0000-3433-296571c957dd)',
    'GPU 3: Data Center GPU Max 1550 (UUID: 00000000-0000-0000-e12e-5bce745481f2)',
    'GPU 4: Data Center GPU Max 1550 (UUID: 00000000-0000-0000-584d-2fffd73f8b78)',
    'GPU 5: Data Center GPU Max 1550 (UUID: 00000000-0000-0000-9a16-c2b794b6fad7)'
]


@mark.unit
def test_xpu_smi() -> None:
    """Test xpu-smi parser."""
    assert XPU_EXAMPLE_RESULT == parse_xpu_smi(XPU_EXAMPLE)
