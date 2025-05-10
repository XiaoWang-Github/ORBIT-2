from enum import Enum
"""
This module defines the `FusedAttn` enumeration
Classes:
    FusedAttn (Enum): An enumeration representing the modes of fused attention.
        - CK: Represents the "CK" mode using ROCm Composable Kernels.
        - DEFAULT: Represents the "DEFAULT" mode using PyTorch/Triton.
        - NONE: Represents no fused attention.

"""

class FusedAttn(Enum):
    CK = "CK"
    DEFAULT = "DEFAULT"
    NONE = "NONE"

