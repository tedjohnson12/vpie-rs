"""
Python interface for VPIE

Most of the interesting stuff is written in Rust. This is just an easy interface.
"""

__version__ = "0.1.0"
from .vpie import search_next_best, get_coeffs, get_reconstruction, get_vpie
from .retrieve import Parameter, Prior
