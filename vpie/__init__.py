"""
Python interface for VPIE

Most of the interesting stuff is written in Rust. This is just an easy interface.
"""

__version__ = "0.1.0"
import logging
from colorlog import ColoredFormatter

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter(
    "%(log_color)s[%(levelname)-5s %(name)s]%(reset)s %(message)s"
))

root = logging.getLogger()
root.setLevel(logging.INFO)
root.addHandler(handler)


from .vpie import search_next_best, get_coeffs, get_reconstruction, get_vpie, bin_image, fold_image
from .retrieve import Parameter, Prior