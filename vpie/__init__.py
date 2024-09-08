"""
Python interface for VPIE

Most of the interesting stuff is written in Rust. This is just an easy interface.
"""

from typing import Set
import numpy as np


from .wrappers import search_next_best, get_coeffs, get_reconstruction

def get_vpie(
    f_org: np.ndarray,
    f_err: np.ndarray,
    cutoff_index: int,
    use_mean_error: bool
) -> np.ndarray:
    """
    Get the important quantities needed to do a retrieval later.
    
    Parameters
    ----------
    f_org : np.ndarray
        The original spectra to be approximated.
    f_err : np.ndarray
        The error of the spectra.
    cutoff_index : int
        The number of wavelength points to use in the basis.
    use_mean_error : bool
        Whether to use the mean error in the reconstruction.
    
    Returns
    -------
    s : set of int
        The set of bases.
    coeffs : np.ndarray
        The basis coefficients.
    f_rec : np.ndarray
        The reconstructed observation.
    """
    s: Set[int] = search_next_best(f_org, f_err, cutoff_index, use_mean_error)
    coeffs: np.ndarray = get_coeffs(f_org[:,:cutoff_index], f_err[:,:cutoff_index], s, use_mean_error)
    f_rec: np.ndarray = get_reconstruction(f_org, coeffs, s)
    return s, coeffs, f_rec