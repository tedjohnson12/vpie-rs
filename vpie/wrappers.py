"""
Python wrappers for Rust functions.

Just so language servers can find them.
"""

import numpy as np
from typing import Set

from . import _vpie_rs

def search_next_best(
    f_org: np.ndarray,
    f_err: np.ndarray,
    cutoff_index: int,
    use_mean_error: bool
) -> Set[int]:
    """
    Search for the best set of bases using the Next Best Algorithm.
    
    This algoritm starts with :math:`q=1` basis. Once it finds the best basis with :math:`q=1`, it
    it uses that basis as the starting point for finding a set of bases with :math:`q=2`.
    
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
    set of int
        The set of bases.
    """
    return _vpie_rs.search_next_best(f_org, f_err, cutoff_index, use_mean_error)
    

def get_coeffs(
    f_org_nir: np.ndarray,
    f_err_nir: np.ndarray,
    s: Set[int],
    use_mean_error: bool
):
    """
    Get the basis coefficients given the basis set :math:`s`.
    
    Parameters
    ----------
    f_org_nir : np.ndarray
        The original spectra to be approximated.
    f_err_nir : np.ndarray
        The error of the spectra.
    s : set of int
        The set of bases.
    use_mean_error : bool
        Whether to use the mean error in the reconstruction.
    
    Returns
    -------
    np.ndarray
        The basis coefficients.
    """
    return _vpie_rs.get_coeffs(f_org_nir, f_err_nir, s, use_mean_error)