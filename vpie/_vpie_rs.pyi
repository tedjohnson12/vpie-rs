"""
Stub file for _vpie_rs

These functions are defined in src/lib.rs
"""
from typing import Set
import numpy as np


def search_next_best(
    f_org: np.ndarray,
    f_err: np.ndarray,
    cutoff_index: int,
    use_mean_error: bool
) -> Set[int]:
    """
    Use the Next Best Algorithm to find the best set of bases.
    
    Parameters
    ----------
    f_org : np.ndarray (m, n)
        The original spectra to be approximated.
    f_err : np.ndarray (m, n)
        The error of the spectra.
    cutoff_index : int
        The number of wavelength points to use in the basis.
    use_mean_error : bool
        Whether to use the mean error in the reconstruction.
    
    Returns
    -------
    set of int (length q)
        The set of bases.
    """
    ...


def get_coeffs(
    f_org_nir: np.ndarray,
    f_err_nir: np.ndarray,
    s: Set[int],
    use_mean_error: bool
) -> np.ndarray:
    """
    Get the coefficient matrix.
    
    Parameters
    ----------
    f_org_nir : np.ndarray (m, n)
        The original spectra to be approximated.
    f_err_nir : np.ndarray (m, n)
        The error of the spectra.
    s : set of int (length q)
        The set of bases.
    use_mean_error : bool
        Whether to use the mean error in the reconstruction.
    
    Returns
    -------
    np.ndarray (m, q)
        The basis coefficients.
    """
    ...
    
    
def get_reconstruction(
    flux: np.ndarray,
    coeffs: np.ndarray,
    s: Set[int],
)-> np.ndarray:
    """
    Reconstruct some phase curve given the basis coefficients and the basis set :math:`s`.
    
    Parameters
    ----------
    flux : np.ndarray (m, n)
        The phase curve to be reconstructed.
    coeffs : np.ndarray (m, q)
        The basis coefficients.
    s : set of int (length q)
        The set of bases.
    
    Returns
    -------
    np.ndarray (m, n)
        The reconstructed observation.
    """
    ...