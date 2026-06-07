"""
Stub file for _vpie_rs

These functions are defined in src/lib.rs
"""
from typing import Set
import numpy as np

# pylint: disable=unused-argument
# pylint: disable=unnecessary-ellipsis

def search_next_best(
    f_org: np.ndarray,
    f_err: np.ndarray,
    cutoff_index: int,
    use_mean_error: bool,
    ic_string: str,
    max_bases: int|None,
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
    ic_string : str
        The information criterion to use. Options: BIC, AIC
    max_bases : int
        The maximum number of bases to use. If None, rely on stopping criterion.

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
) -> np.ndarray:
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
def bin_image(
    image: np.ndarray,
    nwl: int,
    ntime: int,
    power: int
):
    """
    Reduce the size of an image using a 2D window. The value of each pixel is
    computing using a generic mean with a power specified by `power`.
    
    Parameters
    ----------
    image : np.ndarray (nwl, ntime)
        The image to be binned. Strictly must be 2D and have dtype np.float64.
        For a more forgiving interface, use `vpie.bin_image`
        
    nwl : int
        The window size along the wavelength axis.
    ntime : int
        The window size along the time axis.
    power : int
        The power to use in the mean. Use 1 for a linear mean, 2 for a quadratic mean,
        -1 for inverses, etc.

    Returns
    -------
    np.ndarray (new_nwl, new_ntime)
        The binned image
    """
    ...
def fold_image(
    image: np.ndarray,
    stride: int,
    power: int
):
    """
    Phase fold an image using a 1D time stride. The value of each pixel is
    computing using a generic mean with a power specified by `power`.
    
    Parameters
    ----------
    image : np.ndarray (nwl, ntime)
        The image to be folded. Strictly must be 2D and have dtype np.float64.
        For a more forgiving interface, use `vpie.fold_image`
    stride : int
        The fold period in pixels.
    power : int
        The power to use in the mean. Use 1 for a linear mean, 2 for a quadratic mean,
        -1 for inverses, etc.

    Returns
    -------
    np.ndarray (nwl, new_ntime)
        The folded image
    """
    ...
