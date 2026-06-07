"""
Main VPIE function
"""

from typing import Set, Tuple
import numpy as np

# pylint: disable-next=no-name-in-module
from ._vpie_rs import search_next_best, get_coeffs, get_reconstruction, bin_image as _bin_image_rs, fold_image as _fold_image_rs


def get_vpie(
    f_org: np.ndarray,
    f_err: np.ndarray,
    cutoff_index: int,
    use_mean_error: bool,
    ic_string: str,
    max_basis_size: int|None = None
) -> Tuple[Set[int], np.ndarray, np.ndarray]:
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
    ic_string : str
        The information criterion to use. Options: BIC, AIC

    Returns
    -------
    s : set of int
        The set of bases.
    coeffs : np.ndarray
        The basis coefficients.
    f_rec : np.ndarray
        The reconstructed observation.
    """
    s: Set[int] = search_next_best(f_org, f_err, cutoff_index, use_mean_error, ic_string, max_basis_size)
    coeffs: np.ndarray = get_coeffs(
        f_org[:, :cutoff_index], f_err[:, :cutoff_index], s, use_mean_error)
    f_rec: np.ndarray = get_reconstruction(f_org, coeffs, s)
    return s, coeffs, f_rec

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
        The image to be binned.
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
    return _bin_image_rs(np.atleast_2d(image).astype(np.float64), nwl, ntime, power)

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
        The image to be folded.
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
    return _fold_image_rs(np.atleast_2d(image).astype(np.float64), stride, power)
    
    