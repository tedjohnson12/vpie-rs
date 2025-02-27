"""
Python wrappers for Rust functions.

Just so language servers can find them.
"""

from typing import Set, Tuple
import logging
import numpy as np
import colorlog
from . import _vpie_rs

handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    "[%(asctime)s] %(log_color)s[%(levelname)s]%(reset)s %(message)s (%(filename)s:%(lineno)d)%(reset)s",
    datefmt="%Y-%m-%d %H:%M:%S",
	reset=True,
	log_colors={
		'DEBUG':    'cyan',
		'INFO':     'green',
		'WARNING':  'yellow',
		'ERROR':    'red',
		'CRITICAL': 'red,bg_white',
	},
	# secondary_log_colors={},
	# style='%'
)

handler.setFormatter(formatter)


# FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
# logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(handler)

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

def get_reconstruction(
    flux: np.ndarray,
    coeffs: np.ndarray,
    s: Set[int],
) -> np.ndarray:
    """
    Reconstruct an observation given the basis coefficients and the basis set :math:`s`.
    
    Essentially selects some set of spectra in ``flux`` and then does a matrix multiplication.
    
    Parameters
    ----------
    flux : np.ndarray
        The observation to be reconstructed.
    coeffs : np.ndarray
        The basis coefficients.
    s : set of int
        The set of bases.
    
    Returns
    -------
    np.ndarray
        The reconstructed observation.
    """
    return _vpie_rs.get_reconstruction(flux, coeffs, s)

def get_vpie(
    f_org: np.ndarray,
    f_err: np.ndarray,
    cutoff_index: int,
    use_mean_error: bool
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