"""
I would like this all to be in Rust someday. I don't know enough about nested sampling though.
"""
from pathlib import Path
from time import time
from typing import Callable, Tuple, Set
import numpy as np
import matplotlib.pyplot as plt

import dynesty

from . wrappers import get_reconstruction, get_vpie



class Parameter:
    """
    A grid axis parameter
    
    This defines part of the domain of our problem.
    
    Parameters
    ----------
    name : str
        The name of the parameter
    prior : Callable
        The prior distribution
    values : np.ndarray or None
        The values of the parameter in the grid
    """
    def __init__(
        self,
        name: str,
        prior: Callable,
        truth: float = None,
        values: np.ndarray = None,
    ):
        self.name = name
        self.values = values
        self.truth = truth
        self._prior = prior
    @property
    def is_grid(self):
        return self.values is not None
    def prior(self,u:float | np.ndarray) -> float | np.ndarray:
        """
        Transform random draws using the prior.
        """
        match u:
            case float():
                if u<0 or u>=1:
                    raise ValueError('u must be on the interval [0,1)')
            case np.ndarray():
                if np.any(u<0) or np.any(u>=1):
                    raise ValueError('u must be on the interval [0,1)')
        return self._prior(u)

class Prior:
    def __init__(
        self,
        forward: Callable,
        inverse: Callable
    ):
        self._forward = forward
        self._inverse = inverse
    
    def __call__(self, u):
        return self._forward(u)
    
    def inverse(self, x):
        """
        Inverse transform
        
        Map x to u where u is on the interval [0,1]
        """
        return self._inverse(x)
    
    @classmethod
    def uniform(cls, a, b):
        """
        Uniform prior on the interval [a,b)
        """
        return cls(
            lambda u: a + (b-a)*u,
            lambda x: (x-a)/(b-a)
        )