from typing import Callable, Union, Tuple
import numpy as np
from itertools import product

class WildDrawFunctionException(Exception):
    pass

def rademacher(n: int) -> np.ndarray:
    return np.random.choice([-1,1],size=n, replace=True)

def mammen(n: int) -> np.ndarray:
    return np.random.choice(
        a= np.array([-1, 1]) * (np.sqrt(5) + np.array([-1, 1])) / 2, #TODO: #10 Should this divide the whole expression by 2 or just the second part
        size=n,
        replace=True,
        p = (np.sqrt(5) + np.array([1, -1])) / (2 * np.sqrt(5))
    )
    
def norm(n):
    return np.random.normal(size=n)

def webb(n):
    return np.random.choice(
        a = np.concatenate([-np.sqrt(np.array([3,2,1]) / 2), np.sqrt(np.array([1,2,3]) / 2)]),
        replace=True,
        size=n
    )
    
wild_draw_fun_dict = {
    'rademacher' : rademacher,
    'mammen' : mammen,
    'norm' : norm,
    'webb' : webb
}

  
def draw_weights(t : Union[str, Callable], full_enumeration: bool, N_G_bootcluster: int, boot_iter: int) -> Tuple[np.ndarray, int]:
    """draw bootstrap weights
    Args:
        t (str|callable): the type of the weights distribution. Either 'rademacher', 'mammen', 'norm' or 'webb'
        If `t` is a callable, must be a function of one variable, `n`, and return a vector of size `n`
        full_enumeration (bool): should deterministic full enumeration be employed
        N_G_bootcluster (int): the number of bootstrap clusters
        boot_iter (int): the number of bootstrap iterations
    Returns:
        Tuple[np.ndarray, int]: a matrix of dimension N_G_bootcluster x (boot_iter + 1) and the number of iterations
    """    
    
    #TODO: we can use the `case` feature in python, but that's only available in 3.10+ will do a 3.7 version for now
    # Will take out this and make separate functions for readability
    
    if isinstance(t, str):
        wild_draw_fun = wild_draw_fun_dict.get(t)
        if wild_draw_fun is None:
            raise WildDrawFunctionException("Function type specified is not supported or there is a typo.")
    elif callable(t):
        wild_draw_fun = t
    elif t is None:
        raise WildDrawFunctionException("`t` must be specified")
    else:
        raise ValueError(f"t can be string or callable, but got {type(t)}")
    # do full enumeration for rademacher weights if bootstrap iterations
    # B exceed number of possible permutations else random sampling

    # full_enumeration only for rademacher weights (set earlier)
    if full_enumeration: 
        t = 0 # what is this needed for? 
        # with N_G_bootcluster draws, get all combinations of [-1,1] WITH 
        # replacement, in matrix form
        v0 = np.transpose(np.array(list(product([-1,1], repeat=N_G_bootcluster))))
    else:
        # else: just draw with replacement - by chance, some permutations
        # might occur more than once
        v0 = wild_draw_fun(n = N_G_bootcluster * boot_iter)
        v0 = v0.reshape(N_G_bootcluster, boot_iter) # weights matrix
    
    # update boot_iter (B) - only relevant in enumeration case
    boot_iter = v0.shape[1] 
    #v = np.insert(v0, 0, 1,axis = 1)

    return v0, boot_iter