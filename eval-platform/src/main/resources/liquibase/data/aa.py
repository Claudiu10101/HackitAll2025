# Example using Numba
from numba import jit
import numpy as np

@jit(nopython=True)
def fast_sum(a):
    total = 0
    for x in a:
        total += x
    return total

data = np.arange(100000)
# The function will be compiled on the first call
result = fast_sum(data)