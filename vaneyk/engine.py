# ADiff engine.
# PyTorch has Automatic differentiation algorithm Autograd. we have ADiff.

import numpy as np

def backward(self): 
  pass

class Function:
  def __init__(self):
    pass

  def apply(self, op, x1, x2):

    _tensor = {
        0 : lambda x1, x2: x1 + x2,
        1 : lambda x1, x2: x1 - x2, 
        2 : lambda x1, x2: x1 / x2,
        3 : lambda x1, x2: np.dot(x1, x2) 
    }.get(op, 4)(x1, x2)
    return _tensor 
