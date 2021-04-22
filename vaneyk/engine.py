# ADiff engine.
# PyTorch has Automatic differentiation algorithm Autograd. we have ADiff.

import numpy as np

class Function:
  def __init__(self):
    pass

  def apply(self, op, x1, x2):
    from vaneyk.tensor import Tensor # Otherwise causes circular import error 

    _out = {
        'Add' : lambda x1, x2 : x1 + x2,
        'Sub' : lambda x1, x2 : x1 - x2, 
        'Div' : lambda x1, x2 : x1 / x2,
        'Mul' : lambda x1, x2 : x1 * x2,    
        'Dot' : lambda x1, x2 : x1.dot(x2), # dot product
        0     : lambda x1, x2 : ValueError("operation has not identified")
    }.get(op, 0)(x1.data, x2.data)

    req_grad = False
    if x1.requires_grad or x2.requires_grad:
      req_grad = True

    _tensor = Tensor(_out, requires_grad=req_grad)
    _tensor.parents = [x1, x2]
    _tensor.ctx = op
    return _tensor 
