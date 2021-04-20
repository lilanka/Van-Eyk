# ADiff engine.
# PyTorch has Automatic differentiation algorithm Autograd. we have ADiff.

import numpy as np
from nn.functional import F

def backward(): 
  pass

class Function:
  def __init__(self):
    pass

  def apply(self, op, x1, x2):
    import tensor 

    _out = {
        0 : lambda x1, x2 : x1 + x2,
        1 : lambda x1, x2 : x1 - x2, 
        2 : lambda x1, x2 : x1 / x2,
        3 : lambda x1, x2 : x1 * x2,    # x1 :: tensor * x2 :: scalar
        4 : lambda x1, x2 : x1.dot(x2), # dot product
        5 : lambda x1, x2 : ValueError("operation has not identified")
    }.get(op, 5)(x1.data, x2.data)

    req_grad = False
    if x1.requires_grad or x2.requires_grad:
      req_grad = True

    _tensor = tensor.Tensor(_out, requires_grad=req_grad)
   
    _tensor.parent.append(x1)
    _tensor.parent.append(x2)
    print(_tensor)
    return _tensor 
