# ADiff engine.
# PyTorch has Automatic differentiation algorithm Autograd. we have ADiff.

import numpy as np
from vaneyk.nn.functional import F

class Adiff:
  def __init__(self):
    self.F = F()

  def apply(self, op, x1, x2):
    from vaneyk.tensor import Tensor
    
    _out = {
        'Add' : lambda x1, x2 : self.F.Add(x1, x2),
        'Sub' : lambda x1, x2 : self.F.Sub(x1, x2), 
        'Div' : lambda x1, x2 : self.F.Div(x1, x2),
        'Mul' : lambda x1, x2 : self.F.Mul(x1, x2),    
        'Dot' : lambda x1, x2 : self.F.Dot(x1, x2), 
        'Pow' : lambda x1, x2 : self.F.Pow(x1, x2),
        0     : lambda x1, x2 : ValueError("operation has not identified")
    }.get(op, 0)(x1.data, x2.data)
    
    _out = Tensor(_out, requires_grad=True if (x1.requires_grad or x2.requires_grad) else False)
    _out.parents = [x1, x2]
    _out.ctx = op

    return _out
