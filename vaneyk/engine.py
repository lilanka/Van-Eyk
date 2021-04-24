# ADiff engine.
# PyTorch has Automatic differentiation algorithm Autograd. we have ADiff.

import numpy as np
from vaneyk.nn.functional import * 

class Adiff:
  def __init__(self):
    pass

  def apply(self, op, a, b):
    from vaneyk.tensor import Tensor
     
    _op = {
        'Add' :  Add(),
        'Sub' :  Sub(), 
        'Div' :  Div(),
        'Mul' :  Mul(),    
        'Dot' :  Dot(), 
        'Pow' :  Pow(),
    }

    _out = Tensor(_op[op].forward(a.data, b.data), requires_grad=True if (a.requires_grad or b.requires_grad) else False)
    _out.parents = [a, b]
    _out.ctx =  _op[op]  

    return _out
