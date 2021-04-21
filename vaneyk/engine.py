# ADiff engine.
# PyTorch has Automatic differentiation algorithm Autograd. we have ADiff.

import numpy as np


def backward():
  """
  Performs the DFS
  Args:
    gradient of the final node w.r.t our output :: use this to calculate own gradients
    grad_fn :: contains the current node object
               node has method .apply() which contains instructions on what to do with the given gradient
  """
  pass 

class Function:
  def __init__(self):
    pass

  def apply(self, op, x1, x2):
    from vaneyk.tensor import Tensor # Otherwise causes circular import error 

    _out = {
        0 : lambda x1, x2 : x1 + x2,
        1 : lambda x1, x2 : x1 - x2, 
        2 : lambda x1, x2 : x1 / x2,
        3 : lambda x1, x2 : x1 * x2,    
        4 : lambda x1, x2 : x1.dot(x2), # dot product
        5 : lambda x1, x2 : ValueError("operation has not identified")
    }.get(op, 5)(x1.data, x2.data)

    req_grad = False
    if x1.requires_grad or x2.requires_grad:
      req_grad = True

    _tensor = Tensor(_out, requires_grad=req_grad)

    _tensor.grad_fn['Op'] = op  
    _tensor.parent.append(x1)
    _tensor.parent.append(x2)

    print(_tensor)
    return _tensor 

class AccumulateGrad():
  pass

class ContextManager():
  pass

class BackwardFunction():
  """
  An intermediate node. Gradients are calculated and passed. 
  Storing?. well, we don't do that in here.
  """
  def __init__(self):
    pass

  def apply(self, grad):
    pass
