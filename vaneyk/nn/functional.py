"""
Contains the forward/backward behaviour of operations in the computational graph.
Any operation that effects the network's final loss should have an implementation here.
"""
import numpy as np

class F:
  def __init__(self):
    pass

  def Add(self, x1, x2):
    if x1.shape != x2.shape:
      raise ValueError(f"operands could not be broadcast together with shape()s {x1.shape()} {x2.shape()}") 
    return np.add(x1, x2)

  def Sub(self, x1, x2):
    if x1.shape != x2.shape:
      raise ValueError(f"operands could not be broadcast together with shape()s {x1.shape()} {x2.shape()}") 
    return np.subtract(x1, x2)

  def Mul(self, x1, x2):
    if x1.shape[1] != x2.shape[0]:
      raise ValueError(f"shape()s {x1.shape()} and {x2.shape()} not aligned: {x1.shape()[1]} (dim 1) != {x2.shape()[0]} (dim 0)") 
    return np.matmul(x1, x2)

  def Div(self, x, num):
    if num == 0:
      raise ValueError("division by zero")
    return np.true_divide(x, num)

  def Transpose(self, x):
    return np.transpose(x)

  def Reshape(self, x, new_shape):
    # shape() should be (<shape()>)
    return np.reshape(x, new_shape)

  def Log(self):
    """Element-wise log of a tensor"""
    pass

  def cross_entropy(self):
    """Cross-entropy loss (loss function in comp graph"""
    pass
