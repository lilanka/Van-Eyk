import numpy as np


class F: 
  """
  Contain activation functions
  """
  def __init__(self):
    pass

  def sine(self, in_tensor):
    return np.sin(in_tensor)

  def relu(self, in_tensor):
    return self._assign(in_tensor) 

  def leaky_relu(self, in_tensor, a=0.01):
    return self._assign(in_tensor, a)

  def prelu(self, in_tensor, a):
    return self._assign(in_tensor, a)

  def nodeOp(self, in_tensor, weights, bias):
    """
    linear calculation in nn
    -> WX + b
    """
    return np.add(np.dot(in_tensor, weights), bias)

  def _assign(self, in_tensor, a=0):

    for vec in in_tensor:
      for j in range(len(vec)):
        vec[j] = max(0, vec[j]) + a*min(0, vec[j])
    return in_tensor 
