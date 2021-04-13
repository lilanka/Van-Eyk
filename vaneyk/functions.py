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

    for vec in in_tensor:
      for j in range(len(vec)):
        vec[j] = max(0, vec[j])
    return in_tensor 

  def leaky_relu(self, in_tensor, a=0.01):

    for vec in in_tensor:
      for j in range(len(vec)):
        vec[j] = max(0, vec[j]) + a*min(0, vec[j])
    return in_tensor 

  def prelu(self, in_tensor, a):

    for vec in in_tensor:
      for j in range(len(vec)):
        vec[j] = max(0, vec[j]) + min(0, vec[j])
    return in_tensor 

  def nodeOp(self, in_tensor, weights, bias):
    """
    linear calculation in nn
    -> WX + b
    """
    return np.add(np.dot(in_tensor, weights), bias)
