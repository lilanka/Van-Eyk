import numpy as np

class F: 
  """
  Contain activation functions
  """
  def __init__(self):
    pass

  def Sine(self, in_tensor):
    """
    Sine functions 
    """
    return np.sin(in_tensor)

  def linear(self, in_tensor, weights, bias):
    """
    linear calculation in nn
    -> WX + b
    """
    return np.add(np.dot(in_tensor, weights), bias)
