import numpy as np

from vaneyk.grad import grad


def backprop(self, params, method="Adam"):

  


class MSE:
  def __init__(self, in_tensor, out_tensor):
    self.loss = np.square(np.substract(in_tensor - out_tensor))
    
