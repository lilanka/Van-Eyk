import numpy as np

from vaneyk.grad import grad

class MSE:
  def __init__(self, in_tensor, out_tensor):
    self.loss = np.square(np.substract(in_tensor - out_tensor))

  def backprop(self, method="Adam"):
    """
    Weight update method
    
    just random for testing
    >> loss.backprop(method)
    """ 
    

