import numpy as np

class F:
  def Add(self, x1, x2):
    if x1.shape != x2.shape:
      raise ValueError(\
          f"operands could not be broadcast together with shape()s {x1.shape()} {x2.shape()}") 
    def backward(dy):
      pass
    return np.add(x1, x2)

  def Sub(self, x1, x2):
    if x1.shape != x2.shape:
      raise ValueError(\
          f"operands could not be broadcast together with shape()s {x1.shape()} {x2.shape()}") 
    def backward(dy):
      pass
    return np.subtract(x1, x2)

  def Mul(self, x1, x2):
    def backward(dy):
      pass
    return x1 * x2

  def Div(self, x, num):
    if num == 0:
      raise ValueError("division by zero")
    def backward(dy):
      pass
    return np.true_divide(x, num)

  def Transpose(self, x):
    return np.transpose(x)

  def Reshape(self, x, new_shape):
    # shape() should be (<shape()>)
    return np.reshape(x, new_shape)

  def Pow(self, x, num):
    return x ** num 

  def Log(self):
    """Element-wise log of a tensor"""
    pass

  def cross_entropy(self):
    """Cross-entropy loss (loss function in comp graph"""
    pass
