import numpy as np

class Add():
  @staticmethod
  def forward(x1, x2):
    if x1.shape != x2.shape:
      raise ValueError(\
          f"operands could not be broadcast together with shape()s {x1.shape()} {x2.shape()}") 
    return np.add(x1, x2)

  @staticmethod
  def backward(ctx, out_grad):
    pass

class Sub():
  @staticmethod
  def forward(x1, x2):
    if x1.shape != x2.shape:
      raise ValueError(\
          f"operands could not be broadcast together with shape()s {x1.shape()} {x2.shape()}") 
    return np.subtract(x1, x2)

  @staticmethod
  def backward(ctx, grad):
    pass

class Mul():
  @staticmethod
  def forward(x1, x2):
    return x1 * x2
  @staticmethod
  def backward():
    pass

class Div():
  @staticmethod
  def forward(x, num):
    if num == 0:
      raise ValueError("division by zero")
    return np.true_divide(x, num)
  @staticmethod
  def backward(dy):
    pass

class Pow():
  @staticmethod
  def forward(x, num):
    return x ** num 

  @staticmethod
  def backward():
    pass

class Dot():
  @staticmethod
  def forward(x1, x2):
    return np.dot(x1, x2)

  @staticmethod
  def backward():
    pass
