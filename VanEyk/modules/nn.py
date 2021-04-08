import numpy as np

class nn(object):
  def __init__(self):
    pass

  def Sine(self, X):
    """
    sine activation function := return tensor of sine values according to tensor x 
    """
    return np.sine(X)

  def Layer(self, n1, n2, bias=False):
    """
    Setting up a layer
    inputs := X(n1, 1)
    nodes  := Y(n2, 1)
    weight := W(n2, n1) matrix
    bias   := B(n2, 1)

    Y = X.W + B(0) -> activation function
    """
    self.weights = np.zeros((n1, n2)) 
    if bias:
      self.bias = np.zeros((1, n2))

y = nn()
x = y.Layer(2, 5, True)
print(id(y.weights))
y.weights[:] = np.array([[2, 3, 4, 5, 6], [1, 2, 4, 5, 6]])
print(id(y.weights))
