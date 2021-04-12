import numpy as np

from vaneyk.functions import F

class nn:
  """
  Linear nn
  """
  def __init__(self, in_features: int=0, out_features: int=0, bias: bool = True) -> None:
    super(nn, self).__init__()

    """
    Input dim   -> (1, in_features)
    Weights dim -> (in_features, out_features)  
    Bias dim    -> (1, out_features)
    """
    self.in_features = in_features
    self.out_features = out_features 
    self.weights = np.random.rand(self.in_features, self.out_features) 

    if bias:
      self.bias = np.random.rand(1, self.out_features) # initialize bias using uniform distribution :: random for now
    else:
      self.bias = np.zeros((1, self.out_features))  # initialize bias to zero  

  def forward(self, in_tensor):
    """
    Forward propagation
    """   
    x = F()
    return x.linear(in_tensor, self.weights, self.bias) 
