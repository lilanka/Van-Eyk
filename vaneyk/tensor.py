import numpy as np

from nn.functional import *
import engine 

class Tensor():
  """Wrapper for np.array.
  Allows interaction with VanEyk
  """
  def __init__(self, data: np.array, requires_grad: bool=False, is_leaf:bool=True, is_parameter:bool=False):
    self.tensor = np.array(data)
    self.requires_grad = requires_grad
    self.tensor_info = (np.array(data), f"reqires_grad={requires_grad}")

    #if requires_grad and is_leaf -> AccumulateGrad node. This node is a gradient enabled node that has no parent.
                                    #.apply() haddles accumulation gradients in the tensor's .grad
    #if is_leaf=False and requires_grad=True -> BackwardFunction an intermediate node, gradients are calculated and passed. not stored
                                    #.apply() calculate the gradient w.r.t the inputs and returns them
    # is_leaf=True and requires_grad=False -> Constant. User created tensor with 
    # Function.apply() -> creating and storing the nodes in (Tensor.grad_fn) 

    # is_parameter -> tensor contains parameters of the network.
                      #if is_parameter =True -> requires_grad, is_leaf == True
    
  def backward(self):
    if self.requires_grad == False:
      raise ValueError(f"reqires_grad={self.requires_grad}. Can't generate gradient")

    _x = engine.backward(np.ones(self.tensor.shape))

  def __sub__(self, x:np.array, y:np.array) -> np.array:
    return Sub(x, y)

  def __add__(self, x:np.array, y:np.array) -> np.array:
    return Add(x, y)

  def __mul__(self, x:np.array, y:np.array) -> np.array:
    return Mul(x, y)

  def __truediv__(self, x:np.array, num) -> np.array:
    return Div(x, num)

# *************** for testing
x = Tensor(np.zeros((3, 4)))
print(x.tensor)
print(x.tensor_info)
print(x.backward())
print(np.random.rand(2, 3) - np.random.rand(2, 3))
