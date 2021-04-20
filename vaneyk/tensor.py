import numpy as np

import engine 
from nn.functional import F 

class Tensor:
  __counter = 0  
  def __init__(self, data, requires_grad=True, is_leaf=True, is_parameter=False):
    if not requires_grad and not is_leaf: 
      raise TensorError('Non leaf nodes should be require_grad=True') 
    self.id = Tensor.__counter
    Tensor.__counter += 1

    self.data = data

    self.requires_grad = requires_grad
    self.is_parameter = is_parameter
    self.is_leaf = is_leaf
    self.parent = []
    self._grad = 0
  
    self.f = F()
    #if requires_grad and is_leaf -> AccumulateGrad node. This node is a gradient enabled node that has no parent.
                                    #.apply() haddles accumulation gradients in the tensor's .grad
    #if is_leaf=False and requires_grad=True -> BackwardFunction an intermediate node, gradients are calculated and passed. not stored
                                    #.apply() calculate the gradient w.r.t the inputs and returns them
    # is_leaf=True and requires_grad=False -> Constant. User created tensor with 
    # Function.apply() -> creating and storing the nodes in (Tensor.grad_fn) 

    # is_parameter -> tensor contains parameters of the network.
                      #if is_parameter =True -> requires_grad, is_leaf == True
    
  def backward(self):
    if not self.requires_grad or self.is_parameter:
      raise TensorError(f"reqires_grad={self.requires_grad}. Can't generate gradient")

    # kicks off the DFS
    x = Tensor(np.full_like(self.data, 1))
    engine.backward(Tensor(np.full_like(self.data, 1)))
    return x

  def __sub__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return self.f.Sub(self.data, other.data) 

  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return self.f.Add(self.data, other.data) 

  def __mul__(self, other):
    return self.f.Mul(self.data, other) 

  def __truediv__(self, num):
    return self.f.Div(self.data, num) 
 
  @property
  def dot(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return np.dot(self.data, other.data)

  def __repr__(self):
    return f"Tensor({self.data!r}, reqires_grad={self.requires_grad!r})"

  @property
  def type(self):
    return f"eydtensor"

  @property
  def shape(self):
    return self.data.shape

  @property
  def grad(self):
    return self._grad
