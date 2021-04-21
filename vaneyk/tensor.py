import numpy as np

import vaneyk.engine
from vaneyk.nn.functional import F 

class Tensor:
  """
  if requires_grad=True and is_leaf=True  -> AccumulateGrad node. This node is a gradient enabled node that has no parent.
                                             .apply() haddles accumulation gradients in the tensor's .grad
  if is_leaf=False and requires_grad=True -> BackwardFunction an intermediate node, gradients are calculated and passed. not stored
                                             .apply() calculate the gradient w.r.t the inputs and returns them
  is_leaf=True and requires_grad=False    -> Constant. User created tensor with 
  is_parameter=True                       -> tensor contains parameters of the network.
                                             if is_parameter =True -> requires_grad, is_leaf == True
  Function.apply()                        -> creating and storing the nodes in (Tensor.grad_fn) 

  """ 
  __counter = 0
  def __init__(self, data, requires_grad=True, is_leaf=False, is_parameter=False):
    if not requires_grad and not is_leaf: 
      raise ValueError('Non leaf nodes should be require_grad=True') 
    self.id = Tensor.__counter
    Tensor.__counter += 1

    self.data = data

    self.requires_grad = requires_grad
    self.is_parameter = is_parameter
    self.is_leaf = is_leaf

    self.parent = []
    self._grad = 0
    self.grad_fn = {'Op': ()} 

    self.f = F()

  def backward(self):
    if not self.requires_grad or self.is_parameter:
      raise ValueError(f"reqires_grad={self.requires_grad}. Can't generate gradient")

    # kicks off the DFS
    engine.backward(Tensor(np.full_like(self.data, 1)), )

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
    return f"Tensor({self.data}, reqires_grad={self.requires_grad})"

  @property
  def type(self):
    return f"eyk tensor"

  @property
  def shape(self):
    return self.data.shape

  @property
  def grad(self):
    return self._grad
