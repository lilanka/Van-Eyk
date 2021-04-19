import numpy as np

import engine 
from nn.functional import F 

class Tensor:
  __counter = 0  
  def __init__(self, data, requires_grad=True, is_leaf=True, is_parameter=False):
    if not requires_grad and not is_leaf: 
      raise ValueError('Non leaf nodes should be require_grad=True') 
    self.id = Tensor.__counter
    Tensor.__counter += 1

    self.data = data
    self.requires_grad = requires_grad
    
    self.is_leaf = is_leaf
    self.parent = []
    self.grad = 0

    self.f = F()
    #if requires_grad and is_leaf -> AccumulateGrad node. This node is a gradient enabled node that has no parent.
                                    #.apply() haddles accumulation gradients in the tensor's .grad
    #if is_leaf=False and requires_grad=True -> BackwardFunction an intermediate node, gradients are calculated and passed. not stored
                                    #.apply() calculate the gradient w.r.t the inputs and returns them
    # is_leaf=True and requires_grad=False -> Constant. User created tensor with 
    # Function.apply() -> creating and storing the nodes in (Tensor.grad_fn) 

    # is_parameter -> tensor contains parameters of the network.
                      #if is_parameter =True -> requires_grad, is_leaf == True
    
  def backward(self, other):
    if not self.requires_grad or is_parameter:
      raise ValueError(f"reqires_grad={self.requires_grad}. Can't generate gradient")
    engine.backward(Tensor(np.ones(self.data.shape)))

  def __sub__(self, other):
    Tensor(other)
    return self.f.Sub(self.data, other.data) 

  def __add__(self, other):
    Tensor(other)
    return self.f.Add(self.data, other.data) 

  def __matmul__(self, other):
    Tensor(other)
    return self.f.Mul(self.data, other.data) 

  def __truediv__(self, num):
    return self.f.Div(self.data, num) 

  def shape(self):
    return self.tensor.shape

  def __repr__(self):
    return f"tensor({self.data}, reqires_grad={self.requires_grad})"
x1 = np.random.rand(1, 10) 
x2 = np.random.rand(1, 10)
x3 = np.random.rand(10, 100)
x = Tensor(x1)
y = Tensor(x2)
s = engine.Function()
print(x)
print(s.apply(3, x2, x3))
