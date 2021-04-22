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

    self.id = Tensor.__counter
    Tensor.__counter += 1

    self.data = data

    self.requires_grad = requires_grad
    self.is_parameter = is_parameter
    self.is_leaf = is_leaf

    self.parents = None 
    self.ctx = None
    self._grad = None

    self.f = F()

  # drawing the graph. Inspired by https://github.com/geohot/tinygrad
  def deepwalk(self):
    def _deepwalk(node, visited, nodes):
      visited.add(node)
      if node.parents:
        [_deepwald(i, visited, nodes) for i in node.parents if i not in visited]
        nodes.append(node)
      return nodes
    return _deepwalk(self, set(), [])

  # Backpropagation
  def backward(self):
    if not self.requires_grad or self.is_parameter:
      raise ValueError(f"reqires_grad={self.requires_grad}. Can't generate gradient")

    self._grad = Tensor(np.ones(self.data.shape), requires_grad=False, is_leaf=False)

    for node in reversed(self.deepwalk()):
      assert (node._grad is not None)
      grads = node.ctx.backward(node.ctx, node.grad.data)
      if len(node.parents) == 1:
        grads = [grads]
      for t, g in zip(node.parents, grads):
        if g is not None:
          assert g.shape == t.shape, \
              "grad shape must match tensor shape"
          gt = Tensor(g, requires_grad=False, is_leaf=False)
          t.grad = gt if t.grad is None else (t.grad + gt)

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
