import numpy as np

from vaneyk.engine import Adiff 
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

    self.parents, self.ctx, self._grad = None, None, None

    self.A = Adiff()

  # drawing the graph. Inspired by https://github.com/geohot/tinygrad
  def deepwalk(self):
    def _deepwalk(node, visited, nodes):
      visited.add(node)
      if node.ctx:
        [_deepwalk(i, visited, nodes) for i in node.parents if i not in visited]
        nodes.append(node)
      return nodes
    return _deepwalk(self, set(), [])

  # Backpropagation
  def backward(self):
    if not self.requires_grad or self.is_parameter:
      raise ValueError(f"reqires_grad={self.requires_grad}. Can't generate gradient")

    self._grad = Tensor(np.ones(self.data.shape), requires_grad=False)

    for node in reversed(self.deepwalk()):
      assert (node._grad is not None)
      grads = node.ctx.backward(node.ctx, node._grad) 
      if len(node.parents) == 1:
        grads = [grads]
      for t, g in zip(node.parents, grads):
        if g is not None:
          assert g.shape == t.shape, \
              "grad shape must match tensor shape"
          gt = Tensor(g, requires_grad=False, is_leaf=False)
          t._grad = gt if t._grad is None else (t._grad + gt)

  def __sub__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return self.A.apply('Sub', self, other)

  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return self.A.apply('Add', self, other.data)

  def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return self.A.apply('Mul', self, other)

  def __truediv__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return self.A.apply('Div', self, other)

  def __pow__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return self.A.apply('Pow', self, other)

  @property
  def dot(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return self.A.apply('Dot', self, other.data)

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
