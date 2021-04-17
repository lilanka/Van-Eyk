# ADiff engine.
# PyTorch has Automatic differentiation algorithm Autograd. we have ADiff.

import numpy as np
from nn.functional import F

F = F()
operations = {
    "Add" : F.Add,
    "Sub" : F.Sub,
    "Div" : F.Div,
    "Mul" : F.Mul,
    }

nodes = [] 

def backward(self): 
  pass

class Function:
  """Base class for functions in functional.py"""
  def __init__(self):
    pass

  def apply(self, op, x1, x2):
    """
    X    :: tuples 
    X[0] :: np.ndarray -> tensor value
    X[1] :: dict       -> {op of the tensor, its parents} 
    """
    node_data = {'Tensor': (), 'Op': (), 'Parent': ()}
    node_data['Op'] = op
    node_data['Parent'] = (x1['Tensor'], x2['Tensor']) 

    node_data['Tensor'] = operations[op](x1['Tensor'], x2['Tensor'])     
    nodes.append(node_data)
    return node_data 
