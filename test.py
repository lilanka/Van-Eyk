#!/usr/bin/env python3

from vaneyk.tensor import Tensor
from vaneyk.engine import Function
import numpy as np

x1 = np.random.rand(1, 5)
x2 = np.random.rand(1, 5)
x3 = np.random.rand(5, 10)
x = Tensor(x1)
y = Tensor(x3)
z = Tensor(5)
s = Function()
f = s.apply('Mul', x, z)
#print(f.shape)
#print(x.shape, y.shape, x.type, y.type)
#print(f.shape, f.type, f.parent.shape)
#print(y.backward().shape, y.shape)
x.deepwalk()
print(x.backward())
#x.backward()
#print(x + y)
"""
import torch
x1_t = torch.Tensor(x1)
x2_t = torch.Tensor(x2)
x3_t = torch.Tensor(x3)
#print(torch.mm(x1_t, x3_t))
print(x1_t * 5)
"""
