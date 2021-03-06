#!/usr/bin/env python3

import numpy as np
from vaneyk.tensor import Tensor

a = Tensor(np.array([2, 3]), requires_grad=True)
b = Tensor(np.array([6, 4]), requires_grad=True) 
Q = a + b

print(Q)
Q.backward()
#print(a.grad)
