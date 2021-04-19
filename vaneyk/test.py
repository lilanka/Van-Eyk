from tensor import Tensor
from engine import Function
import numpy as np

x = Tensor(np.random.rand(3, 4))
xx = Tensor(np.random.rand(3, 4))

l = Function()
print(l.apply("Add", x, xx))
