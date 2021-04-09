#!/usr/bin/env python3

import numpy as np

from VanEyk.nn import nn
from VanEyk.functions import F

class sirenNN:
  def __init__(self, input_size, l1_size, l2_size, output_size):
    super(sirenNN, self).__init__()

    self.layer1 = nn(input_size, l1_size)
    self.layer2 = nn(l1_size, l2_size)
    self.layer3 = nn(l2_size, output_size)

    self.f = F()

    """
    initialize the parameters of nn
    """
  def forward(self, in_tensor):

    o_l1 = self.layer1.forward(in_tensor)
    x1 = self.f.Sine(o_l1) 
    o_l2 = self.layer2.forward(x1)
    x2 = self.f.Sine(o_l2)
    o_l3 = self.layer3.forward(x2)
    x3 = self.f.Sine(o_l3)
    return x3

in_tensor = np.random.rand(100, 10)
p = np.zeros((100, 10))
x = sirenNN(10, 20, 30, 5)
y = x.forward(p)
print(y)
