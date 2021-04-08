import numpy as np

from VanEyk.modules.nn import nn

class Network:
  def __init__(self, nodes):
    self.net = nn()

    # setting up layers  
    self.layers = [] 

    for i in nodes[:-1]:
      self.layers.append(self.net.Layer(nodes[i], nodes[i+1]))

nodes = np.array([10, 20, 20, 30])
x = Network(nodes)

