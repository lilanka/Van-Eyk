#include <iostream>
#include <vector>

#include "tensor.h"

int main() {
  Tensor<double>(4, 5, 5);
  std::cout << "Successfull" << std::endl;
  return 0;
}
