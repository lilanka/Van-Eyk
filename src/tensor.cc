#include <iostream>
#include <vector>

#include "tensor.h"
#include "distributions.h"

using namespace std;

Tensor::Tensor(int d1) {
  vector<int> sizes {d1};
}

Tensor::Tensor(int d1, int d2) {
  vector<int> sizes {d1, d2};
}

Tensor::Tensor(int d1, int d2, int d3) {
  vector<int> sizes {d1, d2, d3};
}

Tensor::Tensor(int d1, int d2, int d3, int d4) {
  vector<int> sizes {d1, d2, d3, d4};
}

Tensor::Tensor(int d1, int d2, int d3, int d4, int d5) {
  vector<int> sizes {d1, d2, d3, d4, d5};
}

Tensor::Tensor(int d1, int d2, int d3, int d4, int d5, int d6) {
  vector<int> sizes {d1, d2, d3, d4, d5, d6};
}

void Distributions::zeros(vector<int>& v, int n=0) {

}



auto Tensor::makingTensor() { 
  switch(dim) {
    case 1 : 
       

  //}
//} 

int main() {

  Tensor x(3);
  

  std::cout << "Successfull" << " " << x.makingTensor()[0][1]<< std::endl;
  return 0;
}
