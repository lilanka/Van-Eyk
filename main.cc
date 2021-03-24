#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Don't fuck with names in c++, 

int add(int i, int j) {return i + j;}

xt::xarray<double> addd() {
  xt::xarray<double> a = {{1., 2.}, {3., 4.}};
  return a;
}

PYBIND11_MODULE(go, m) {
  m.doc() = "pybind11 example pluging";
  m.def("add", &add, "a function with add two numbers");
}

int main() {
  xt::xarray<double> a = {{1., 2.}, {3., 4.}};
  std::cout << a << std::endl;

  xt::xtensor<double, 2> b = {{1., 2.}, {3., 4.}};
  std::cout << b << std::endl;

  xt::xtensor_fixed<double, xt::xshape<2, 2>> c = {{1., 2.}, {3., 4.}};
  std::cout << c << std::endl;
}
