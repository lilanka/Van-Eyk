#include <boost/config/assert_cxx17.hpp>
#include <boost/python.hpp>

char const* greet() {
  return "hello, lila";
}

BOOST_PYTHON_MODULE(hello_so) {
  using namespace boost::python;
  def("greet", greet);
}
