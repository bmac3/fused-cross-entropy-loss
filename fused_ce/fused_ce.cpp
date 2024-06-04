#include <pybind11/pybind11.h>


int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;


PYBIND11_MODULE(c_ext, m) {
    m.def("add", &add, "A function that adds two numbers");
}
