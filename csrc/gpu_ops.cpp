#include <pybind11/pybind11.h>
#include "kernels.h"


namespace py = pybind11;


PYBIND11_MODULE(c_ext, m) {
    m.def("add", &add, "A function that adds two numbers");
}
