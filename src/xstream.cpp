<%
cfg['compiler_args'] = ['-std=c++11']
cfg['sources'] = ['main.cpp', 'chain.cpp', 'docopt.cpp','MurmurHash3.cpp', 'streamhash.cpp']
cfg['dependencies'] = ['main.h']
setup_pybind11(cfg)
%>

#include "main.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


PYBIND11_MODULE(xstream, m) {

    // PYBIND11_NUMPY_DTYPE(minDensity, depth, features);

    py::class_<xStream>(m, "xStream")
    .def(py::init<>(&xStream::init))
    .def("getScores", &xStream::getScores)
    .def("getMinDensity", &xStream::getMinDensity)
    .def("getFeatureProjectionMap", &xStream::getFeatureProjectionMap)
    .def("fit", &xStream::fit);
}

