// Python bindings: import _netsentinel_cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "netsentinel/baseline.hpp"

namespace py = pybind11;

namespace {

py::array_t<double> seasonal_baseline_py(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& values,
    const py::array_t<bool, py::array::c_style | py::array::forcecast>& weekend,
    int readings_per_day, int baseline_days, int fallback_window) {
    if (values.ndim() != 2) throw std::invalid_argument("values must be a 2-D array (readings x KPIs)");
    if (weekend.ndim() != 1) throw std::invalid_argument("weekend must be a 1-D array");
    const auto n = static_cast<std::size_t>(values.shape(0));
    const auto k = static_cast<std::size_t>(values.shape(1));

    std::vector<double> v(values.data(), values.data() + n * k);
    std::vector<bool> w(weekend.data(), weekend.data() + weekend.size());
    const netsentinel::BaselineConfig cfg{readings_per_day, baseline_days, fallback_window};

    std::vector<double> base;
    {
        py::gil_scoped_release release;   // pure C++ from here on
        base = netsentinel::seasonal_baseline(v, n, k, w, cfg);
    }
    py::array_t<double> out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(k)});
    std::copy(base.begin(), base.end(), out.mutable_data());
    return out;
}

}  // namespace

PYBIND11_MODULE(_netsentinel_cpp, m) {
    m.doc() = "NetSentinel C++ kernels";
    m.def("seasonal_baseline", &seasonal_baseline_py,
          py::arg("values"), py::arg("weekend"),
          py::arg("readings_per_day") = 96, py::arg("baseline_days") = 7, py::arg("fallback_window") = 12,
          "Per-cell time-of-day baseline. values: (n, k) float array in time order; weekend: (n,) bool.");
}
