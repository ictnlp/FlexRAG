#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>

namespace py = pybind11;

bool contains_any(const std::vector<std::string>& evidences, const std::vector<std::string>& retrieved) {
    for (const auto& evd : evidences) {
        for (const auto& ret : retrieved) {
            if (ret.find(evd) != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

PYBIND11_MODULE(lib_rel, m) {
    m.def("contains_any", &contains_any, "Check if any evidence is in the retrieved strings");
}
