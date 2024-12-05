#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>

namespace py = pybind11;

std::vector<std::vector<bool>> get_contain_map(const std::vector<std::string>& evs, const std::vector<std::string>& rets) {
    std::vector<std::vector<bool>> results;
    results.reserve(rets.size());

    for (const auto& ret : rets) {
        std::vector<bool> result_row;
        result_row.reserve(evs.size());
        for (const auto& ev : evs) {
            result_row.push_back(ret.find(ev) != std::string::npos);
        }
        results.push_back(std::move(result_row));
    }

    return results;
}

PYBIND11_MODULE(lib_rel, m) {
    m.def("get_contain_map", &get_contain_map, "Get contain map.");
}
