#include "linesegment.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("encodels", &encodels, "Encoding line segments to maps");
}