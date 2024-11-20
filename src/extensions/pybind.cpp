#include <torch/extension.h>
#include "ops.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def(
        "add",
        &add_gpu,
        "custom addition for bfp kmeans"
    );
}
