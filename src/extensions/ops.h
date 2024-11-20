#pragma once
#include <torch/extension.h>

at::Tensor add_gpu(const at::Tensor &a_tensor, const at::Tensor &b_tensor);