#include <utility>

#include <tensorflow/core/framework/tensor.h>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/ops.h"
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <torch/script.h>

#include "../movegenerator/Movereceiver.cpp"

template <typename input_type>
struct ModelOutputs
{
  double eval;
  input_type policy;
};

template <typename arr_type>
arr_type zero_tensor(long last_dim);

template <typename input_type>
input_type concat(input_type &arr1, input_type &arr2, input_type &arr3, input_type &arr4, input_type &arr5, input_type &arr6, input_type &arr7, input_type &arr8);

template <typename model_type, typename input_type>
ModelOutputs<input_type> call_model(model_type &model, input_type &arr);

template <bool create_new>
void assign_ind(map pos, std::unique_ptr<tensorflow::Tensor> &arr, uint64_t size, uint64_t ind)
{
  if constexpr (create_new)
    arr = &zero_tensor(size);
  auto tensor = (*arr).tensor<double, 2>();
  Bitloop(pos)
  {
    tensor(SquareOf(pos), ind) = 1.;
  }
}
template <bool create_new>
void assign_ind(map pos, std::unique_ptr<at::Tensor> &arr, uint64_t size, uint64_t ind)
{
  if constexpr (create_new)
    arr = &at::zeros({size, 64});
  auto tensor = (*arr).accessor<double, 2>();
  Bitloop(pos)
  {
    tensor[ind][SquareOf(pos)] = 1.;
  }
}

void fill_dim(double val, std::unique_ptr<tensorflow::Tensor> &arr, uint64_t ind, long last_dim)
{
  if (arr == nullptr)
    *arr = zero_tensor<tensorflow::Tensor>(last_dim);

  auto tensor = (*arr).tensor<double, 2>();
  for (uint64_t c = 0; c < 64; ++c)
  {
    tensor(c, ind) = val;
  }
}
void fill_dim(double val, std::unique_ptr<at::Tensor> &arr, uint64_t ind, long last_dim)
{
  if (arr == nullptr)
    *arr = at::zeros({last_dim, 64});

  auto tensor = (*arr).accessor<double, 2>();
  for (uint64_t c = 0; c < 64; ++c)
  {
    tensor[ind][c] = val;
  }
}