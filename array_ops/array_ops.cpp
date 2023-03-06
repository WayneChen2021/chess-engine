#include "array_ops.hpp"

template struct ModelOutputs<tensorflow::Tensor>;
template struct ModelOutputs<at::Tensor>;

template <>
tensorflow::Tensor zero_tensor(long last_dim)
{
  using namespace std;
  using namespace tensorflow;
  using namespace tensorflow::ops;

  Scope root = Scope::NewRootScope();
  auto shape = tensorflow::TensorShape({64, last_dim});
  Tensor arr = Tensor(tensorflow::DT_FLOAT, shape);
  auto operation = ZerosLike(root, arr);

  std::vector<Tensor> outputs;
  ClientSession session(root);
  TF_CHECK_OK(session.Run({operation}, &outputs));
  return outputs[0];
}
template <>
at::Tensor zero_tensor(long last_dim)
{
  return at::zeros({last_dim, 64});
}

template <>
tensorflow::Tensor concat<tensorflow::Tensor>(tensorflow::Tensor &arr1, tensorflow::Tensor &arr2, tensorflow::Tensor &arr3, tensorflow::Tensor &arr4, tensorflow::Tensor &arr5, tensorflow::Tensor &arr6, tensorflow::Tensor &arr7, tensorflow::Tensor &arr8)
{
  // tensors are shape (64, _)
  using namespace tensorflow;
  using namespace tensorflow::ops;
  Scope root = Scope::NewRootScope();
  auto concat = Concat(root, {arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8}, 1);
  std::vector<Tensor> outputs;
  ClientSession session(root);
  TF_CHECK_OK(session.Run({concat}, &outputs));
  return outputs[0];
}
template <>
at::Tensor concat<at::Tensor>(at::Tensor &arr1, at::Tensor &arr2, at::Tensor &arr3, at::Tensor &arr4, at::Tensor &arr5, at::Tensor &arr6, at::Tensor &arr7, at::Tensor &arr8)
{
  // tensors are shaped (_, 64)
  return torch::cat({arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8});
}

template <>
ModelOutputs<tensorflow::Tensor> call_model<tensorflow::SavedModelBundleLite, tensorflow::Tensor>(tensorflow::SavedModelBundleLite &model, tensorflow::Tensor &arr)
{
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_data = {{"board_input", arr}};
  std::vector<std::string> output_nodes = {{"policy", "value"}};
  std::vector<tensorflow::Tensor> predictions;
  model.GetSession()->Run(inputs_data, output_nodes, {}, &predictions);

  auto value = predictions[1].tensor<double, 1>();
  return ModelOutputs<tensorflow::Tensor>{value(0), predictions[0]};
}
template <>
ModelOutputs<at::Tensor> call_model<torch::jit::script::Module, at::Tensor>(torch::jit::script::Module &model, at::Tensor &arr)
{
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(arr);
  auto outputs = model.forward(inputs).toTuple();
  auto output_elements = outputs->elements();

  auto value = output_elements[1].toTensor().accessor<double, 1>();
  return ModelOutputs<at::Tensor>{value[0], output_elements[0].toTensor()};
}