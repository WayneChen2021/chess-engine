#include "main.hpp"

template Params<tensorflow::Tensor> setup(json &data);
template Params<at::Tensor> setup(json &data);

template <>
tensorflow::SavedModelBundleLite assign_model<tensorflow::SavedModelBundleLite>(std::string &dir)
{
  using namespace tensorflow;
  SavedModelBundleLite model_bundle;
  SessionOptions session_options = SessionOptions();
  RunOptions run_options = RunOptions();
  LoadSavedModel(session_options, run_options, dir, {kSavedModelTagServe}, &model_bundle);
  return model_bundle;
}
template <>
torch::jit::script::Module assign_model<torch::jit::script::Module>(std::string &dir)
{
  return torch::jit::load(dir);
}

template win_totals simulate_games<true, tensorflow::SavedModelBundleLite, tensorflow::Tensor>(Params<tensorflow::Tensor> &params, uint64_t games, uint64_t sims, std::string &white_dir, std::string &black_dir);
template win_totals simulate_games<true, torch::jit::script::Module, at::Tensor>(Params<at::Tensor> &params, uint64_t games, uint64_t sims, std::string &white_dir, std::string &black_dir);
template win_totals simulate_games<false, tensorflow::SavedModelBundleLite, tensorflow::Tensor>(Params<tensorflow::Tensor> &params, uint64_t games, uint64_t sims, std::string &white_dir, std::string &black_dir);
template win_totals simulate_games<false, torch::jit::script::Module, at::Tensor>(Params<at::Tensor> &params, uint64_t games, uint64_t sims, std::string &white_dir, std::string &black_dir);

template <>
tensorflow::Tensor set_output(MCST<tensorflow::Tensor> &node)
{
  auto arr = zero_tensor<tensorflow::Tensor>(Constants::in_total_planes);
  auto tensor = arr.tensor<double, 2>();
  for (MCST<tensorflow::Tensor> *child : node.children)
    tensor(SquareOf(child->moved_from), child->output_ind) = child->probability;
  return arr;
}

template <>
at::Tensor set_output(MCST<at::Tensor> &node)
{
  auto arr = at::zeros({Constants::in_total_planes, 64});
  auto tensor = arr.accessor<double, 2>();
  for (MCST<at::Tensor> *child : node.children)
    tensor[child->output_ind][SquareOf(child->moved_from)] = child->probability;
  return arr;
}

template void start<tensorflow::SavedModelBundleLite, tensorflow::Tensor>(json &);
template void start<torch::jit::script::Module, at::Tensor>(json &);

int main()
{
  PyObject *pymodule = PyImport_ImportModule("models/initialize");
  PyObject *pyfunc = PyObject_GetAttrString(pymodule, "initialize");
  PyObject_CallObject(pyfunc, NULL);

  std::ifstream f("config.json");
  json data = json::parse(f);

  if (data["model"]["tf"])
    start<tensorflow::SavedModelBundleLite, tensorflow::Tensor>(data);
  else
    start<torch::jit::script::Module, at::Tensor>(data);

  return 0;
}