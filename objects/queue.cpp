#include "queue.hpp"

template class Queue<tensorflow::Tensor>;
template class Queue<at::Tensor>;

template <>
void Queue<tensorflow::Tensor>::assign_inputs(std::unique_ptr<TrainingExample<tensorflow::Tensor>> &ex, uint64_t ind)
{
  long flat_input_dim = (long)Constants::in_total_planes * 64;
  long flat_output_dim = (long)Constants::out_total_planes * 64;
  long start_inputs = ind * flat_input_dim;
  long start_outputs = ind * flat_output_dim;
  tensorflow::TensorShape input_shape = tensorflow::TensorShape({flat_input_dim});
  tensorflow::TensorShape output_shape = tensorflow::TensorShape({flat_output_dim});

  tensorflow::Tensor flat_input(tensorflow::DT_FLOAT, input_shape);
  tensorflow::Tensor flat_output(tensorflow::DT_FLOAT, output_shape);
  flat_input.CopyFrom((*(*ex).input), input_shape);
  flat_output.CopyFrom((*(*ex).output), output_shape);
  auto index_input = flat_input.tensor<double, 1>();
  auto index_output = flat_output.tensor<double, 1>();

  for (uint64_t c = 0; c < flat_input_dim; ++c)
    PyList_SET_ITEM(Constants::inputs, start_inputs + c, PyFloat_FromDouble(index_input(c)));
  for (uint64_t c = 0; c < flat_output_dim; ++c)
    PyList_SET_ITEM(Constants::outputs, start_outputs + c, PyFloat_FromDouble(index_input(c)));
  PyList_SetItem(Constants::outcomes, ind, PyFloat_FromDouble((*ex).outcome));
}
template <>
void Queue<at::Tensor>::assign_inputs(std::unique_ptr<TrainingExample<at::Tensor>> &ex, uint64_t ind)
{
  long flat_input_dim = (long)Constants::in_total_planes * 64;
  long flat_output_dim = (long)Constants::out_total_planes * 64;
  long start_inputs = ind * flat_input_dim;
  long start_outputs = ind * flat_output_dim;

  at::Tensor flat_input = (*(*ex).input).view({flat_input_dim});
  at::Tensor flat_output = (*(*ex).output).view({flat_output_dim});
  auto index_input = flat_input.accessor<double, 1>();
  auto index_output = flat_output.accessor<double, 1>();

  for (uint64_t c = 0; c < flat_input_dim; ++c)
    PyList_SET_ITEM(Constants::inputs, start_inputs + c, PyFloat_FromDouble(index_input[c]));
  for (uint64_t c = 0; c < flat_output_dim; ++c)
    PyList_SET_ITEM(Constants::outputs, start_outputs + c, PyFloat_FromDouble(index_input[c]));
  PyList_SetItem(Constants::outcomes, ind, PyFloat_FromDouble((*ex).outcome));
}