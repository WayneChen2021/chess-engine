#include <memory>

#include <tensorflow/core/framework/tensor.h>
#include <ATen/ATen.h>

template <typename input_type>
class TrainingExample
{
public:
  std::unique_ptr<input_type> input;
  std::unique_ptr<input_type> output;
  double outcome;

  TrainingExample(std::unique_ptr<input_type> in, input_type out, double res)
      : input(std::move(in)), output(std::make_unique<input_type>(out)), outcome(res)
  {
  }
};