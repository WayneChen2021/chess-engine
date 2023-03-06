#include <unordered_map>

#include <tensorflow/core/framework/tensor.h>
#include <ATen/ATen.h>

#include "../array_ops/array_ops.cpp"
#include "trainingexample.cpp"
#include "../variables.hpp"

template <typename input_type>
class Queue
{
public:
  std::unordered_map<uint64_t, std::unique_ptr<TrainingExample<input_type>>> queue;

  Queue() : queue(*(new std::unordered_map<uint64_t, std::unique_ptr<TrainingExample<input_type>>>()))
  {
  }

  void queue_insert(std::unique_ptr<TrainingExample<input_type>> &ex, Params<input_type> &params)
  {
    std::lock_guard<std::mutex> lk(Constants::queue_lock);
    uint ind = Constants::example_count;
    queue[ind] = std::move(ex);
    queue.erase(ind + params.training_params.queue_size);
    Constants::example_count--;
    if (queue.size() == params.training_params.queue_size)
      Constants::start_train.notify_one();
  }

  void fill_inputs(Params<input_type> &params)
  {
    std::unique_lock<std::mutex> lck(Constants::queue_lock);
    Constants::start_train.wait(lck, []
                                { return params.queue.queue.size() < params.training_params.queue_size; });
    std::uniform_int_distribution<> distr(Constants::example_count + 1, Constants::example_count + 1 + params.training_params.queue_size);
    uint64_t size = params.training_params.batch_size * params.training_params.iters_between_eval;
    Constants::inputs = PyList_New(size * Constants::in_total_planes * 64);
    Constants::outputs = PyList_New(size * Constants::out_total_planes * 64);
    Constants::outcomes = PyList_New(size);
    for (uint64_t ind = 0; ind < size; ++ind)
    {
      std::unique_ptr<TrainingExample<input_type>> ex = queue[distr(gen)];
      assign_inputs(ex, ind);
    }
  }

private:
  void assign_inputs(std::unique_ptr<TrainingExample<input_type>> &ex, uint64_t ind);
};