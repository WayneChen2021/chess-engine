#include <memory>
#include <mutex>
#include <condition_variable>
#include <Python.h>
#include <string>

namespace Constants
{
  const uint64_t in_special_planes = 7;
  const uint64_t in_default_planes = 14;
  const uint64_t in_total_planes = 8 * in_default_planes + in_special_planes;
  const uint64_t out_total_planes = 73;

  std::mutex queue_lock;
  uint64_t example_count = -1;
  std::condition_variable start_train;
  bool finished = false;
  const std::string ckpt_path = "data";
  uint64_t best_ckpt = 0;
  uint64_t latest_ckpt = 0;

  PyObject *inputs;
  PyObject *outputs;
  PyObject *outcomes;
}