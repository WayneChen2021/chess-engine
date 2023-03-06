#include "json.hpp"
#include "queue.cpp"

using json = nlohmann::json;

class TrainingParams
{
public:
  uint64_t curr_iter_num;
  const uint64_t training_iters;
  const uint64_t game_len_estimate;
  const uint64_t batch_size;
  const uint64_t retained_games;
  const uint64_t queue_size;
  const uint64_t iters_between_eval;

  TrainingParams(json js)
      : curr_iter_num(js["curr_iter_num"]), training_iters(js["training_iters"]),
        game_len_estimate(js["game_len_estimate"]), batch_size(js["batch_size"]),
        retained_games(js["retained_games"]), queue_size(game_len_estimate * retained_games),
        iters_between_eval(js["iters_between_eval"])
  {
  }
};
class PitParams
{
public:
  const uint64_t game_count;
  const double update_thresh;
  const uint64_t thread_count;

  PitParams(json js)
      : game_count(js["game_count"]), update_thresh(js["update_thresh"]), thread_count(js["thread_count"])
  {
  }
};
class MCSTParams
{
public:
  const uint64_t temp_threshold;
  const double zero_temp;
  const double cpuct;
  const uint64_t games_per_iter;
  const uint64_t searches_per_state;
  const uint64_t legal_moves_estimate;
  const double noise_alpha;
  const double noise_weight;
  const uint64_t thread_count;

  MCSTParams(json js)
      : temp_threshold(js["temp_threshold"]), zero_temp(js["zero_temp"]), cpuct(js["cpuct"]),
        games_per_iter(js["games_per_iter"]), searches_per_state(js["searches_per_state"]),
        legal_moves_estimate(js["legal_moves_estimate"]), noise_alpha(js["noise_alpha"]),
        noise_weight(js["noise_weight"]), thread_count(js["data_threads"])
  {
  }
};
template <typename input_type>
struct Params
{
  bool is_tf;
  TrainingParams training_params;
  PitParams pit_params;
  MCSTParams mcst_params;
  Queue<input_type> queue;
};