#include <fstream>

#include "array_ops/dirichlet.hpp"
#include "mcst/mcst.cpp"

using json = nlohmann::json;

struct win_totals
{
  uint64_t black;
  uint64_t white;
  uint64_t draws;
};

template <typename input_type>
Params<input_type> setup(json &data)
{
  return Params<input_type>{true,
                            TrainingParams(data["training_params"]["cpp"]),
                            PitParams(data["pit_params"]),
                            MCSTParams(data["mcst_params"]),
                            Queue<input_type>()};
}

template <typename model_type>
model_type assign_model(std::string &dir);

template <typename input_type>
int append_training_examples(MCST<input_type> *last_node, double outcome, Params<input_type> &params)
{
  while (!last_node)
  {
    TrainingExample<input_type> example = {std::make_unique<input_type>(last_node->input), set_input<input_type>(last_node), outcome};
    params.queue.queue_insert(std::make_unique<TrainingExample<input_type>>(TrainingExample<input_type>(last_node->input, set_output<input_type>(*last_node), outcome)), params);
    outcome *= -1;
    last_node = last_node->parent;
  }

  return 0;
}

template <typename input_type>
input_type set_output(MCST<input_type> &node);

template <bool is_pit, typename model_type, typename input_type>
win_totals simulate_games(Params<input_type> &params, uint64_t games, uint64_t sims, std::string &white_dir, std::string &black_dir)
{
  MoveReceiver mr = MoveReceiver();
  uint64_t white_wins = 0;
  uint64_t black_wins = 0;
  for (uint64_t game_c = 0; game_c < games; ++game_c)
  {
    std::vector<double> noise = basic_distr(params.mcst_params.noise_alpha, 20);
    Zobrist zob = Zobrist();
    zob.insert_base(Zobrist::starting_zob());
    uint64_t game_step = 0;

    MCST<input_type> *curr_step = &MCST<input_type>();
    MCST<input_type> *next_step;
    model_type global_model = assign_model<model_type>(Constants::ckpt_path + std::to_string(Constants::best_ckpt));
    model_type white_model = assign_model<model_type>(white_dir);
    model_type black_model = assign_model<model_type>(black_dir);

    while (true)
    {
      if constexpr (!is_pit)
        MCST<input_type>::multiple_search<model_type>(mr, zob, params, curr_step, sims, global_model, noise, game_step);
      else if (game_step % 2)
        MCST<input_type>::multiple_search<model_type>(mr, zob, params, curr_step, sims, black_model, noise, game_step);
      else
        MCST<input_type>::multiple_search<model_type>(mr, zob, params, curr_step, sims, white_model, noise, game_step);

      if (game_step <= params.mcst_params.temp_threshold && !is_pit)
        next_step = MCST<input_type>::best_child_policy(*curr_step, 1.);
      else
        next_step = MCST<input_type>::best_child_policy(*curr_step, params.mcst_params.zero_temp);

      next_step->parent = curr_step;
      curr_step = next_step;
      ++game_step;

      if (curr_step->state == WinState::CAN_MATE)
      {
        if constexpr (is_pit)
          game_step % 2 ? ++black_wins : ++white_wins;
        else
        {
          curr_step->children[0]->probability = 1.;
          append_training_examples<input_type>(curr_step, 1, params);
        }
        break;
      }
      else if (curr_step->state == WinState::DRAW)
      {
        if (!is_pit)
          append_training_examples<input_type>(curr_step->parent, 0, params);
        break;
      }
    }
  }

  return {black_wins, white_wins, games - black_wins - white_wins};
}

template <typename model_type, typename input_type>
void eval_network(Params<input_type> &params)
{
  std::string latest_ckpt = Constants::ckpt_path + std::to_string(Constants::latest_ckpt);
  std::string best_ckpt = Constants::ckpt_path + std::to_string(Constants::best_ckpt);
  uint64_t new_wins = 0;
  uint64_t old_wins = 0;
  uint64_t draws = 0;

  win_totals result = simulate_games<true, model_type, input_type>(params, params.pit_params.game_count / 2, params.mcst_params.searches_per_state, best_ckpt, latest_ckpt);
  old_wins += result.white;
  new_wins += result.black;
  draws += result.draws;

  win_totals result = simulate_games<true, model_type, input_type>(params, params.pit_params.game_count / 2, params.mcst_params.searches_per_state, latest_ckpt, best_ckpt);
  old_wins += result.black;
  new_wins += result.white;
  draws += result.draws;

  if ((double)new_wins / (new_wins + old_wins) > params.pit_params.update_thresh)
    Constants::best_ckpt = Constants::latest_ckpt;
}

template <typename model_type, typename input_type>
void train_eval(Params<input_type> &params)
{
  Py_Initialize();
  PyObject *sys_path = PySys_GetObject("path");
  PyList_Append(sys_path, PyUnicode_FromString("/root/chess/training"));
  PyObject *pymodule;
  if (params.is_tf)
    PyObject *pymodule = PyImport_ImportModule("train_tf");
  else
    PyObject *pymodule = PyImport_ImportModule("train_torch");
  PyObject *pyfunc = PyObject_GetAttrString(pymodule, "train");
  params.queue.fill_inputs<input_type>(params);
  PyObject *arglist = Py_BuildValue("(o)", Constants::inputs, Constants::outputs, Constants::outcomes);
  PyObject *result = PyObject_CallObject(pyfunc, arglist);
  ++Constants::latest_ckpt;

  std::thread t(eval_network<model_type, input_type>, params);
  t.join();
}

template <typename model_type, typename input_type>
void start(json &data)
{
  Params<input_type> params = setup<input_type>(data);
  uint64_t num_games = 1 + params.training_params.training_iters * params.mcst_params.games_per_iter / params.mcst_params.thread_count;
  for (uint64_t i = 0; i < params.mcst_params.thread_count; ++i)
  {
    std::thread t(simulate_games<false, model_type, input_type>, params, num_games, params.mcst_params.searches_per_state, std::string(), std::string());
    t.join()
  }
}