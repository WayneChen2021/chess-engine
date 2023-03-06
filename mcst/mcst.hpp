#include <vector>
#include <cmath>
#include <random>

#include "../objects/params.cpp"
#include "zobrist.cpp"
#include "mappings.cpp"

enum WinState
{
  CONTINUE,
  CAN_MATE,
  DRAW
};

template <typename T>
class MCST
{
public:
  MCST<T> *parent;
  std::vector<MCST<T> *> children;
  map moved_from;
  double probability;
  std::unique_ptr<T> input;
  uint64_t output_ind;

  MCST() : brd(Board::Default()), status(BoardStatus::Default()), ep(0),
           get_children(&MoveReceiver::PerfT<Board::Default()>), parent(nullptr), hash(Zobrist::starting_zob()),
           eval(0), visits(0), fifty_move(0), repetitions(0), state(WinState::CONTINUE)
  {
  }

  MCST(MoveInfo &info, MCST<T> &pt) : move(info.move_type), moved_from(info.from), moved_to(info.to),
                                      fifty_move(info.break_consecutive ? pt.fifty_move + 1 : 0),
                                      parent(&pt), eval(0), visits(0), input(nullptr),
                                      white_input(nullptr), black_input(nullptr)
  {
    state = WinState::DRAW;
    if (fifty_move < 100)
    {
      brd = info.brd;
      status = info.status;
      ep = info.ep;
      get_children = &MoveReceiver::PerfT<status>;
      if (pt.status->WhiteMove)
        hash = Zobrist::zob_hash<move, false, pt.status->BCastleL != status->BCastleL, pt.status->BCastleR != status->BCastleR, info.takes, pt.ep != 0>(pt.hash, moved_from, moved_to, pt.brd.White, pt.ep);
      else
        hash = Zobrist::zob_hash<move, true, pt.status->WCastleL != status->WCastleL, pt.status->WCastleR != status->WCastleR, info.takes, pt.ep != 0>(pt.hash, moved_from, moved_to, pt.brd.Black, pt.ep);

      if (Zobrist::handle_zob(hash) != Repeats::Thrice)
        state = WinState::CONTINUE;
    }
  }

  ~MCST()
  {
    if (parent != nullptr)
      delete parent;
    if (get_children != nullptr)
      delete get_children;
  }

  template <typename model_type>
  static void multiple_search(MoveReceiver &mr, Zobrist &zob, Params<T> &params, MCST<T> *start_step, uint64_t sims, model_type &model, std::vector<double> &noise, uint64_t move_cnt)
  {
    for (uint64_t i = 0; i < sims; ++i)
    {
      zob.clear_temp();
      MCST<T> *curr_step = start_step;
      MCST<T> *next_step = curr_step;
      double eval = 0;

      while (true)
      {
        if (curr_step->state == WinState::CONTINUE)
        {
          if (!curr_step->visits)
          {
            MoveInfoList outcomes = curr_step->get_children(curr_step->brd, curr_step->ep, params);
            BoardStatus stat = curr_step->status;
            uint64_t infos_count = outcomes.move_infos.size();
            if (outcomes.mated != -1) // can checkmate from curr_step
              MCST<T>::handle_mate(outcomes.move_infos[outcomes.mated], *curr_step, move_cnt);
            else if (!infos_count) // no legal moves from curr_step
              curr_step->state = WinState::DRAW;
            else
            {
              curr_step->children.reserve(infos_count);
              for (int i = 0; i < infos_count; ++i)
                curr_step->children.push_back(&MCST<T>(outcomes.move_infos[i], curr_step));
              eval = MCST<T>::position_probabilities<stat, model_type>(*curr_step, model, move_cnt);
              curr_step->visits++;
            }
          }
          else
          {
            next_step = MCST<T>::best_child_search<move_cnt == 0>(params, *curr_step, noise);
            next_step->parent = curr_step;
            move_cnt++;
          }
        }
        if (next_step == curr_step)
          break;
        zob.insert_temp(curr_step->hash);
        curr_step = next_step;
      }

      if (curr_step->state == WinState::CAN_MATE)
        MCST<T>::back_track(curr_step, 1);
      else if (curr_step->state == WinState::DRAW)
        MCST<T>::back_track(curr_step->parent, 0);
      else
        MCST<T>::back_track(curr_step->parent, -1 * eval);
    }
  }

  static MCST<T> *best_child_policy(MCST<T> &node, double temperature)
  {
    std::vector<double> probabilities;
    probabilites.reserve(node.children.size());
    for (MCST<T> *ptr : node.children)
    {
      ptr->probability = std::pow(ptr->visits, 1 / temperature) / std::pow(node.visits - 1, 1 / temperature);
      probabilities.push_back(ptr->probability);
    }

    std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<std::size_t> d{probabilities.begin(), probabilites.end()};
    return node.children[d(gen)];
  }

private:
  Board brd;
  BoardStatus status;
  map ep;
  MoveInfoList (*get_children)(Board &, map, Params<T> &);

  uint64_t hash;
  double eval;
  uint64_t visits;

  WinState state;
  map moved_to;
  MoveType move;
  uint64_t fifty_move;
  uint64_t repetitions;

  std::unique_ptr<T> white_input;
  std::unique_ptr<T> black_input;

  static void handle_mate(MoveInfo &info, MCST<T> &parent, uint64_t move_cnt)
  {
    parent.state = WinState::CAN_MATE;

    MCST<T> new_node = MCST<T>();
    new_node.moved_from = info.from;
    new_node.moved_to = info.to;
    new_node.output_ind = get_position<info.move_type>(info.from, info.to);
    initialize_inputs<status, false>(new_node, move_cnt, info.break_consecutive ? parent.fifty_move + 1 : 0);
    concat_inputs<T>(new_node);

    parent.children.push_back(&new_node);
  }

  template <class BoardStatus status, typename model_type>
  static double position_probabilities(MCST<T> &node, model_type &model, uint64_t move_cnt)
  {
    initialize_inputs<status, node.repetitions == 2>(node, move_cnt, node.fifty_move);
    concat_inputs<T>(node);
    ModelOutputs<T> model_pred = call_model<model_type, T>(model, node);

    for (MCST<T> *child : node.children)
    {
      uint64_t ind = get_position<child->move>(child->moved_from, child->moved_to);
      child->probability = tensor_ind<T>(model_pred.policy, child->moved_from, ind);
      child->output_ind = ind;
    }

    return model_pred.eval;
  }

  template <class BoardStatus status, bool repeat_twice>
  static void initialize_inputs(MCST<T> &node, uint64_t move_cnt, uint64_t progress_cnt)
  {
    assign_ind<true>(node.brd.WPawn, node.white_input, Constants::in_default_planes, 0);
    assign_ind<false>(node.brd.WKnight, node.white_input, Constants::in_default_planes, 1);
    assign_ind<false>(node.brd.WBishop, node.white_input, Constants::in_default_planes, 2);
    assign_ind<false>(node.brd.WRook, node.white_input, Constants::in_default_planes, 3);
    assign_ind<false>(node.brd.WQueen, node.white_input, Constants::in_default_planes, 4);
    assign_ind<false>(node.brd.WKing, node.white_input, Constants::in_default_planes, 5);
    assign_ind<false>(node.brd.BPawn, node.white_input, Constants::in_default_planes, 6);
    assign_ind<false>(node.brd.BKnight, node.white_input, Constants::in_default_planes, 7);
    assign_ind<false>(node.brd.BBishop, node.white_input, Constants::in_default_planes, 8);
    assign_ind<false>(node.brd.BRook, node.white_input, Constants::in_default_planes, 9);
    assign_ind<false>(node.brd.BQueen, node.white_input, Constants::in_default_planes, 10);
    assign_ind<false>(node.brd.BKing, node.white_input, Constants::in_default_planes, 11);

    assign_ind<true>(flip_position(node.brd.BPawn), node.black_input, Constants::in_default_planes, 0);
    assign_ind<false>(flip_position(node.brd.BKnight), node.black_input, Constants::in_default_planes, 1);
    assign_ind<false>(flip_position(node.brd.BBishop), node.black_input, Constants::in_default_planes, 2);
    assign_ind<false>(flip_position(node.brd.BRook), node.black_input, Constants::in_default_planes, 3);
    assign_ind<false>(flip_position(node.brd.BQueen), node.black_input, Constants::in_default_planes, 4);
    assign_ind<false>(flip_position(node.brd.BKing), node.black_input, Constants::in_default_planes, 5);
    assign_ind<false>(flip_position(node.brd.WPawn), node.black_input, Constants::in_default_planes, 6);
    assign_ind<false>(flip_position(node.brd.WKnight), node.black_input, Constants::in_default_planes, 7);
    assign_ind<false>(flip_position(node.brd.WBishop), node.black_input, Constants::in_default_planes, 8);
    assign_ind<false>(flip_position(node.brd.WRook), node.black_input, Constants::in_default_planes, 9);
    assign_ind<false>(flip_position(node.brd.WQueen), node.black_input, Constants::in_default_planes, 10);
    assign_ind<false>(flip_position(node.brd.WKing), node.black_input, Constants::in_default_planes, 11);

    if constexpr (repeat_twice)
      if constexpr (status.WhiteMove)
        fill_dim(1., node.white_input, 13, Constants::in_default_planes);
      else
        fill_dim(1., node.black_input, 13, Constants::in_default_planes);
    else if constexpr (status.WhiteMove)
      fill_dim(1., node.white_input, 12, Constants::in_default_planes);
    else
      fill_dim(1., node.black_input, 12, Constants::in_default_planes);

    if constexpr (status.WhiteMove)
    {
      fill_dim(1., node.input, 0, Constants::in_special_planes);
      if constexpr (status.WCastleL)
        fill_dim(1., node.input, 2, Constants::in_special_planes);
      if constexpr (status.WCastleR)
        fill_dim(1., node.input, 3, Constants::in_special_planes);
      if constexpr (status.BCastleR)
        fill_dim(1., node.input, 4, Constants::in_special_planes);
      if constexpr (status.BCastleL)
        fill_dim(1., node.input, 5, Constants::in_special_planes);
    }
    else
    {
      if constexpr (status.BCastleL)
        fill_dim(1., node.input, 2, Constants::in_special_planes);
      if constexpr (status.BCastleR)
        fill_dim(1., node.input, 3, Constants::in_special_planes);
      if constexpr (status.WCastleR)
        fill_dim(1., node.input, 4, Constants::in_special_planes);
      if constexpr (status.WCastleL)
        fill_dim(1., node.input, 5, Constants::in_special_planes);
    }
    fill_dim(std::static_cast<double>(move_cnt), node.input, 1, Constants::in_special_planes);
    fill_dim(std::static_cast<double>(progress_cnt), node.input, 6, Constants::in_special_planes);
  }

  void concat_inputs(MCST<T> &node)
  {
    MCST<T> *curr = &node;
    if (curr->parent == nullptr)
    {
      T zeros = zero_tensor<T>(Constants::in_total_planes);
      if (node.status->WhiteMove)
        node.input = concat<T>(zeros, zeros, zeros, zeros, zeros, zeros, zeros, *node.white_input, *node.input);
      else
        node.input = concat<T>(zeros, zeros, zeros, zeros, zeros, zeros, zeros, *node.black_input, *node.input);
    }
    else
    {
      curr = curr->parent;
      if (curr->parent == nullptr)
      {
        T zeros = zero_tensor<T>(Constants::in_total_planes);
        if (node.status->WhiteMove)
          node.input = concat<T>(zeros, zeros, zeros, zeros, zeros, zeros, *curr->black_input, *node.white_input, *node.input);
        else
          node.input = concat<T>(zeros, zeros, zeros, zeros, zeros, zeros, *curr->white_input, *node.black_input, *node.input);
      }
      else
      {
        curr = curr->parent;
        if (curr->parent == nullptr)
        {
          T zeros = zero_tensor<T>(Constants::in_total_planes);
          if (node.status->WhiteMove)
            node.input = concat<T>(zeros, zeros, zeros, zeros, zeros, *curr->white_input, *curr->black_input, *node.white_input, *node.input);
          else
            node.input = concat<T>(zeros, zeros, zeros, zeros, zeros, *curr->black_input, *curr->white_input, *node.black_input, *node.input);
        }
        else
        {
          curr = curr->parent;
          if (curr->parent == nullptr)
          {
            T zeros = zero_tensor<T>(Constants::in_total_planes);
            if (node.status->WhiteMove)
              node.input = concat<T>(zeros, zeros, zeros, zeros, *curr->black_input, *curr->white_input, *curr->black_input, *node.white_input, *node.input);
            else
              node.input = concat<T>(zeros, zeros, zeros, zeros, *curr->white_input, *curr->black_input, *curr->white_input, *node.black_input, *node.input);
          }
          else
          {
            curr = curr->parent;
            if (curr->parent == nullptr)
            {
              T zeros = zero_tensor<T>(Constants::in_total_planes);
              if (node.status->WhiteMove)
                node.input = concat<T>(zeros, zeros, zeros, *curr->white_input, *curr->black_input, *curr->white_input, *curr->black_input, *node.white_input, *node.input);
              else
                node.input = concat<T>(zeros, zeros, zeros, *curr->black_input, *curr->white_input, *curr->black_input, *curr->white_input, *node.black_input, *node.input);
            }
            else
            {
              curr = curr->parent;
              if (curr->parent == nullptr)
              {
                T zeros = zero_tensor<T>(Constants::in_total_planes);
                if (node.status->WhiteMove)
                  node.input = concat<T>(zeros, zeros, *curr->black_input, *curr->white_input, *curr->black_input, *curr->white_input, *curr->black_input, *node.white_input, *node.input);
                else
                  node.input = concat<T>(zeros, zeros, *curr->white_input, *curr->black_input, *curr->white_input, *curr->black_input, *curr->white_input, *node.black_input, *node.input);
              }
              else
              {
                curr = curr->parent;
                if (curr->parent == nullptr)
                {
                  T zeros = zero_tensor<T>(Constants::in_total_planes);
                  if (node.status->WhiteMove)
                    node.input = concat<T>(zeros, *curr->white_input, *curr->black_input, *curr->white_input, *curr->black_input, *curr->white_input, *curr->black_input, *node.white_input, *node.input);
                  else
                    node.input = concat<T>(zeros, *curr->black_input, *curr->white_input, *curr->black_input, *curr->white_input, *curr->black_input, *curr->white_input, *node.black_input, *node.input);
                }
                else
                {
                  if (node.status->WhiteMove)
                    node.input = concat<T>(*curr->black_input, *curr->white_input, *curr->black_input, *curr->white_input, *curr->black_input, *curr->white_input, *curr->black_input, *node.white_input, *node.input);
                  else
                    node.input = concat<T>(*curr->white_input, *curr->black_input, *curr->white_input, *curr->black_input, *curr->white_input, *curr->black_input, *curr->white_input, *node.black_input, *node.input);
                }
              }
            }
          }
        }
      }
    }
  }

  template <bool include_noise>
  static MCST<T> *best_child_search(Params<T> &params, MCST<T> &node, std::vector<double> &noise)
  {
    double best_bound = -1;
    MCST<T> *ptr;
    uint64_t i = 0;
    for (MCST<T> *child : node.children)
    {
      double bound;
      if constexpr (include_noise)
      {
        bound = params.mcst_params.noise_weight * noise[i] + (1 - params.mcst_params.noise_weight) * (child->eval / std::max(child->visits, 1) + params.mcst_params.cpuct * child->probability * sqrt(node.visits / (1 + child->visits)));
        i++;
      }
      else
        bound = child->eval / std::max(child->visits, 1) + params.mcst_params.cpuct * child->probability * sqrt(node.visits / (1 + child->visits));

      if (bound > best_bound)
      {
        best_bound = bound;
        ptr = child;
      }
    }
    return ptr;
  }

  static void back_track(MCST<T> *start, double eval)
  {
    while (start)
    {
      start->eval += eval;
      start->visits++;
      eval *= -1;
      start = start->parent;
    }
  }
};