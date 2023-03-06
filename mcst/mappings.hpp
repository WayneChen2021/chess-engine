#include <tensorflow/core/framework/tensor.h>
#include <ATen/ATen.h>

#include "../movegenerator/Movereceiver.cpp"
#include "../objects/movetype.hpp"

enum Direction
{
  NW,
  N,
  NE,
  E,
  SE,
  S,
  SW,
  W
};

map flip_position(map x)
{
  x = Byteswap64(x);
  const uint64_t k1 = 0x5555555555555555;
  const uint64_t k2 = 0x3333333333333333;
  const uint64_t k4 = 0x0f0f0f0f0f0f0f0f;
  x = ((x >> 1) & k1) + 2 * (x & k1);
  x = ((x >> 2) & k2) + 4 * (x & k2);
  x = ((x >> 4) & k4) + 16 * (x & k4);
  return x;
}

template <bool is_right>
std::uint64_t check_shifts(uint64_t increment, map from, map to)
{
  if constexpr (is_right)
  {
    for (int i = 1; i <= 7; ++i)
    {
      from >>= increment * i;
      if (from == to)
        return i;
      if (from == 0)
        return 0;
    }
    return 0;
  }

  for (int i = 1; i <= 7; ++i)
  {
    from <<= increment * i;
    if (from == to)
      return i;
    if (from == 0)
      return 0;
  }
  return 0;
}

template <MoveType type>
uint64_t get_position(map from_pos, map to_pos)
{
  if constexpr (type == MoveType::King)
  {
    if (from_pos << 8 == to_pos)
      return Direction::N;
    if (from_pos >> 8 == to_pos)
      return Direction::S;
    if (from_pos << 1 == to_pos)
      return Direction::W;
    if (from_pos >> 1 == to_pos)
      return Direction::E;
    if (from_pos << 9 == to_pos)
      return Direction::NW;
    if (from_pos << 7 == to_pos)
      return Direction::NE;
    if (from_pos >> 7 == to_pos)
      return Direction::SW;
    return Direction::SE;
  }
  else if constexpr (type == MoveType::KingCastle)
    return (from_pos << 2 == to_pos) ? 8 + Direction::W : 8 + Direction::E;
  else if constexpr (type == MoveType::Pawn)
    return Direction::N;
  else if constexpr (type == MoveType::PawnTake)
    return (from_pos << 9 == to_pos) ? Direction::NW : Direction::NE;
  else if constexpr (type == MoveType::PawnPush)
    return 8 + Direction::N;
  else if constexpr (type == MoveType::PromoteRook)
  {
    if (from_pos << 9 == to_pos)
      return 64;
    if (from_pos << 7 == to_pos)
      return 66;
    return 65;
  }
  else if constexpr (type == MoveType::PromoteBishop)
  {
    if (from_pos << 9 == to_pos)
      return 67;
    if (from_pos << 7 == to_pos)
      return 69;
    return 68;
  }
  else if constexpr (type == MoveType::PromoteKnight)
  {
    if (from_pos << 9 == to_pos)
      return 70;
    if (from_pos << 7 == to_pos)
      return 72;
    return 71;
  }
  else if constexpr (type == MoveType::PromoteQueen)
  {
    if (from_pos << 9 == to_pos)
      return Direction::NW;
    if (from_pos << 7 == to_pos)
      return Direction::NE;
    return Direction::N;
  }
  else if constexpr (type == MoveType::Knight)
  {
    if (from_pos << 10 == to_pos)
      return 56; // NW going clockwise
    if (from_pos << 17 == to_pos)
      return 57;
    if (from_pos << 15 == to_pos)
      return 58;
    if (from_pos << 6 == to_pos)
      return 59;
    if (from_pos >> 10 == to_pos)
      return 60;
    if (from_pos >> 17 == to_pos)
      return 61;
    if (from_pos >> 15 == to_pos)
      return 62;
    return 63;
  }
  else if constexpr (type == MoveType::Bishop)
  {
    uint64_t size = check_shifts<false>(9, from_pos, to_pos);
    if (size != 0)
      return Direction::NW + (size - 1) * 8;
    size = check_shifts<false>(7, from_pos, to_pos);
    if (size != 0)
      return Direction::NE + (size - 1) * 8;
    size = check_shifts<true>(9, from_pos, to_pos);
    if (size != 0)
      return Direction::SE + (size - 1) * 8;
    return Direction::SW + (check_shifts<true>(7, from_pos, to_pos) - 1) * 8;
  }
  else if constexpr (type == MoveType::Rook)
  {
    uint64_t size = check_shifts<false>(8, from_pos, to_pos);
    if (size != 0)
      return Direction::N + (size - 1) * 8;
    size = check_shifts<true>(8, from_pos, to_pos);
    if (size != 0)
      return Direction::S + (size - 1) * 8;
    size = check_shifts<false>(1, from_pos, to_pos);
    if (size != 0)
      return Direction::W + (size - 1) * 8;
    return Direction::E + (check_shifts<true>(1, from_pos, to_pos) - 1) * 8;
  }
  else if constexpr (type == MoveType::Queen)
  {
    uint64_t size = check_shifts<false>(9, from_pos, to_pos);
    if (size != 0)
      return Direction::NW + (size - 1) * 8;
    size = check_shifts<false>(7, from_pos, to_pos);
    if (size != 0)
      return Direction::NE + (size - 1) * 8;
    size = check_shifts<true>(9, from_pos, to_pos);
    if (size != 0)
      return Direction::SE + (size - 1) * 8;
    size = check_shifts<true>(7, from_pos, to_pos);
    if (size != 0)
      return Direction::SW + (size - 1) * 8;
    size = check_shifts<false>(8, from_pos, to_pos);
    if (size != 0)
      return Direction::N + (size - 1) * 8;
    size = check_shifts<true>(8, from_pos, to_pos);
    if (size != 0)
      return Direction::S + (size - 1) * 8;
    size = check_shifts<false>(1, from_pos, to_pos);
    if (size != 0)
      return Direction::W + (size - 1) * 8;
    return Direction::E + (check_shifts<true>(1, from_pos, to_pos) - 1) * 8;
  }
}

template <typename input_type>
double tensor_ind(input_type &arr, map from, uint64_t last_dim);