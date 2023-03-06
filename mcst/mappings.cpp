#include "mappings.hpp"

template std::uint64_t check_shifts<true>(uint64_t, map, map);
template std::uint64_t check_shifts<false>(uint64_t, map, map);

template uint64_t get_position<MoveType::King>(map, map);
template uint64_t get_position<MoveType::KingCastle>(map, map);
template uint64_t get_position<MoveType::Pawn>(map, map);
template uint64_t get_position<MoveType::PawnTake>(map, map);
template uint64_t get_position<MoveType::PawnPush>(map, map);
template uint64_t get_position<MoveType::PromoteQueen>(map, map);
template uint64_t get_position<MoveType::PromoteRook>(map, map);
template uint64_t get_position<MoveType::PromoteKnight>(map, map);
template uint64_t get_position<MoveType::PromoteBishop>(map, map);
template uint64_t get_position<MoveType::Knight>(map, map);
template uint64_t get_position<MoveType::Bishop>(map, map);
template uint64_t get_position<MoveType::Rook>(map, map);
template uint64_t get_position<MoveType::Queen>(map, map);

template <>
double tensor_ind<tensorflow::Tensor>(tensorflow::Tensor &arr, map from, uint64_t last_dim)
{
  auto tensor = (arr).tensor<double, 2>();
  return tensor(from, last_dim);
}
template <>
double tensor_ind<at::Tensor>(at::Tensor &arr, map from, uint64_t last_dim)
{
  auto tensor = (arr).accessor<double, 2>();
  return tensor[last_dim][from];
}