#include <unordered_set>

#include "../movegenerator/Movereceiver.cpp"
#include "../objects/movetype.hpp"

enum Repeats
{
  Once,
  Twice,
  Thrice
};

class Zobrist
{
public:
  std::unordered_set<uint64_t> visited_1;
  std::unordered_set<uint64_t> visited_2;
  std::unordered_set<uint64_t> temp_visited_1;
  std::unordered_set<uint64_t> temp_visited_2;

  static constexpr uint64_t starting_zob()
  {
    uint64_t output = Chess_Lookup::zob_pieces[64 * BoardPiece::Rook];
    output ^= Chess_Lookup::zob_pieces[1 + 64 * BoardPiece::Knight];
    output ^= Chess_Lookup::zob_pieces[2 + 64 * BoardPiece::Bishop];
    output ^= Chess_Lookup::zob_pieces[3 + 64 * BoardPiece::King];
    output ^= Chess_Lookup::zob_pieces[4 + 64 * BoardPiece::Queen];
    output ^= Chess_Lookup::zob_pieces[5 + 64 * BoardPiece::Bishop];
    output ^= Chess_Lookup::zob_pieces[6 + 64 * BoardPiece::Knight];
    output ^= Chess_Lookup::zob_pieces[7 + 64 * BoardPiece::Rook];
    output ^= Chess_Lookup::zob_pieces[8 + 64 * BoardPiece::Pawn];
    output ^= Chess_Lookup::zob_pieces[9 + 64 * BoardPiece::Pawn];
    output ^= Chess_Lookup::zob_pieces[10 + 64 * BoardPiece::Pawn];
    output ^= Chess_Lookup::zob_pieces[11 + 64 * BoardPiece::Pawn];
    output ^= Chess_Lookup::zob_pieces[12 + 64 * BoardPiece::Pawn];
    output ^= Chess_Lookup::zob_pieces[13 + 64 * BoardPiece::Pawn];
    output ^= Chess_Lookup::zob_pieces[14 + 64 * BoardPiece::Pawn];
    output ^= Chess_Lookup::zob_pieces[15 + 64 * BoardPiece::Pawn];

    output ^= Chess_Lookup::zob_pieces[56 + 64 * BoardPiece::Rook];
    output ^= Chess_Lookup::zob_pieces[57 + 64 * BoardPiece::Knight];
    output ^= Chess_Lookup::zob_pieces[58 + 64 * BoardPiece::Bishop];
    output ^= Chess_Lookup::zob_pieces[59 + 64 * BoardPiece::King];
    output ^= Chess_Lookup::zob_pieces[60 + 64 * BoardPiece::Queen];
    output ^= Chess_Lookup::zob_pieces[61 + 64 * BoardPiece::Bishop];
    output ^= Chess_Lookup::zob_pieces[62 + 64 * BoardPiece::Knight];
    output ^= Chess_Lookup::zob_pieces[63 + 64 * BoardPiece::Rook];
    output ^= Chess_Lookup::zob_pieces[48 + 64 * BoardPiece::Pawn];
    output ^= Chess_Lookup::zob_pieces[49 + 64 * BoardPiece::Pawn];
    output ^= Chess_Lookup::zob_pieces[50 + 64 * BoardPiece::Pawn];
    output ^= Chess_Lookup::zob_pieces[51 + 64 * BoardPiece::Pawn];
    output ^= Chess_Lookup::zob_pieces[52 + 64 * BoardPiece::Pawn];
    output ^= Chess_Lookup::zob_pieces[53 + 64 * BoardPiece::Pawn];
    output ^= Chess_Lookup::zob_pieces[54 + 64 * BoardPiece::Pawn];
    output ^= Chess_Lookup::zob_pieces[55 + 64 * BoardPiece::Pawn];

    return output;
  }

  template <MoveType move, bool white, bool change_castle_l, bool change_castle_r, bool takes, bool undo_ep>
  static uint64_t zob_hash(uint64_t prev_hash, map from, map to, Board &enemy, map old_ep)
  {
    if constexpr (!white)
      prev_hash ^= Chess_Lookup::zob_black;

    if constexpr (takes)
    {
      uint64_t addition = 0;
      map to_ind = Lookup(to);
      if constexpr (white)
      {
        if (to && enemy->BPawn)
          addition = Chess_Lookup::zob_pieces[to_ind + BoardPiece::Pawn * 64];
        else if (to && enemy->BKnight)
          addition = Chess_Lookup::zob_pieces[to_ind + BoardPiece::Knight * 64];
        else if (to && enemy->BBishop)
          addition = Chess_Lookup::zob_pieces[to_ind + BoardPiece::Bishop * 64];
        else if (to && enemy->BRook)
          addition = Chess_Lookup::zob_pieces[to_ind + BoardPiece::Rook * 64];
        else
          addition = Chess_Lookup::zob_pieces[to_ind + BoardPiece::Queen * 64];
      }
      else
      {
        if (to && enemy->WPawn)
          addition = Chess_Lookup::zob_pieces[to_ind + BoardPiece::Pawn * 64];
        else if (to && enemy->WKnight)
          addition = Chess_Lookup::zob_pieces[to_ind + BoardPiece::Knight * 64];
        else if (to && enemy->WBishop)
          addition = Chess_Lookup::zob_pieces[to_ind + BoardPiece::Bishop * 64];
        else if (to && enemy->WRook)
          addition = Chess_Lookup::zob_pieces[to_ind + BoardPiece::Rook * 64];
        else
          addition = Chess_Lookup::zob_pieces[to_ind + BoardPiece::Queen * 64];
      }
      prev_hash ^= addition;
    }

    if constexpr (undo_ep)
      prev_hash ^= Chess_Lookup::zob_enpassant[Lookup(old_ep) >> 3];

    Square to_sq = Lookup(to);
    Square from_sq = Lookup(from);

    if constexpr (move == move_type::King)
    {
      prev_hash ^= Chess_Lookup::zob_pieces[from_sq + BoardPiece::King * 64];
      prev_hash ^= Chess_Lookup::zob_pieces[to_sq * BoardPiece::King * 64];
      if constexpr (change_castle_l || change_castle_r)
      {
        // Set any castling flag to indicate first king move
        if constexpr (white)
        {
          prev_hash ^= Chess_Lookup::zob_WCastleR;
          prev_hash ^= Chess_Lookup::zob_WCastleL;
        }
        else
        {
          prev_hash ^= Chess_Lookup::zob_BCastleR;
          prev_hash ^= Chess_Lookup::zob_BCastleL;
        }
      }
    }
    else if constexpr (move == move_type::KingCastle)
    {
      prev_hash ^= Chess_Lookup::zob_pieces[from_sq * BoardPiece::King * 64];
      prev_hash ^= Chess_Lookup::zob_pieces[to_sq * BoardPiece::King * 64];
      // only the castling flag for the castling side is set to true
      if constexpr (change_castle_l)
      {
        if constexpr (white)
        {
          prev_hash ^= Chess_Lookup::zob_pieces[7 + BoardPiece::Rook * 64];
          prev_hash ^= Chess_Lookup::zob_pieces[4 + BoardPiece::Rook * 64];
        }
        else
        {
          prev_hash ^= Chess_Lookup::zob_pieces[56 + BoardPiece::Rook * 64];
          prev_hash ^= Chess_Lookup::zob_pieces[58 + BoardPiece::Rook * 64];
        }
      }
      else
      {
        if constexpr (white)
        {
          prev_hash ^= Chess_Lookup::zob_pieces[BoardPiece::Rook * 64];
          prev_hash ^= Chess_Lookup::zob_pieces[2 + BoardPiece::Rook * 64];
        }
        else
        {
          prev_hash ^= Chess_Lookup::zob_pieces[63 + BoardPiece::Rook * 64];
          prev_hash ^= Chess_Lookup::zob_pieces[60 + BoardPiece::Rook * 64];
        }
      }
    }
    else if constexpr (move == move_type::Pawn || move == move_type::PawnTake)
    {
      prev_hash ^= Chess_Lookup::zob_pieces[from_sq + BoardPiece::Pawn * 64];
      prev_hash ^= Chess_Lookup::zob_pieces[to_sq + BoardPiece::Pawn * 64];
    }
    else if constexpr (move == move_type::PawnPush)
    {
      prev_hash ^= Chess_Lookup::zob_pieces[from_sq + BoardPiece::Pawn * 64];
      prev_hash ^= Chess_Lookup::zob_pieces[to_sq + BoardPiece::Pawn * 64];
      prev_hash ^= Chess_Lookup::zob_enpassant[to_sq >> 3];
    }
    else if constexpr (move == move_type::PromoteQueen)
    {
      prev_hash ^= Chess_Lookup::zob_pieces[from_sq + BoardPiece::Pawn * 64];
      prev_hash ^= Chess_Lookup::zob_pieces[to_sq + BoardPiece::Queen * 64];
    }
    else if constexpr (move == move_type::PromoteRook)
    {
      prev_hash ^= Chess_Lookup::zob_pieces[from_sq + BoardPiece::Pawn * 64];
      prev_hash ^= Chess_Lookup::zob_pieces[to_sq + BoardPiece::Rook * 64];
    }
    else if constexpr (move == move_type::PromoteKnight)
    {
      prev_hash ^= Chess_Lookup::zob_pieces[from_sq + BoardPiece::Pawn * 64];
      prev_hash ^= Chess_Lookup::zob_pieces[to_sq + BoardPiece::Knight * 64];
    }
    else if constexpr (move == move_type::Bishop)
    {
      prev_hash ^= Chess_Lookup::zob_pieces[from_sq + BoardPiece::Pawn * 64];
      prev_hash ^= Chess_Lookup::zob_pieces[to_sq + BoardPiece::Bishop * 64];
    }
    else if constexpr (move == move_type::Knight)
    {
      prev_hash ^= Chess_Lookup::zob_pieces[from_sq + BoardPiece::Knight * 64];
      prev_hash ^= Chess_Lookup::zob_pieces[to_sq + BoardPiece::Knight * 64];
    }
    else if constexpr (move == move_type::Bishop)
    {
      prev_hash ^= Chess_Lookup::zob_pieces[from_sq + BoardPiece::Bishop * 64];
      prev_hash ^= Chess_Lookup::zob_pieces[to_sq + BoardPiece::Bishop * 64];
    }
    else if constexpr (move == move_type::Rook)
    {
      prev_hash ^= Chess_Lookup::zob_pieces[from_sq + BoardPiece::Rook * 64];
      prev_hash ^= Chess_Lookup::zob_pieces[to_sq + BoardPiece::Rook * 64];
    }
    else
    {
      prev_hash ^= Chess_Lookup::zob_pieces[from_sq + BoardPiece::Queen * 64];
      prev_hash ^= Chess_Lookup::zob_pieces[to_sq + BoardPiece::Queen * 64];
    }

    return prev_hash;
  }

  void clear_temp()
  {
    temp_visited_1.clear();
    temp_visited_2.clear();
  }

  void insert_base(uint64_t new_hash)
  {
    insert_impl(new_hash, visited_1, visited_2);
  }

  void insert_temp(uint64_t new_hash)
  {
    insert_impl(new_hash, temp_visited_1, temp_visited_2);
  }

  Repeats handle_hash(uint64_t hash)
  {
    if (visited_1.contains(hash))
    {
      temp_visited_2.insert(hash);
      return Repeats::Twice;
    }
    else if (temp_visited_1.erase(hash))
    {
      temp_visited_2.insert(hash);
      return Repeats::Twice;
    }
    else
    {
      if (visited_2.contains(hash))
        return Repeats::Thrice;
      else if (temp_visited_2.erase(hash))
        return Repeats::Thrice;
      else
        temp_visited_1.insert(hash);
    }
  }

private:
  void insert_impl(uint64_t new_hash, std::unordered_set<uint64_t> cont_1, std::unordered_set<uint64_t> cont_2)
  {
    if (!cont_1.erase(new_hash))
      cont_1.insert(new_hash);
    else
      cont_2.insert(new_hash);
  }
};