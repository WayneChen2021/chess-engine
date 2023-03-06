#include <memory>
#include <vector>

#include "Movelist.hpp"
#include "../objects/movetype.hpp"

struct MoveInfo
{
  BoardStatus status;
  Board brd;
  map ep;
  map from;
  map to;
  bool break_consecutive;
  MoveType move_type;
  bool takes;
};
struct MoveInfoList
{
  int mated;
  std::vector<std::unique_ptr<MoveInfo>> move_infos;
};

class MoveReceiver
{
public:
  int mated;
  std::vector<std::unique_ptr<MoveInfo>> move_infos;

  template <class BoardStatus status>
  static _ForceInline MoveInfoList PerfT(Board &brd, map ep, uint64_t legal_moves_estimate)
  {
    mated = -1;
    move_infos.reserve(legal_moves_estimate);

    Movelist::EnPassantTarget = ep;
    Movelist::InitStack<status>(brd);
    Movelist::EnumerateMoves<status, MoveReceiver>(brd);

    return MoveInfoList{mated, move_infos};
  }

private:
  template <class BoardStatus status>
  static _ForceInline bool PerfT1(Board &brd)
  {
    return Movelist::count<status, MoveReceiver>(brd);
  }

  template <class BoardStatus status>
  static _Inline void Kingmove(const Board &brd, uint64_t from, uint64_t to)
  {
    Board next = Board::Move<BoardPiece::King, status.WhiteMove>(brd, from, to, to & Enemy<status.WhiteMove>(brd));
    RegisterMove<status.KingMove>(next, from, to, 0, false, move_type::King);
  }

  template <class BoardStatus status>
  static _Inline void KingCastle(const Board &brd, uint64_t kingswitch, uint64_t rookswitch)
  {
    Board next = Board::MoveCastle<status.WhiteMove>(brd, kingswitch, rookswitch);
    RegisterMove<status.KingMove>(next, from, to, 0, false, move_type::KingCastle);
  }

  template <class BoardStatus status>
  static _Inline void PawnCheck(map eking, uint64_t to)
  {
    constexpr bool white = status.WhiteMove;
    map pl = Pawn_AttackLeft<white>(to & Pawns_NotLeft());
    map pr = Pawn_AttackRight<white>(to & Pawns_NotRight());

    if (eking & (pl | pr))
      Movestack::Check_Status[1] = to;
  }

  template <class BoardStatus status>
  static _Inline void KnightCheck(map eking, uint64_t to)
  {
    constexpr bool white = status.WhiteMove;
    if (Lookup::Knight(SquareOf(eking)) & to)
      Movestack::Check_Status[1] = to;
  }

  template <class BoardStatus status>
  static _Inline void Pawnmove(const Board &brd, uint64_t from, uint64_t to)
  {
    Board next = Board::Move<BoardPiece::Pawn, status.WhiteMove, false>(brd, from, to);
    PawnCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);
    RegisterMove<status.SilentMove>(next, from, to, 0, true, move_type::Pawn);
    Movestack::Check_Status[1] = 0xffffffffffffffffull;
  }

  template <class BoardStatus status>
  static _Inline void Pawnatk(const Board &brd, uint64_t from, uint64_t to)
  {
    Board next = Board::Move<BoardPiece::Pawn, status.WhiteMove, true>(brd, from, to);
    PawnCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);
    RegisterMove<status.SilentMove>(next, from, to, 0, true, move_type::PawnTake);
    Movestack::Check_Status[1] = 0xffffffffffffffffull;
  }

  template <class BoardStatus status>
  static _Inline void PawnEnpassantTake(const Board &brd, uint64_t from, uint64_t enemy, uint64_t to)
  {
    Board next = Board::MoveEP<status.WhiteMove>(brd, from, enemy, to);
    PawnCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);
    RegisterMove<status.SilentMove>(next, from, to, 0, true, move_type::PawnTake);
    Movestack::Check_Status[1] = 0xffffffffffffffffull;
  }

  template <class BoardStatus status>
  static _Inline void Pawnpush(const Board &brd, uint64_t from, uint64_t to)
  {
    Board next = Board::Move<BoardPiece::Pawn, status.WhiteMove, false>(brd, from, to);
    Movelist::EnPassantTarget = to;
    PawnCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);
    RegisterMove<status.PawnPush>(next, from, to, to, true, move_type::PawnPush);
    Movestack::Check_Status[1] = 0xffffffffffffffffull;
  }

  template <class BoardStatus status>
  static _Inline void Pawnpromote(const Board &brd, uint64_t from, uint64_t to)
  {
    Board next1 = Board::MovePromote<BoardPiece::Queen, status.WhiteMove>(brd, from, to);
    RegisterMove<status.SilentMove>(next1, from, to, 0, true, move_type::PromoteQueen);

    Board next2 = Board::MovePromote<BoardPiece::Knight, status.WhiteMove>(brd, from, to);
    KnightCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);
    RegisterMove<status.SilentMove>(next2, from, to, 0, true, move_type::PromoteKnight);
    Movestack::Check_Status[1] = 0xffffffffffffffffull;

    Board next3 = Board::MovePromote<BoardPiece::Bishop, status.WhiteMove>(brd, from, to);
    RegisterMove<status.SilentMove>(next3, from, to, 0, true, move_type::PromoteBishop);

    Board next4 = Board::MovePromote<BoardPiece::Rook, status.WhiteMove>(brd, from, to);
    RegisterMove<status.SilentMove>(next4, from, to, 0, true, move_type::PromoteRook);
  }

  template <class BoardStatus status>
  static _Inline void Knightmove(const Board &brd, uint64_t from, uint64_t to)
  {
    Board next = Board::Move<BoardPiece::Knight, status.WhiteMove>(brd, from, to, to & Enemy<status.WhiteMove>(brd));
    KnightCheck<status, depth>(EnemyKing<status.WhiteMove>(brd), to);
    RegisterMove<status.SilentMove>(next, from, to, 0, false, move_type::Knight);
    Movestack::Check_Status[1] = 0xffffffffffffffffull;
  }

  template <class BoardStatus status>
  static _Inline void Bishopmove(const Board &brd, uint64_t from, uint64_t to)
  {
    Board next = Board::Move<BoardPiece::Bishop, status.WhiteMove>(brd, from, to, to & Enemy<status.WhiteMove>(brd));
    RegisterMove<status.SilentMove>(next, from, to, 0, false, move_type::Bishop);
  }

  template <class BoardStatus status>
  static _Inline void Rookmove(const Board &brd, uint64_t from, uint64_t to)
  {
    Board next = Board::Move<BoardPiece::Rook, status.WhiteMove>(brd, from, to, to & Enemy<status.WhiteMove>(brd));
    if constexpr (status.CanCastle())
    {
      if (status.IsLeftRook(from))
        RegisterMove<status.RookMove_Left>(next, from, to, 0, false, move_type::Rook);
      else if (status.IsRightRook(from))
        RegisterMove<status.RookMove_Right>(next, from, to, 0, false, move_type::Rook);
      else
        RegisterMove<status.SilentMove>(next, from, to, 0, false, move_type::Rook);
    }
    else
      RegisterMove<status.SilentMove>(next, from, to, 0, false, move_type::Rook);
  }

  template <class BoardStatus status>
  static _Inline void Queenmove(const Board &brd, uint64_t from, uint64_t to)
  {
    Board next = Board::Move<BoardPiece::Queen, status.WhiteMove>(brd, from, to, to & Enemy<status.WhiteMove>(brd));
    RegisterMove<status.SilentMove>(next, from, to, 0, false, move_type::Queen);
  }

  template <class BoardStatus status>
  static _Inline void RegisterMove(const Board &brd, uint64_t from, uint64_t to, uint64_t ep, bool pawn_move, MoveType move)
  {
    bool takes = to & Enemy<status.WhiteMove>(brd);
    move_infos.push_back(
        std::make_unique<MoveInfo>(
            {std::make_unique<BoardStatus> status,
             std::make_unique<Board> brd,
             ep,
             from,
             to,
             pawm_move || takes,
             move,
             takes}));

    if (mated != -1)
      PerfT1<status>(brd);
  }
};