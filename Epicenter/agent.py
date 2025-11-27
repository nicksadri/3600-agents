from collections.abc import Callable
from typing import List, Tuple, Optional

import time

from game.board import Board
from game.enums import Direction, MoveType, Result
from .trapdoor_tracker import TrapdoorTracker


class PlayerAgent:
    """
    Version 1: Risk-aware alpha–beta search on a belief state.

    - Uses TrapdoorTracker for Bayesian trapdoor probabilities.
    - Uses iterative deepening alpha–beta (no explicit “killer move” rules yet).
    - Heuristic is state-based (no action-dependent bonuses yet).

    Heuristic components:
      * Egg advantage (scoreboard eggs, including corner/bonus effects)
      * Turd resource advantage (remaining turds)
      * Mobility advantage (valid moves difference)
      * Trapdoor risk penalty at our own location
    """

    # --------------------------- INIT --------------------------- #

    def __init__(self, board: Board, time_left: Callable[[], float]):
        # Board / game info
        self.map_size = board.game_map.MAP_SIZE

        # Search settings
        self.max_search_depth = 3      # iterative deepening up to this
        self.global_safety_margin = 0.2 # leave a bit of time unspent overall
        self.max_per_move_time = 4.0    # hard cap per move in seconds

        # Heuristic weights (can tune later)
        self.w_egg_adv = 10.0      # eggs are primary win condition
        self.w_turd_res = 0.5      # turds are valuable but secondary
        self.w_mobility = 1.0      # mobility helps avoid traps / blocking
        self.w_risk = 15.0         # penalty per unit trapdoor probability

        # Trapdoor belief tracker
        self.tracker = TrapdoorTracker(self.map_size)

        # Move counter (optional; might be useful later)
        self.turn_index = 0

    # --------------------- PERSPECTIVE HELPERS --------------------- #

    def _get_root_perspective_values(
        self,
        board: Board,
        root_is_player: bool,
    ):
        """
        Return eggs and turds from the root player's perspective,
        regardless of how Board has been reversed.

        root_is_player == True means:
          - board.chicken_player is the root's chicken.
        root_is_player == False means:
          - board.chicken_enemy is the root's chicken.
        """

        if root_is_player:
            my_chicken = board.chicken_player
            opp_chicken = board.chicken_enemy
            my_eggs_set = board.eggs_player
            opp_eggs_set = board.eggs_enemy
            my_turds_set = board.turds_player
            opp_turds_set = board.turds_enemy
        else:
            my_chicken = board.chicken_enemy
            opp_chicken = board.chicken_player
            my_eggs_set = board.eggs_enemy
            opp_eggs_set = board.eggs_player
            my_turds_set = board.turds_enemy
            opp_turds_set = board.turds_player

        my_eggs_score = my_chicken.get_eggs_laid()
        opp_eggs_score = opp_chicken.get_eggs_laid()
        my_turds_left = my_chicken.get_turds_left()
        opp_turds_left = opp_chicken.get_turds_left()

        return (
            my_eggs_score,
            opp_eggs_score,
            my_turds_left,
            opp_turds_left,
            my_eggs_set,
            opp_eggs_set,
            my_turds_set,
            opp_turds_set,
        )

    def _root_location(self, board: Board, root_is_player: bool):
        """
        Current location of the root player's chicken in this board orientation.
        """
        if root_is_player:
            return board.chicken_player.get_location()
        else:
            return board.chicken_enemy.get_location()

    # --------------------- HEURISTIC COMPONENTS --------------------- #

    def _egg_advantage(self, board: Board, root_is_player: bool) -> float:
        my_eggs, opp_eggs, _, _, _, _, _, _ = \
            self._get_root_perspective_values(board, root_is_player)
        return float(my_eggs - opp_eggs)

    def _turd_resource_advantage(self, board: Board, root_is_player: bool) -> float:
        _, _, my_turds_left, opp_turds_left, _, _, _, _ = \
            self._get_root_perspective_values(board, root_is_player)
        return float(my_turds_left - opp_turds_left)

    def _mobility_advantage(self, board: Board, root_is_player: bool) -> float:
        """
        Mobility is measured as:
          (root's number of moves) - (opponent's number of moves)
        using get_valid_moves(enemy=...) without mutating perspective.
        """
        if root_is_player:
            my_moves = len(board.get_valid_moves(enemy=False))
            opp_moves = len(board.get_valid_moves(enemy=True))
        else:
            # In this orientation, "player" is actually the opponent of the root.
            my_moves = len(board.get_valid_moves(enemy=True))
            opp_moves = len(board.get_valid_moves(enemy=False))

        return float(my_moves - opp_moves)

    def _trapdoor_risk_penalty(self, board: Board, root_is_player: bool) -> float:
        """
        Penalize being on squares with high trapdoor probability.
        """
        loc = self._root_location(board, root_is_player)
        risk = self.tracker.get_probability(loc)
        return float(risk)

    # --------------------- MAIN EVALUATION FUNCTION --------------------- #

    def _evaluate(self, board: Board, root_is_player: bool) -> float:
        """
        Evaluate the board from the root player's perspective.

        Positive = good for root player.
        Negative = good for the opponent.
        """
        # Game over? Give huge win/loss scores.
        if board.is_game_over():
            winner = board.get_winner()
            if winner == Result.PLAYER:
                # In this orientation, "PLAYER" = board.chicken_player.
                # If root_is_player, root wins; else root loses.
                return 1e6 if root_is_player else -1e6
            elif winner == Result.ENEMY:
                return -1e6 if root_is_player else 1e6
            else:
                return 0.0

        score = 0.0

        # 1. Eggs (scoreboard)
        egg_adv = self._egg_advantage(board, root_is_player)
        score += self.w_egg_adv * egg_adv

        # 2. Turd resources
        turd_res = self._turd_resource_advantage(board, root_is_player)
        score += self.w_turd_res * turd_res

        # 3. Mobility
        mobility_adv = self._mobility_advantage(board, root_is_player)
        score += self.w_mobility * mobility_adv

        # 4. Trapdoor risk at root location
        risk = self._trapdoor_risk_penalty(board, root_is_player)
        score -= self.w_risk * risk

        return score

    # -------------------------- SEARCH CORE -------------------------- #

    def _order_moves(
        self,
        board: Board,
        moves: List[Tuple[Direction, MoveType]],
    ) -> List[Tuple[Direction, MoveType]]:
        """
        Simple move ordering heuristic *for the side to move*:

          - Prefer EGG moves.
          - Among PLAIN/TURD moves, prefer lower trap risk at destination.

        IMPORTANT: moves are always for board.chicken_player (the side to move),
        so we must start from board.chicken_player.get_location(), not from
        the root player's location.
        """
        ordered = []
        # Side-to-move location
        x0, y0 = board.chicken_player.get_location()

        for d, mt in moves:
            # Compute destination from the current player's location
            if d == Direction.UP:
                new_loc = (x0, y0 - 1)
            elif d == Direction.DOWN:
                new_loc = (x0, y0 + 1)
            elif d == Direction.LEFT:
                new_loc = (x0 - 1, y0)
            else:  # Direction.RIGHT
                new_loc = (x0 + 1, y0)

            # Safety check (should always be true for valid moves)
            x, y = new_loc
            if not (0 <= x < self.map_size and 0 <= y < self.map_size):
                # Treat off-board as very risky; but in practice this shouldn't happen
                risk = 1.0
            else:
                risk = self.tracker.get_probability(new_loc)

            # egg moves first, then plain, then turd (arbitrary order),
            # with risk as tiebreaker.
            if mt == MoveType.EGG:
                move_type_rank = 0
            elif mt == MoveType.PLAIN:
                move_type_rank = 1
            else:  # TURD
                move_type_rank = 2

            key = (move_type_rank, risk)
            ordered.append((key, (d, mt)))

        ordered.sort(key=lambda x: x[0])
        return [m for _, m in ordered]

    def _alpha_beta(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        root_is_player: bool,
        time_remaining: Callable[[], float],
    ) -> float:
        """
        Standard alpha–beta search.
        board: current node state (orientation as given)
        maximizing: True if this node is the root player to move; False otherwise.
        root_is_player: True if board.chicken_player is currently the root side.
        """
        # Time check
        if time_remaining() <= 0.0:
            return self._evaluate(board, root_is_player)

        # Depth or terminal
        if depth == 0 or board.is_game_over():
            return self._evaluate(board, root_is_player)

        moves = board.get_valid_moves()
        if not moves:
            # No moves: just evaluate; the real +5 eggs penalty is handled
            # in the real game logic; we don't simulate it here.
            return self._evaluate(board, root_is_player)

        moves = self._order_moves(board, moves)

        if maximizing:
            value = -float("inf")
            for move in moves:
                if time_remaining() <= 0.0:
                    break

                child = board.forecast_move(*move)
                if child is None:
                    continue

                # After a real move, the engine reverses perspective.
                child.reverse_perspective()

                # After reversing, the "player" side is the opponent of previous board.
                # So root_is_player flips:
                child_value = self._alpha_beta(
                    child,
                    depth - 1,
                    alpha,
                    beta,
                    maximizing=False,
                    root_is_player=not root_is_player,
                    time_remaining=time_remaining,
                )
                value = max(value, child_value)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break

            if value == -float("inf"):
                return self._evaluate(board, root_is_player)
            return value

        else:
            value = float("inf")
            for move in moves:
                if time_remaining() <= 0.0:
                    break

                child = board.forecast_move(*move)
                if child is None:
                    continue

                child.reverse_perspective()

                child_value = self._alpha_beta(
                    child,
                    depth - 1,
                    alpha,
                    beta,
                    maximizing=True,
                    root_is_player=not root_is_player,
                    time_remaining=time_remaining,
                )
                value = min(value, child_value)
                beta = min(beta, value)
                if beta <= alpha:
                    break

            if value == float("inf"):
                return self._evaluate(board, root_is_player)
            return value

    def _iterative_deepening(
        self,
        board: Board,
        time_remaining: Callable[[], float],
    ) -> Tuple[Direction, MoveType]:
        moves = board.get_valid_moves()
        if not moves:
            return (Direction.UP, MoveType.PLAIN)

        # Hard constraint: our real move choices must avoid known trapdoors
        moves = self._filter_trapdoor_moves(board, moves)

        best_move = moves[0]
        best_value = -float("inf")

        root_is_player = True

        for depth in range(1, self.max_search_depth + 1):
            if time_remaining() <= 0.0:
                break

            current_best = None
            current_best_value = -float("inf")

            ordered_moves = self._order_moves(board, moves)

            for move in ordered_moves:
                if time_remaining() <= 0.0:
                    break

                child = board.forecast_move(*move)
                if child is None:
                    continue

                child.reverse_perspective()

                value = self._alpha_beta(
                    child,
                    depth=depth - 1,
                    alpha=-float("inf"),
                    beta=float("inf"),
                    maximizing=False,
                    root_is_player=False,
                    time_remaining=time_remaining,
                )

                if value > current_best_value:
                    current_best_value = value
                    current_best = move

            if current_best is not None:
                best_move = current_best
                best_value = current_best_value

        return best_move

    def _filter_trapdoor_moves(
        self,
        board: Board,
        moves: List[Tuple[Direction, MoveType]],
    ) -> List[Tuple[Direction, MoveType]]:
        """
        Return only those moves that do NOT land on a discovered trapdoor.

        If filtering would remove all moves, returns the original list to avoid
        getting stuck or crashing.
        """
        if not board.found_trapdoors or not moves:
            return moves

        safe_moves: List[Tuple[Direction, MoveType]] = []

        # The side to move is always chicken_player
        x0, y0 = board.chicken_player.get_location()
        found = board.found_trapdoors

        for d, mt in moves:
            # Compute destination from current player's location
            if d == Direction.UP:
                new_loc = (x0, y0 - 1)
            elif d == Direction.DOWN:
                new_loc = (x0, y0 + 1)
            elif d == Direction.LEFT:
                new_loc = (x0 - 1, y0)
            else:  # Direction.RIGHT
                new_loc = (x0 + 1, y0)

            if new_loc in found:
                # This move would step onto a known trapdoor – skip it
                continue

            safe_moves.append((d, mt))

        # If everything got filtered out, fall back to original moves
        return safe_moves if safe_moves else moves

    # --------------------------- PLAY API --------------------------- #

    def play(
        self,
        board: Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable[[], float],
    ) -> Tuple[Direction, MoveType]:
        """
        Called by the engine each turn.

        - Updates trapdoor beliefs based on sensor_data and found_trapdoors.
        - Allocates a per-move time budget based on remaining global time.
        - Runs iterative deepening alpha–beta within that budget.
        """
        self.turn_index += 1

        # Update trapdoor beliefs
        my_loc = board.chicken_player.get_location()
        self.tracker.update_with_found_trapdoors(board.found_trapdoors)
        self.tracker.update_with_sensors(my_loc, sensor_data)

        # Global remaining time (for the whole match)
        global_remaining = time_left()

        # If we're basically out of time, just play any legal move.
        moves = board.get_valid_moves()
        if global_remaining <= self.global_safety_margin or not moves:
            return moves[0] if moves else (Direction.UP, MoveType.PLAIN)

        # Estimate per-move budget using remaining turns
        turns_left = max(1, board.turns_left_player)
        # Keep a small reserve of global time
        safe_global = max(0.0, global_remaining - self.global_safety_margin)
        per_move_budget = min(self.max_per_move_time, safe_global / turns_left)

        start = time.perf_counter()
        deadline = start + per_move_budget

        def time_remaining():
            return deadline - time.perf_counter()

        # Run search within budget
        best_move = self._iterative_deepening(board, time_remaining)

        # Final safety check: if somehow best_move is invalid, fall back
        if best_move not in moves:
            return moves[0]

        return best_move
