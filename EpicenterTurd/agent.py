from collections.abc import Callable
from typing import List, Tuple, Optional

import time

from game.board import Board
from game.enums import Direction, MoveType, Result
from .trapdoor_tracker import TrapdoorTracker


class PlayerAgent:
    """
    Epicenter V2: Risk-aware alpha–beta agent with turd economics.

    - Still uses TrapdoorTracker for Bayesian trapdoor probabilities.
    - Still uses iterative deepening alpha–beta search.
    - Hard constraint: never step on a discovered trapdoor at the root.
    - Heuristic now includes:
        * Egg advantage (scoreboard)
        * Turd resource advantage (remaining turds)
        * Turd economic advantage (how many potential opponent egg squares
          our turds deny vs how many of ours theirs deny)
        * Mobility advantage (valid moves difference)
        * Trapdoor risk at our own location
    """

    # --------------------------- INIT --------------------------- #

    def __init__(self, board: Board, time_left: Callable[[], float]):
        self.map_size = board.game_map.MAP_SIZE

        # Search settings
        self.max_search_depth = 3        # depth for iterative deepening
        self.global_safety_margin = 0.2  # leave a bit of time unspent
        self.max_per_move_time = 4.0     # max time per move in seconds

        # Heuristic weights (tuneable)
        self.w_egg_adv = 10.0       # eggs are the win condition
        self.w_turd_res = 0.5       # keep some turds in reserve
        self.w_turd_eco = 2.0       # value of economic damage from turds
        self.w_mobility = 1.0       # having more moves is good
        self.w_risk = 15.0          # penalty per unit trapdoor probability

        # Trapdoor belief tracker
        self.tracker = TrapdoorTracker(self.map_size)

        self.turn_index = 0

    # --------------------- PERSPECTIVE HELPERS --------------------- #

    def _get_root_perspective_values(
        self,
        board: Board,
        root_is_player: bool,
    ):
        """
        Return eggs, turds, and sets from the root player's perspective,
        regardless of how Board has been reversed.
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

    def _get_root_parities(self, board: Board, root_is_player: bool) -> Tuple[int, int]:
        """
        Return (my_parity, opp_parity) from root's perspective.
        even_chicken is 0 for A, 1 for B (i+j%2).
        """
        if root_is_player:
            my_chicken = board.chicken_player
            opp_chicken = board.chicken_enemy
        else:
            my_chicken = board.chicken_enemy
            opp_chicken = board.chicken_player

        return my_chicken.even_chicken, opp_chicken.even_chicken

    def _root_location(self, board: Board, root_is_player: bool):
        if root_is_player:
            return board.chicken_player.get_location()
        else:
            return board.chicken_enemy.get_location()

    # --------------------- HEURISTIC COMPONENTS --------------------- #

    def _is_corner(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return (x == 0 or x == self.map_size - 1) and \
               (y == 0 or y == self.map_size - 1)

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
        Mobility = (#root moves) - (#opponent moves)
        using get_valid_moves(enemy=...) without mutating perspective.
        """
        if root_is_player:
            my_moves = len(board.get_valid_moves(enemy=False))
            opp_moves = len(board.get_valid_moves(enemy=True))
        else:
            my_moves = len(board.get_valid_moves(enemy=True))
            opp_moves = len(board.get_valid_moves(enemy=False))

        return float(my_moves - opp_moves)

    def _trapdoor_risk_penalty(self, board: Board, root_is_player: bool) -> float:
        loc = self._root_location(board, root_is_player)
        risk = self.tracker.get_probability(loc)
        return float(risk)

    def _turd_economic_advantage(self, board: Board, root_is_player: bool) -> float:
        """
        Turd economic advantage = (damage from my turds) - (damage from opp turds),
        where "damage" is how many potential egg squares get blocked,
        with corners valued more.
        """
        (
            _my_eggs_score,
            _opp_eggs_score,
            _my_turds_left,
            _opp_turds_left,
            my_eggs_set,
            opp_eggs_set,
            my_turds_set,
            opp_turds_set,
        ) = self._get_root_perspective_values(board, root_is_player)

        my_parity, opp_parity = self._get_root_parities(board, root_is_player)

        # Precompute occupied squares (no future eggs here)
        occupied = my_eggs_set | opp_eggs_set | my_turds_set | opp_turds_set

        def aura_squares(turd_loc: Tuple[int, int]):
            x, y = turd_loc
            for dx, dy in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    yield (nx, ny)

        my_damage = 0.0
        opp_damage = 0.0

        # My turds blocking opponent egg squares
        for t in my_turds_set:
            for sq in aura_squares(t):
                if sq in occupied:
                    continue
                x, y = sq
                if (x + y) % 2 != opp_parity:
                    continue
                # Potential opponent egg square
                if self._is_corner(sq):
                    my_damage += 3.0  # corner worth 3 eggs
                else:
                    my_damage += 1.0  # normal egg square

        # Opponent turds blocking my egg squares
        for t in opp_turds_set:
            for sq in aura_squares(t):
                if sq in occupied:
                    continue
                x, y = sq
                if (x + y) % 2 != my_parity:
                    continue
                if self._is_corner(sq):
                    opp_damage += 3.0
                else:
                    opp_damage += 1.0

        return my_damage - opp_damage

    # --------------------- MAIN EVALUATION FUNCTION --------------------- #

    def _evaluate(self, board: Board, root_is_player: bool) -> float:
        """
        Evaluate the board from the root player's perspective.

        Positive = good for root player.
        Negative = good for the opponent.
        """
        # Game over? Massive win/loss reward.
        if board.is_game_over():
            winner = board.get_winner()
            if winner == Result.PLAYER:
                return 1e6 if root_is_player else -1e6
            elif winner == Result.ENEMY:
                return -1e6 if root_is_player else 1e6
            else:
                return 0.0

        score = 0.0

        # 1. Eggs
        score += self.w_egg_adv * self._egg_advantage(board, root_is_player)

        # 2. Turd resource advantage
        score += self.w_turd_res * self._turd_resource_advantage(board, root_is_player)

        # 3. Turd economic advantage
        score += self.w_turd_eco * self._turd_economic_advantage(board, root_is_player)

        # 4. Mobility advantage
        score += self.w_mobility * self._mobility_advantage(board, root_is_player)

        # 5. Trapdoor risk penalty
        risk = self._trapdoor_risk_penalty(board, root_is_player)
        score -= self.w_risk * risk

        return score

    # -------------------------- MOVE HELPERS -------------------------- #

    def _order_moves(
        self,
        board: Board,
        moves: List[Tuple[Direction, MoveType]],
    ) -> List[Tuple[Direction, MoveType]]:
        """
        Move ordering for the *side to move* (always chicken_player):

          - Prefer EGG moves.
          - Then TURD moves.
          - Then PLAIN moves.
          - Within each type, prefer lower trapdoor risk at destination.
        """
        ordered = []
        x0, y0 = board.chicken_player.get_location()

        for d, mt in moves:
            # Destination from current player's location
            if d == Direction.UP:
                new_loc = (x0, y0 - 1)
            elif d == Direction.DOWN:
                new_loc = (x0, y0 + 1)
            elif d == Direction.LEFT:
                new_loc = (x0 - 1, y0)
            else:  # RIGHT
                new_loc = (x0 + 1, y0)

            x, y = new_loc
            if not (0 <= x < self.map_size and 0 <= y < self.map_size):
                risk = 1.0
            else:
                risk = self.tracker.get_probability(new_loc)

            # Order: EGG (0), TURD (1), PLAIN (2)
            if mt == MoveType.EGG:
                move_type_rank = 0
            elif mt == MoveType.TURD:
                move_type_rank = 1
            else:
                move_type_rank = 2

            key = (move_type_rank, risk)
            ordered.append((key, (d, mt)))

        ordered.sort(key=lambda x: x[0])
        return [m for _, m in ordered]

    def _filter_trapdoor_moves(
        self,
        board: Board,
        moves: List[Tuple[Direction, MoveType]],
    ) -> List[Tuple[Direction, MoveType]]:
        """
        Hard constraint at the root:

        Return only those moves that do NOT land on a discovered trapdoor.

        If filtering would remove all moves, return the original list.
        """
        if not board.found_trapdoors or not moves:
            return moves

        safe_moves: List[Tuple[Direction, MoveType]] = []

        x0, y0 = board.chicken_player.get_location()
        found = board.found_trapdoors

        for d, mt in moves:
            if d == Direction.UP:
                new_loc = (x0, y0 - 1)
            elif d == Direction.DOWN:
                new_loc = (x0, y0 + 1)
            elif d == Direction.LEFT:
                new_loc = (x0 - 1, y0)
            else:
                new_loc = (x0 + 1, y0)

            if new_loc in found:
                continue

            safe_moves.append((d, mt))

        return safe_moves if safe_moves else moves

    # -------------------------- SEARCH CORE -------------------------- #

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

        maximizing: True if it's the root player's turn in this node,
                    False if it's the opponent's turn.
        root_is_player: True if board.chicken_player is the root side now.
        """
        if time_remaining() <= 0.0:
            return self._evaluate(board, root_is_player)

        if depth == 0 or board.is_game_over():
            return self._evaluate(board, root_is_player)

        moves = board.get_valid_moves()
        if not moves:
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

                child.reverse_perspective()

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
        """
        Root search with iterative deepening.
        """
        moves = board.get_valid_moves()
        if not moves:
            return (Direction.UP, MoveType.PLAIN)

        # Root-level hard filter: don't step on discovered trapdoors.
        moves = self._filter_trapdoor_moves(board, moves)

        best_move = moves[0]
        best_value = -float("inf")

        root_is_player = True  # At the root, we are "player" side

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
                    maximizing=False,      # opponent to move
                    root_is_player=False,  # board.chicken_player is now opponent
                    time_remaining=time_remaining,
                )

                if value > current_best_value:
                    current_best_value = value
                    current_best = move

            if current_best is not None:
                best_move = current_best
                best_value = current_best_value

        return best_move

    # --------------------------- PLAY API --------------------------- #

    def play(
        self,
        board: Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable[[], float],
    ) -> Tuple[Direction, MoveType]:
        """
        Called by the engine each turn.
        """
        self.turn_index += 1

        # Update trapdoor beliefs
        my_loc = board.chicken_player.get_location()
        self.tracker.update_with_found_trapdoors(board.found_trapdoors)
        self.tracker.update_with_sensors(my_loc, sensor_data)

        global_remaining = time_left()

        moves = board.get_valid_moves()
        if not moves:
            return (Direction.UP, MoveType.PLAIN)

        # Even in panic mode, don't step on discovered trapdoors if avoidable
        moves = self._filter_trapdoor_moves(board, moves)

        if global_remaining <= self.global_safety_margin:
            return moves[0]

        # Per-move time budget
        turns_left = max(1, board.turns_left_player)
        safe_global = max(0.0, global_remaining - self.global_safety_margin)
        per_move_budget = min(self.max_per_move_time, safe_global / turns_left)

        start = time.perf_counter()
        deadline = start + per_move_budget

        def time_remaining():
            return deadline - time.perf_counter()

        best_move = self._iterative_deepening(board, time_remaining)

        # Safety: if for some reason best_move isn't in the current legal set,
        # fall back to something valid.
        if best_move not in moves:
            return moves[0]

        return best_move
