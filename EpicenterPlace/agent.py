from collections.abc import Callable
from typing import List, Tuple

import time

from game.board import Board
from game.enums import Direction, MoveType, Result
from .trapdoor_tracker import TrapdoorTracker


class PlayerAgent:
    """
    Epicenter: Eggs-first alpha–beta agent with:
      - Egg reachability heuristic
      - Early-game centrality
      - Revisit penalty to avoid wasting moves
      - Bayesian trapdoor tracking
      - Hard constraint: never step on discovered trapdoors at the root
      - Reduced action space: if we can lay an egg, we *only* consider egg moves
      - Far-turd penalty: discourage dropping turds far away from the opponent

    Turd economy scoring is disabled (w_turd_eco = 0).
    """

    # --------------------------- INIT --------------------------- #

    def __init__(self, board: Board, time_left: Callable[[], float]):
        self.map_size = board.game_map.MAP_SIZE
        self.max_turds = board.game_map.MAX_TURDS  # usually 5

        # Search settings
        self.base_search_depth = 3      # starting depth
        self.max_search_depth = 3       # will be updated dynamically each turn
        self.hard_max_search_depth = 3
        self.global_safety_margin = 0.2  # leave this many seconds unspent in total
        self.max_per_move_time = 8.0     # hard cap per move in seconds
        self.trapdoor_prune_threshold = 0.7

        # Heuristic weights
        self.w_egg_adv = 10.0        # eggs are the main score
        self.w_turd_eco = 0.0        # turd econ disabled
        self.w_mobility = 1.0        # mobility helps avoid traps / blocking
        self.w_risk = 200.0           # penalty per unit trapdoor probability at our location
        self.w_center = 0.0          # early-game centrality advantage
        self.w_revisit = 2.0         # penalty for revisiting old real-game cells
        self.w_reach = 2.0           # egg reachability advantage
        self.w_far_turd = 1.5        # penalty weight for far-away turds

        # Trapdoor belief tracker
        self.tracker = TrapdoorTracker(self.map_size)

        # Track real-game cells we've actually visited (root player's path)
        self.visited_real: set[Tuple[int, int]] = set()

        self.turn_index = 0
    
    def _update_search_depth(self, board: Board, global_remaining: float):
        """
        Dynamically adjust search depth over the game:

        - Start at depth 3.
        - Gradually increase toward 7 as the game approaches the end.
        - If we're very low on time, cap the depth to avoid timeouts.
        """
        # How far along the game are we? 0.0 at start, 1.0 near end.
        total_turns = 2 * board.MAX_TURNS  # both players
        progress = board.turn_count / max(1, total_turns)  # 0..1

        # Target depth from 3 up to 7 as the game progresses
        target = self.base_search_depth + int(
            progress * (self.hard_max_search_depth - self.base_search_depth)
        )

        # Clamp between base and hard max
        target = max(self.base_search_depth, min(self.hard_max_search_depth, target))

        self.max_search_depth = target

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

    def _opp_location(self, board: Board, root_is_player: bool):
        if root_is_player:
            return board.chicken_enemy.get_location()
        else:
            return board.chicken_player.get_location()

    # --------------------- HEURISTIC COMPONENTS --------------------- #

    def _is_corner(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return (x == 0 or x == self.map_size - 1) and \
               (y == 0 or y == self.map_size - 1)

    def _egg_advantage(self, board: Board, root_is_player: bool) -> float:
        """
        Egg scoreboard difference: my_eggs - opp_eggs
        (Corner bonus is already baked into eggs_laid by the engine.)
        """
        my_eggs, opp_eggs, _, _, _, _, _, _ = \
            self._get_root_perspective_values(board, root_is_player)
        return float(my_eggs - opp_eggs)

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
        """
        Risk penalty at the root side's current location.
        """
        loc = self._root_location(board, root_is_player)
        risk = self.tracker.get_probability(loc)
        return float(risk)

    def _centrality_advantage(self, board: Board, root_is_player: bool) -> float:
        """
        Who is more central?
        Only active for the first 10 total moves (board.turn_count < 10).
        """
        if board.turn_count >= 10:
            return 0.0

        cx = (self.map_size - 1) / 2.0
        cy = (self.map_size - 1) / 2.0

        if root_is_player:
            my_loc = board.chicken_player.get_location()
            opp_loc = board.chicken_enemy.get_location()
        else:
            my_loc = board.chicken_enemy.get_location()
            opp_loc = board.chicken_player.get_location()

        def manhattan_to_center(loc: Tuple[int, int]) -> float:
            x, y = loc
            return abs(x - cx) + abs(y - cy)

        my_dist = manhattan_to_center(my_loc)
        opp_dist = manhattan_to_center(opp_loc)

        return float(opp_dist - my_dist)

    def _revisit_penalty(self, board: Board, root_is_player: bool) -> float:
        """
        Penalize revisiting cells we've already visited in the *real* game,
        to discourage wasting moves bouncing around old territory.
        Only applied when root_is_player is True.
        """
        if not root_is_player:
            return 0.0

        loc = board.chicken_player.get_location()
        if loc in self.visited_real:
            return self.w_revisit
        return 0.0

    # ---------- Egg reachability: BFS-based egg potential heuristic ---------- #

    def _reachable_egg_potential(
        self,
        start_loc: Tuple[int, int],
        my_parity: int,
        my_eggs_set: set[Tuple[int, int]],
        opp_eggs_set: set[Tuple[int, int]],
        my_turds_set: set[Tuple[int, int]],
        opp_turds_set: set[Tuple[int, int]],
        opp_loc: Tuple[int, int],
    ) -> float:
        """
        From start_loc, compute how much egg-parity territory is reachable.
        """
        from collections import deque

        # Precompute opponent turd blocked cells: turd squares + their 4 neighbors
        opp_turd_blocked = set()
        for tx, ty in opp_turds_set:
            for dx, dy in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = tx + dx, ty + dy
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    opp_turd_blocked.add((nx, ny))

        q = deque()
        visited = set()

        q.append((start_loc, 0))
        visited.add(start_loc)

        potential = 0.0

        while q:
            (x, y), dist = q.popleft()

            # Egg potential at this cell
            if (x + y) % 2 == my_parity:
                if (x, y) not in my_eggs_set and \
                   (x, y) not in opp_eggs_set and \
                   (x, y) not in my_turds_set and \
                   (x, y) not in opp_turds_set:
                    base = 3.0 if self._is_corner((x, y)) else 1.0
                    potential += base / (1.0 + dist)

            # Explore neighbors
            for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                if d == Direction.UP:
                    nx, ny = x, y - 1
                elif d == Direction.DOWN:
                    nx, ny = x, y + 1
                elif d == Direction.LEFT:
                    nx, ny = x - 1, y
                else:
                    nx, ny = x + 1, y

                if not (0 <= nx < self.map_size and 0 <= ny < self.map_size):
                    continue
                if (nx, ny) in visited:
                    continue

                # Can't walk into opponent chicken
                if (nx, ny) == opp_loc:
                    continue

                # Can't walk into opponent eggs
                if (nx, ny) in opp_eggs_set:
                    continue

                # Can't walk into opponent turd aura
                if (nx, ny) in opp_turd_blocked:
                    continue

                visited.add((nx, ny))
                q.append(((nx, ny), dist + 1))

        return potential

    def _egg_reachability_advantage(self, board: Board, root_is_player: bool) -> float:
        """
        Egg reachability heuristic:

        Score = my_reachability_potential - opp_reachability_potential
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

        my_loc = self._root_location(board, root_is_player)
        opp_loc = self._opp_location(board, root_is_player)

        my_potential = self._reachable_egg_potential(
            start_loc=my_loc,
            my_parity=my_parity,
            my_eggs_set=my_eggs_set,
            opp_eggs_set=opp_eggs_set,
            my_turds_set=my_turds_set,
            opp_turds_set=opp_turds_set,
            opp_loc=opp_loc,
        )

        opp_potential = self._reachable_egg_potential(
            start_loc=opp_loc,
            my_parity=opp_parity,
            my_eggs_set=opp_eggs_set,
            opp_eggs_set=my_eggs_set,
            my_turds_set=opp_turds_set,
            opp_turds_set=my_turds_set,
            opp_loc=my_loc,
        )

        return my_potential - opp_potential

    # ---------- Far-turd penalty: turds far from opponent are wasteful ---------- #

    def _far_turd_penalty(self, board: Board, root_is_player: bool) -> float:
        """
        Penalize having turds that are "far" from the opponent:

        For each of my turds at (tx, ty), compute
            d = max(|tx - ox|, |ty - oy|)
        where (ox, oy) is the opponent location.
        If d > 4, add 1 to the penalty.

        The final penalty is scaled by self.w_far_turd in _evaluate.
        """
        (
            _my_eggs_score,
            _opp_eggs_score,
            _my_turds_left,
            _opp_turds_left,
            _my_eggs_set,
            _opp_eggs_set,
            my_turds_set,
            _opp_turds_set,
        ) = self._get_root_perspective_values(board, root_is_player)

        opp_loc = self._opp_location(board, root_is_player)
        ox, oy = opp_loc

        penalty = 0.0
        for tx, ty in my_turds_set:
            d = max(abs(tx - ox), abs(ty - oy))
            if d > 4:
                penalty += 1.0

        return penalty

    # --------------------- MAIN EVALUATION FUNCTION --------------------- #

    def _evaluate(self, board: Board, root_is_player: bool) -> float:
        """
        Evaluate the board from the root player's perspective.
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

        # 1. Eggs – main driver
        score += self.w_egg_adv * self._egg_advantage(board, root_is_player)

        # 2. Mobility advantage
        score += self.w_mobility * self._mobility_advantage(board, root_is_player)

        # 3. Centrality advantage (only first 10 moves)
        score += self.w_center * self._centrality_advantage(board, root_is_player)

        # 4. Egg reachability advantage
        score += self.w_reach * self._egg_reachability_advantage(board, root_is_player)

        # 5. Trapdoor risk penalty
        risk = self._trapdoor_risk_penalty(board, root_is_player)
        score -= self.w_risk * risk

        # 6. Revisit penalty (only when root player is at this location)
        score -= self._revisit_penalty(board, root_is_player)

        # 7. Far-turd penalty
        score -= self.w_far_turd * self._far_turd_penalty(board, root_is_player)

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
        Hard-ish constraint at the root:

        Return only those moves that:
          - do NOT land on a discovered trapdoor, and
          - do NOT land on a square whose trapdoor probability > threshold.

        If filtering would remove all moves, return the original list so we never
        get completely stuck; sometimes you *must* walk through danger.
        """
        if not moves:
            return moves

        safe_moves: List[Tuple[Direction, MoveType]] = []

        x0, y0 = board.chicken_player.get_location()
        found = board.found_trapdoors

        for d, mt in moves:
            # Compute destination
            if d == Direction.UP:
                new_loc = (x0, y0 - 1)
            elif d == Direction.DOWN:
                new_loc = (x0, y0 + 1)
            elif d == Direction.LEFT:
                new_loc = (x0 - 1, y0)
            else:  # Direction.RIGHT
                new_loc = (x0 + 1, y0)

            # 1) Never step on a known trapdoor
            if new_loc in found:
                continue

            # 2) Avoid very high trapdoor probability from tracker
            risk = self.tracker.get_probability(new_loc)
            if risk > self.trapdoor_prune_threshold:
                # Too risky, skip this move
                continue

            safe_moves.append((d, mt))

        # If everything got filtered out, we have to consider all moves;
        # better to risk stepping on a bad square than crash / pass.
        return safe_moves if safe_moves else moves

    def _prune_non_egg_if_possible(
        self,
        board: Board,
        moves: List[Tuple[Direction, MoveType]],
    ) -> List[Tuple[Direction, MoveType]]:
        """
        If this square can lay an egg, we restrict actions to EGG moves only.
        """
        if board.can_lay_egg():
            egg_moves = [m for m in moves if m[1] == MoveType.EGG]
            if egg_moves:
                return egg_moves
        return moves

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
        """
        if time_remaining() <= 0.0:
            return self._evaluate(board, root_is_player)

        if depth == 0 or board.is_game_over():
            return self._evaluate(board, root_is_player)

        moves = board.get_valid_moves()
        if not moves:
            return self._evaluate(board, root_is_player)

        # If we can lay an egg, only consider egg moves at this node
        moves = self._prune_non_egg_if_possible(board, moves)
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

        # Only egg moves if egg is possible here
        moves = self._prune_non_egg_if_possible(board, moves)

        # Root-level hard filter: don't step on discovered trapdoors.
        moves = self._filter_trapdoor_moves(board, moves)

        best_move = moves[0]
        best_value = -float("inf")

        root_is_player = True  # at the root, we are "player" side

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
                    depth=depth - 1,        # we already used 1 ply at root
                    alpha=-float("inf"),
                    beta=float("inf"),
                    maximizing=False,       # opponent to move
                    root_is_player=False,   # board.chicken_player is now opponent
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
        opp_loc = board.chicken_enemy.get_location()
        # self.tracker.update_with_found_trapdoors(board.found_trapdoors)
        # self.tracker.update_with_sensors(my_loc, sensor_data)
        self.tracker.step(
        my_pos=my_loc,
            opp_pos=opp_loc,
            sensor_data=sensor_data,
            eggs_player=board.eggs_player,
            eggs_enemy=board.eggs_enemy,
            turds_player=board.turds_player,
            turds_enemy=board.turds_enemy,
            found_trapdoors=board.found_trapdoors,
        )

        # Record this real-game location as visited (for revisit penalty)
        self.visited_real.add(my_loc)

        global_remaining = time_left()

        self._update_search_depth(board, global_remaining)

        moves = board.get_valid_moves()
        if not moves:
            return (Direction.UP, MoveType.PLAIN)

        # If we can lay an egg, only consider egg moves
        moves = self._prune_non_egg_if_possible(board, moves)

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

        # self.tracker.debug_print(precision=2)

        # Final safety: if for some reason best_move isn't in the current legal set,
        # fall back to something valid.
        if best_move not in moves:
            return moves[0]

        return best_move
