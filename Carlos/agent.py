# agent.py — Carlos, a Monte ("Marco") Carlos Tree Search agent

import math
import random
from collections.abc import Callable
from typing import List, Tuple, Optional

from game.board import Board
from game.enums import Direction, MoveType
from game.game_map import prob_hear, prob_feel  # <-- use these with sensor_data

# ------------------------------
# Helper: list all possible actions
# ------------------------------

ALL_DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
ALL_MOVES = [MoveType.PLAIN, MoveType.EGG, MoveType.TURD]


def all_actions() -> List[Tuple[Direction, MoveType]]:
    return [(d, m) for d in ALL_DIRECTIONS for m in ALL_MOVES]


# ------------------------------
# MCTS Node
# ------------------------------

class MCTSNode:
    def __init__(self, board: Board, parent: Optional["MCTSNode"], action: Optional[Tuple[Direction, MoveType]]):
        self.board = board
        self.parent = parent
        self.action = action  # action taken at parent to reach this node
        self.children: dict[Tuple[Direction, MoveType], "MCTSNode"] = {}
        self.N = 0  # visit count
        self.Q = 0.0  # accumulated value
        self.untried_actions: List[Tuple[Direction, MoveType]] = []

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        # Terminal from tree's POV = no legal actions
        return len(self.untried_actions) == 0 and len(self.children) == 0


# ------------------------------
# Carlos PlayerAgent
# ------------------------------

class PlayerAgent:
    """
    Carlos: Monte ("Marco") Carlos Tree Search agent with a heuristic that
    is a linear combination of egg differential and trapdoor risk.
    """

    def __init__(self, board: Board, time_left: Callable[[], float]):
        self.map_size = board.game_map.MAP_SIZE  # usually 8
        self.size = self.map_size

        # MCTS parameters
        self.C = 1.4               # exploration constant
        self.eps_rollout = 0.2     # ε for ε-greedy rollout policy
        self.max_rollout_depth = 12
        self.time_safety_margin = 0.03  # seconds we want to leave unused per move
        self.max_move_time = 0.20       # hard per-move cap (tune if needed)

        # Heuristic weights
        self.w_eggs = 1.0
        self.w_trap = 0.7

        # Trapdoor priors: index [parity][x, y]
        # parity 0 = white squares (i+j even), parity 1 = black squares (i+j odd)
        self.trap_priors = [
            self._init_trap_prior_for_parity(board.game_map, parity=0),
            self._init_trap_prior_for_parity(board.game_map, parity=1),
        ]

    # -------------- Trapdoor priors --------------

    def _init_trap_prior_for_parity(self, game_map, parity: int):
        """
        Prior over trapdoor locations for a given color parity.
        We approximate the generation described in the assignment:
        squares nearer to the center get higher non-zero weight,
        outer rings cannot contain trapdoors.
        Returns a 2D list [x][y] with probabilities summing to 1.
        """
        size = self.size
        weights = [[0.0 for _ in range(size)] for _ in range(size)]
        total = 0.0

        for x in range(size):
            for y in range(size):
                if (x + y) % 2 != parity:
                    continue
                # distance to nearest edge
                dist = min(x, y, size - 1 - x, size - 1 - y)
                # two outer rings weight 0, inner rings >0
                w = max(0, dist - 1)
                # If you want to exactly match trapdoor_manager.py,
                # you can read its implementation and mirror it here.
                if w > 0:
                    weights[x][y] = float(w)
                    total += float(w)

        if total == 0:
            # Fallback: uniform over correct-color squares
            for x in range(size):
                for y in range(size):
                    if (x + y) % 2 == parity:
                        weights[x][y] = 1.0
                        total += 1.0

        # Normalize
        for x in range(size):
            for y in range(size):
                weights[x][y] /= total

        return weights

    def _trap_prob_at(self, pos: Tuple[int, int]) -> float:
        x, y = pos
        parity = (x + y) % 2
        return self.trap_priors[parity][x][y]

    # -------------- Board helpers --------------

    def _my_pos(self, board: Board) -> Tuple[int, int]:
        return board.chicken_player.get_location()

    def _opp_pos(self, board: Board) -> Tuple[int, int]:
        return board.chicken_enemy.get_location()

    def _my_eggs(self, board: Board) -> int:
        return board.chicken_player.get_eggs_laid()

    def _opp_eggs(self, board: Board) -> int:
        return board.chicken_enemy.get_eggs_laid()

    def _my_turds_remaining(self, board: Board) -> int:
        return board.chicken_player.get_turds_left()

    def _legal_actions(self, board: Board) -> List[Tuple[Direction, MoveType]]:
        """
        Use the engine's legal-move generator.
        We assume board.get_valid_moves() returns a list of (Direction, MoveType) pairs.
        """
        return board.get_valid_moves()

    # -------------- Sensor / belief update --------------

    def _update_trap_beliefs(self, board: Board, sensor_data: List[Tuple[bool, bool]]):
        """
        Use sensor_data to update trapdoor priors via Bayes' rule.

        sensor_data is expected to be:
            [(heard_white, felt_white), (heard_black, felt_black)]
        where index 0 is about the white-square trapdoor (parity 0),
        and index 1 is about the black-square trapdoor (parity 1).
        """
        if not sensor_data or len(sensor_data) < 2:
            return

        my_x, my_y = self._my_pos(board)
        size = self.size

        for parity in (0, 1):
            heard, felt = sensor_data[parity]

            prior = self.trap_priors[parity]
            posterior = [[0.0 for _ in range(size)] for _ in range(size)]
            total = 0.0

            for x in range(size):
                for y in range(size):
                    if (x + y) % 2 != parity:
                        continue
                    p0 = prior[x][y]
                    if p0 == 0.0:
                        continue

                    # distance from us to this candidate trap location
                    delta_x = abs(x - my_x)
                    delta_y = abs(y - my_y)

                    p_h = prob_hear(delta_x, delta_y)
                    p_f = prob_feel(delta_x, delta_y)

                    # Likelihood of the observed (heard, felt) given trap at (x, y)
                    lh = (p_h if heard else (1.0 - p_h)) * (p_f if felt else (1.0 - p_f))

                    val = p0 * lh
                    posterior[x][y] = val
                    total += val

            # If the update wiped everything out numerically, keep the old prior
            if total <= 0.0:
                continue

            # Normalize
            for x in range(size):
                for y in range(size):
                    if (x + y) % 2 == parity:
                        posterior[x][y] /= total

            self.trap_priors[parity] = posterior

    # -------------- Heuristic --------------

    def _heuristic(self, board: Board) -> float:
        """
        Linear combination of eggs and trapdoor risk.
        Higher is better for Carlos.
        """
        E_me = self._my_eggs(board)
        E_opp = self._opp_eggs(board)
        egg_feature = E_me - E_opp

        my_pos = self._my_pos(board)
        opp_pos = self._opp_pos(board)

        p_me = self._trap_prob_at(my_pos)
        p_opp = self._trap_prob_at(opp_pos)

        # 4-egg swing when someone hits a trapdoor
        trap_feature = 4.0 * (p_opp - p_me)

        value = self.w_eggs * egg_feature + self.w_trap * trap_feature
        return value

    # -------------- MCTS core --------------

    def _select_child_ucb1(self, node: MCTSNode) -> MCTSNode:
        best_child = None
        best_score = -float("inf")
        parent_N = max(1, node.N)

        for child in node.children.values():
            mean_value = child.Q / max(1, child.N)
            exploration = self.C * math.sqrt(math.log(parent_N) / (child.N + 1e-9))
            score = mean_value + exploration
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _rollout(self, board: Board, time_left: Callable[[], float]) -> float:
        """
        Simulate from 'board' up to max_rollout_depth using ε-greedy on our heuristic.
        Returns heuristic value at the rollout end state.
        """
        current = board
        depth = 0

        while depth < self.max_rollout_depth and time_left() > self.time_safety_margin:
            actions = self._legal_actions(current)
            if not actions:
                # No legal moves -> opponent gets 5 eggs; treat as strongly losing
                value = self._heuristic(current) - 5.0
                return value

            if random.random() < self.eps_rollout:
                action = random.choice(actions)
            else:
                # Greedy by heuristic: pick successor with best heuristic
                best_val = -float("inf")
                best_action = actions[0]
                for a in actions:
                    try:
                        next_board = current.forecast_move(*a)
                    except Exception:
                        continue
                    v = self._heuristic(next_board)
                    if v > best_val:
                        best_val = v
                        best_action = a
                action = best_action

            try:
                current = current.forecast_move(*action)
            except Exception:
                # If somehow invalid, just break and evaluate
                break

            depth += 1

        return self._heuristic(current)

    def _run_mcts_iteration(self, root: MCTSNode, time_left: Callable[[], float]):
        """
        One MCTS iteration: selection, expansion, rollout, backprop.
        """

        # --- Selection ---
        node = root
        while node.children and node.is_fully_expanded():
            if time_left() <= self.time_safety_margin:
                return
            node = self._select_child_ucb1(node)

        # --- Expansion ---
        if not node.untried_actions:
            node.untried_actions = self._legal_actions(node.board)

        if node.untried_actions:
            action = node.untried_actions.pop()
            try:
                next_board = node.board.forecast_move(*action)
            except Exception:
                # If expansion move somehow fails, mark as unusable
                next_board = None

            if next_board is not None:
                child = MCTSNode(next_board, parent=node, action=action)
                node.children[action] = child
                node = child  # leaf for rollout

        # --- Rollout ---
        if time_left() <= self.time_safety_margin:
            return
        rollout_value = self._rollout(node.board, time_left)

        # --- Backprop ---
        cur = node
        while cur is not None:
            cur.N += 1
            cur.Q += rollout_value
            cur = cur.parent

    # -------------- Internal move selection --------------

    def get_move(self, board: Board, time_left: Callable[[], float]) -> Tuple[Direction, MoveType]:
        """
        Internal helper for choosing a move with MCTS.
        """

        root = MCTSNode(board, parent=None, action=None)
        root.untried_actions = self._legal_actions(board)

        if not root.untried_actions:
            # No legal moves; return something (engine will handle the loss)
            return (Direction.UP, MoveType.PLAIN)

        # Basic move is heuristic-greedy; used as fallback
        greedy_action = self._best_greedy_action(board, root.untried_actions)

        # Time budget for this move
        my_time = time_left()
        allowed = min(self.max_move_time, max(0.01, my_time - self.time_safety_margin))

        start_time = my_time
        # Iterate while we still have at least the safety margin left
        while time_left() > start_time - allowed and time_left() > self.time_safety_margin:
            self._run_mcts_iteration(root, time_left)

        # Choose action with highest visit count
        if root.children:
            best_child = max(root.children.values(), key=lambda c: c.N)
            return best_child.action
        else:
            return greedy_action

    def _best_greedy_action(self, board: Board, actions: List[Tuple[Direction, MoveType]]) -> Tuple[Direction, MoveType]:
        best_val = -float("inf")
        best_action = actions[0]
        for a in actions:
            try:
                next_board = board.forecast_move(*a)
            except Exception:
                continue
            v = self._heuristic(next_board)
            if v > best_val:
                best_val = v
                best_action = a
        return best_action

    # -------------- Public API expected by the engine --------------

    def play(self, board: Board, sensor_data: List[Tuple[bool, bool]], time_left: Callable):
        """
        Entry point called by the game engine each turn.

        sensor_data:
            [(heard_white, felt_white), (heard_black, felt_black)]
        time_left:
            function returning remaining time (in seconds) for this player.
        """
        # Use the sensor readings to refine trapdoor beliefs before searching
        self._update_trap_beliefs(board, sensor_data)

        # Then pick a move via MCTS over those refined beliefs
        return self.get_move(board, time_left)
