# agent.py — Carlos, a Monte Carlo Tree Search agent

import math
import random
from collections.abc import Callable
from typing import List, Tuple, Optional

from game.board import Board
from game.enums import Direction, MoveType

from .trapdoor_tracker import TrapdoorTracker


# ------------------------------
# MCTS Node
# ------------------------------

class MCTSNode:
    def __init__(self, board: Board, parent: Optional["MCTSNode"], 
                 action: Optional[Tuple[Direction, MoveType]]):
        self.board = board
        self.parent = parent
        self.action = action
        self.children: dict[Tuple[Direction, MoveType], "MCTSNode"] = {}
        self.N = 0
        self.Q = 0.0
        self.untried_actions: List[Tuple[Direction, MoveType]] = []

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0


# ------------------------------
# Carlos PlayerAgent
# ------------------------------

class PlayerAgent:
    """Monte Carlo Tree Search agent with Bayesian trapdoor tracking."""

    def __init__(self, board: Board, time_left: Callable[[], float]):
        self.map_size = board.game_map.MAP_SIZE
        
        # MCTS parameters
        self.C = 1.4
        self.eps_rollout = 0.2
        self.max_rollout_depth = 12
        self.time_safety_margin = 0.03
        self.max_move_time = 0.20

        # Heuristic weights
        self.w_eggs = 1.0
        self.w_trap = 0.7

        # Trapdoor tracker
        self.tracker = TrapdoorTracker(self.map_size)

    # -------------- Board helpers --------------

    def _my_pos(self, board: Board) -> Tuple[int, int]:
        return board.chicken_player.get_location()

    def _opp_pos(self, board: Board) -> Tuple[int, int]:
        return board.chicken_enemy.get_location()

    def _my_eggs(self, board: Board) -> int:
        return board.chicken_player.get_eggs_laid()

    def _opp_eggs(self, board: Board) -> int:
        return board.chicken_enemy.get_eggs_laid()

    def _legal_actions(self, board: Board) -> List[Tuple[Direction, MoveType]]:
        return board.get_valid_moves()

    # -------------- Heuristic --------------

    def _heuristic(self, board: Board) -> float:
        """Linear combination of eggs and trapdoor risk."""
        E_me = self._my_eggs(board)
        E_opp = self._opp_eggs(board)
        egg_feature = E_me - E_opp

        my_pos = self._my_pos(board)
        opp_pos = self._opp_pos(board)

        p_me = self.tracker.get_probability(my_pos)
        p_opp = self.tracker.get_probability(opp_pos)

        trap_feature = 4.0 * (p_opp - p_me)

        value = self.w_eggs * egg_feature + self.w_trap * trap_feature
        return value

    # -------------- MCTS core --------------

    def _select_child_ucb1(self, node: MCTSNode) -> MCTSNode:
        """Select child using UCB1 formula."""
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
        """Simulate using ε-greedy on heuristic."""
        current = board
        depth = 0

        while depth < self.max_rollout_depth and time_left() > self.time_safety_margin:
            actions = self._legal_actions(current)
            if not actions:
                value = self._heuristic(current) - 5.0
                return value

            if random.random() < self.eps_rollout:
                action = random.choice(actions)
            else:
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
                break

            depth += 1

        return self._heuristic(current)

    def _run_mcts_iteration(self, root: MCTSNode, time_left: Callable[[], float]):
        """One MCTS iteration: selection, expansion, rollout, backprop."""
        
        # Selection
        node = root
        while node.children and node.is_fully_expanded():
            if time_left() <= self.time_safety_margin:
                return
            node = self._select_child_ucb1(node)

        # Expansion
        if not node.untried_actions:
            node.untried_actions = self._legal_actions(node.board)

        if node.untried_actions:
            action = node.untried_actions.pop()
            try:
                next_board = node.board.forecast_move(*action)
            except Exception:
                next_board = None

            if next_board is not None:
                child = MCTSNode(next_board, parent=node, action=action)
                node.children[action] = child
                node = child

        # Rollout
        if time_left() <= self.time_safety_margin:
            return
        rollout_value = self._rollout(node.board, time_left)

        # Backprop
        cur = node
        while cur is not None:
            cur.N += 1
            cur.Q += rollout_value
            cur = cur.parent

    # -------------- Move selection --------------

    def get_move(self, board: Board, time_left: Callable[[], float]) -> Tuple[Direction, MoveType]:
        """Choose move using MCTS."""
        root = MCTSNode(board, parent=None, action=None)
        root.untried_actions = self._legal_actions(board)

        if not root.untried_actions:
            return (Direction.UP, MoveType.PLAIN)

        greedy_action = self._best_greedy_action(board, root.untried_actions)

        my_time = time_left()
        allowed = min(self.max_move_time, max(0.01, my_time - self.time_safety_margin))

        start_time = my_time
        while time_left() > start_time - allowed and time_left() > self.time_safety_margin:
            self._run_mcts_iteration(root, time_left)

        if root.children:
            best_child = max(root.children.values(), key=lambda c: c.N)
            return best_child.action
        else:
            return greedy_action

    def _best_greedy_action(self, board: Board, actions: List[Tuple[Direction, MoveType]]) -> Tuple[Direction, MoveType]:
        """Select best action by immediate heuristic."""
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

    # -------------- Public API --------------

    def play(self, board: Board, sensor_data: List[Tuple[bool, bool]], time_left: Callable):
        """Entry point called by game engine each turn."""
        # Update trapdoor beliefs
        self.tracker.update_with_found_trapdoors(board.found_trapdoors)
        self.tracker.update_with_sensors(self._my_pos(board), sensor_data)

        # Choose move via MCTS
        return self.get_move(board, time_left)