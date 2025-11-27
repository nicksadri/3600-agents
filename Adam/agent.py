from collections.abc import Callable
from typing import List, Tuple
import numpy as np
import heapq
from game.board import Board
from game.enums import Direction, MoveType, loc_after_direction
from game.game_map import prob_hear, prob_feel

class PlayerAgent:
    """
    A chicken agent that uses Bayesian trapdoor tracking and Dijkstra pathfinding
    to navigate safely to high-value egg-laying locations.
    
    Coordinate Convention:
    - Game locations: (x, y) where x=column, y=row
    - NumPy arrays: [row, col] indexing
    - Grid cell [r, c] corresponds to game location (c, r)
    """
    
    def __init__(self, board: Board, time_left: Callable):
        self.map_size = board.game_map.MAP_SIZE
        # Initialize probability grids for white (even) and black (odd) trapdoors
        self.trapdoor_probs = [self._init_prior(0), self._init_prior(1)]
        self.turn_count = 0

    def _init_prior(self, parity: int) -> np.ndarray:
        """
        Initialize prior probability distribution for trapdoors based on game rules.
        Trapdoors are weighted by distance from edge:
        - Edge squares (distance 0-1): weight 0
        - Inner ring (distance 2): weight 1
        - Center (distance 3+): weight 2
        
        Args:
            parity: 0 for even (white) squares, 1 for odd (black) squares
        
        Returns:
            Normalized probability array [row, col]
        """
        dim = self.map_size
        probs = np.zeros((dim, dim))
        base_probs = np.zeros((dim, dim))
        
        # Set weights based on distance from edge
        base_probs[2 : dim - 2, 2 : dim - 2] = 1.0
        base_probs[3 : dim - 3, 3 : dim - 3] = 2.0
        
        # Only assign probabilities to squares of correct parity
        for r in range(dim):
            for c in range(dim):
                if (r + c) % 2 == parity:
                    probs[r, c] = base_probs[r, c]
                else:
                    probs[r, c] = 0.0
        
        # Normalize to create valid probability distribution
        total = np.sum(probs)
        if total > 0:
            return probs / total
        return probs

    def _update_belief(self, sensor_data: List[Tuple[bool, bool]], player_loc: Tuple[int, int]):
        """
        Update trapdoor probability distributions using Bayesian inference.
        
        Args:
            sensor_data: [(heard_white, felt_white), (heard_black, felt_black)]
            player_loc: Current position as (x, y) = (column, row)
        """
        for i in range(2):
            heard, felt = sensor_data[i]
            grid = self.trapdoor_probs[i]
            new_grid = np.zeros_like(grid)
            
            for r in range(self.map_size):      # r = row index in array
                for c in range(self.map_size):  # c = col index in array
                    if grid[r, c] == 0:
                        continue
                    
                    # Grid cell [r, c] represents game location (x=c, y=r)
                    # Calculate distance from player to potential trapdoor
                    dx = abs(player_loc[0] - c)  # |player_x - trap_x|
                    dy = abs(player_loc[1] - r)  # |player_y - trap_y|
                    
                    # Get sensor probabilities based on distance
                    p_hear = prob_hear(dx, dy)
                    p_feel = prob_feel(dx, dy)
                    
                    # Calculate likelihood: P(sensor_data | trapdoor at [r,c])
                    likelihood = 1.0
                    likelihood *= p_hear if heard else (1 - p_hear)
                    likelihood *= p_feel if felt else (1 - p_feel)
                    
                    # Bayesian update: posterior = prior * likelihood
                    new_grid[r, c] = grid[r, c] * likelihood
            
            # Normalize to maintain valid probability distribution
            total = np.sum(new_grid)
            if total > 0:
                self.trapdoor_probs[i] = new_grid / total

    def _update_found_trapdoors(self, board: Board):
        """
        Update probability distributions when trapdoors are discovered.
        Set found trapdoor locations to 100% certainty.
        
        Args:
            board: Current game board
        """
        for trap_loc in board.found_trapdoors:
            x, y = trap_loc  # Game location (column, row)
            parity = (x + y) % 2
            
            # Create new distribution with 100% certainty at found location
            new_probs = np.zeros_like(self.trapdoor_probs[parity])
            new_probs[y, x] = 1.0  # NumPy indexing: [row, col] = [y, x]
            self.trapdoor_probs[parity] = new_probs

    def _get_risk(self, game_loc: Tuple[int, int]) -> float:
        """
        Get trapdoor risk for a game location.
        
        Args:
            game_loc: Location as (x, y) = (column, row)
        
        Returns:
            Probability that this location is a trapdoor
        """
        x, y = game_loc
        parity = (x + y) % 2
        return self.trapdoor_probs[parity][y, x]  # grid[row, col]

    def _identify_targets(self, board: Board) -> List[Tuple[Tuple[int, int], float]]:
        """
        Identify all valid egg-laying targets and assign values to them.
        
        Args:
            board: Current game board
        
        Returns:
            List of (location, value) tuples where location is (x, y)
        """
        targets = []
        parity = board.chicken_player.even_chicken
        opp_loc = board.chicken_enemy.get_location()
        
        for r in range(self.map_size):
            for c in range(self.map_size):
                # Convert array indices to game location
                game_loc = (c, r)  # (x, y) = (column, row)
                
                # Check parity - can we lay an egg here?
                if (game_loc[0] + game_loc[1]) % 2 != parity:
                    continue
                
                # Skip occupied squares
                if game_loc in board.eggs_player:
                    continue
                if game_loc in board.turds_player or game_loc in board.turds_enemy:
                    continue
                if game_loc in board.eggs_enemy:
                    continue
                
                # Calculate value for this target
                value = 10  # Base value
                risk = self.trapdoor_probs[parity][r, c]
                
                # Bonus for safe squares
                if risk < 0.10:
                    value += 10
                
                # High value for corners (3 eggs instead of 1)
                if (game_loc[0] == 0 or game_loc[0] == self.map_size - 1) and \
                   (game_loc[1] == 0 or game_loc[1] == self.map_size - 1):
                    value += 50
                
                # Medium value for edges
                elif game_loc[0] == 0 or game_loc[0] == self.map_size - 1 or \
                     game_loc[1] == 0 or game_loc[1] == self.map_size - 1:
                    value += 20
                
                # Avoid opponent proximity (risk of being blocked or turded)
                dist_to_opp = abs(game_loc[0] - opp_loc[0]) + abs(game_loc[1] - opp_loc[1])
                if dist_to_opp < 3:
                    value -= 10
                
                # Bonus for clustering near our own eggs
                for egg_loc in board.eggs_player:
                    dist = abs(game_loc[0] - egg_loc[0]) + abs(game_loc[1] - egg_loc[1])
                    if dist == 2:  # Same parity squares are 2 steps apart
                        value += 5
                
                # Penalize risky squares heavily
                value -= risk * 100
                
                targets.append((game_loc, value))
        
        return targets

    def _should_lay_egg_now(self, board: Board) -> bool:
        """
        Determine if we should lay an egg at our current location.
        
        Args:
            board: Current game board
        
        Returns:
            True if we should lay an egg now, False otherwise
        """
        my_loc = board.chicken_player.get_location()
        
        # Check if current location is valid for egg-laying
        if my_loc in board.eggs_player or my_loc in board.turds_player or \
           my_loc in board.turds_enemy or my_loc in board.eggs_enemy:
            return False
        
        # Get risk at current location
        current_risk = self._get_risk(my_loc)
        
        # Check if we're on a high-value square
        is_corner = (my_loc[0] == 0 or my_loc[0] == self.map_size - 1) and \
                    (my_loc[1] == 0 or my_loc[1] == self.map_size - 1)
        is_edge = (my_loc[0] == 0 or my_loc[0] == self.map_size - 1 or \
                   my_loc[1] == 0 or my_loc[1] == self.map_size - 1)
        
        # Lay egg if risk is low, or if we're on corner/edge with acceptable risk
        if current_risk < 0.05:
            return True
        if (is_corner or is_edge) and current_risk < 0.20:
            return True
        
        return False

    def _find_best_path(self, board: Board, targets: List[Tuple[Tuple[int, int], float]]) -> Tuple[Direction, MoveType]:
        """
        Use Dijkstra's algorithm to find the best first move toward valuable targets.
        
        Args:
            board: Current game board
            targets: List of (location, value) tuples
        
        Returns:
            Best (direction, move_type) tuple
        """
        my_loc = board.chicken_player.get_location()
        
        # Initialize Dijkstra structures
        costs = {my_loc: 0}
        first_moves = {my_loc: None}
        queue = [(0, my_loc)]  # (cost, location)
        
        # Pre-compute risk grid for fast lookup
        risks = np.zeros((self.map_size, self.map_size))
        for r in range(self.map_size):
            for c in range(self.map_size):
                game_loc = (c, r)
                parity = (c + r) % 2
                risks[r, c] = self.trapdoor_probs[parity][r, c]
                
                # Found trapdoors have maximum risk
                if game_loc in board.found_trapdoors:
                    risks[r, c] = 1.0
        
        # Convert targets to dictionary for fast lookup
        target_map = {loc: val for loc, val in targets}
        
        best_target_score = -float('inf')
        best_first_move = None
        
        # Dijkstra's algorithm
        while queue:
            cost, u = heapq.heappop(queue)
            
            # Skip if we've found a better path to u
            if cost > costs.get(u, float('inf')):
                continue
            
            # Check if u is a target
            if u in target_map:
                # Score = target value - path cost
                score = target_map[u] - cost
                if score > best_target_score:
                    best_target_score = score
                    best_first_move = first_moves[u]
            
            # Limit search depth for performance
            if cost > 30:
                continue
            
            # Explore neighbors
            for d in Direction:
                try:
                    v = loc_after_direction(u, d)
                except:
                    continue
                
                # Check if neighbor is valid
                if not board.is_valid_cell(v):
                    continue
                if v in board.turds_player or v in board.turds_enemy:
                    continue
                if v in board.eggs_enemy:
                    continue
                if board.is_cell_in_enemy_turd_zone(v):
                    continue
                
                # Calculate edge weight (movement cost)
                x, y = v
                risk = risks[y, x]  # risks[row, col]
                weight = 1 + risk * 50  # Base cost + risk penalty
                
                new_cost = cost + weight
                
                # Update if we found a better path
                if new_cost < costs.get(v, float('inf')):
                    costs[v] = new_cost
                    
                    # Record first move from start location
                    if first_moves[u] is None:
                        first_moves[v] = (d, MoveType.PLAIN)
                    else:
                        first_moves[v] = first_moves[u]
                    
                    heapq.heappush(queue, (new_cost, v))
        
        return best_first_move

    def _pick_best_move_from_list(self, board: Board, moves: List[Tuple[Direction, MoveType]]) -> Tuple[Direction, MoveType]:
        """
        Select the safest move from a list of valid moves.
        
        Args:
            board: Current game board
            moves: List of valid (direction, move_type) tuples
        
        Returns:
            The move with lowest trapdoor risk
        """
        if not moves:
            return (Direction.UP, MoveType.PLAIN)
        
        best_move = None
        min_risk = float('inf')
        
        for direction, move_type in moves:
            my_loc = board.chicken_player.get_location()
            new_loc = loc_after_direction(my_loc, direction)
            risk = self._get_risk(new_loc)
            
            if risk < min_risk:
                min_risk = risk
                best_move = (direction, move_type)
        
        return best_move if best_move else moves[0]

    def _greedy_move(self, board: Board) -> Tuple[Direction, MoveType]:
        """
        Fallback strategy: pick the safest valid move.
        
        Args:
            board: Current game board
        
        Returns:
            A safe (direction, move_type) tuple
        """
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return (Direction.UP, MoveType.PLAIN)
        return self._pick_best_move_from_list(board, valid_moves)

    def play(self, board: Board, sensor_data: List[Tuple[bool, bool]], time_left: Callable) -> Tuple[Direction, MoveType]:
        """
        Main decision function called each turn.
        
        Args:
            board: Current game state
            sensor_data: [(heard_white, felt_white), (heard_black, felt_black)]
            time_left: Function returning remaining time in seconds
        
        Returns:
            (direction, move_type) tuple representing the chosen action
        """
        self.turn_count += 1
        my_loc = board.chicken_player.get_location()
        
        # Update trapdoor beliefs
        self._update_belief(sensor_data, my_loc)
        self._update_found_trapdoors(board)
        
        # Check if we should lay an egg at current location
        if self._should_lay_egg_now(board):
            valid_moves = board.get_valid_moves()
            egg_moves = [m for m in valid_moves if m[1] == MoveType.EGG]
            if egg_moves:
                return self._pick_best_move_from_list(board, egg_moves)
        
        # Identify potential egg-laying targets
        targets = self._identify_targets(board)
        
        if not targets:
            return self._greedy_move(board)
        
        # Find best path to valuable target
        best_move = self._find_best_path(board, targets)
        
        if best_move:
            direction, _ = best_move
            
            # Verify the move is valid and execute it
            valid_moves = board.get_valid_moves()
            for vm in valid_moves:
                if vm[0] == direction and vm[1] == MoveType.PLAIN:
                    return vm
        
        # Fallback to greedy strategy
        return self._greedy_move(board)