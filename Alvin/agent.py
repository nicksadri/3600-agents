from collections.abc import Callable
from typing import List, Tuple
import numpy as np
import heapq
from game.board import Board
from game.enums import Direction, MoveType, loc_after_direction
from game.game_map import prob_hear, prob_feel

class PlayerAgent:
    def __init__(self, board: Board, time_left: Callable):
        self.map_size = board.game_map.MAP_SIZE
        self.trapdoor_probs = [self._init_prior(0), self._init_prior(1)]
        self.turn_count = 0

    def _init_prior(self, parity):
        dim = self.map_size
        probs = np.zeros((dim, dim))
        base_probs = np.zeros((dim, dim))
        base_probs[2 : dim - 2, 2 : dim - 2] = 1.0
        base_probs[3 : dim - 3, 3 : dim - 3] = 2.0
        
        for r in range(dim):
            for c in range(dim):
                if (r + c) % 2 == parity:
                    probs[r, c] = base_probs[r, c]
                else:
                    probs[r, c] = 0.0
                    
        total = np.sum(probs)
        if total > 0:
            return probs / total
        return probs

    def _update_belief(self, sensor_data, player_loc):
        for i in range(2):
            heard, felt = sensor_data[i]
            grid = self.trapdoor_probs[i]
            new_grid = np.zeros_like(grid)
            
            for r in range(self.map_size):
                for c in range(self.map_size):
                    if grid[r, c] == 0:
                        continue
                    
                    dx = abs(player_loc[0] - r)
                    dy = abs(player_loc[1] - c)
                    
                    p_hear = prob_hear(dx, dy)
                    p_feel = prob_feel(dx, dy)
                    
                    likelihood = 1.0
                    if heard:
                        likelihood *= p_hear
                    else:
                        likelihood *= (1 - p_hear)
                        
                    if felt:
                        likelihood *= p_feel
                    else:
                        likelihood *= (1 - p_feel)
                        
                    new_grid[r, c] = grid[r, c] * likelihood
            
            total = np.sum(new_grid)
            if total > 0:
                self.trapdoor_probs[i] = new_grid / total

    def play(self, board: Board, sensor_data: List[Tuple[bool, bool]], time_left: Callable):
        self.turn_count += 1
        my_loc = board.chicken_player.get_location()
        self._update_belief(sensor_data, my_loc)
        
        # 1. Identify targets
        targets = []
        parity = board.chicken_player.even_chicken
        
        for r in range(self.map_size):
            for c in range(self.map_size):
                if (r + c) % 2 != parity:
                    continue
                if (r, c) in board.eggs_player:
                    continue
                if (r, c) in board.turds_player or (r, c) in board.turds_enemy:
                    continue
                if (r, c) in board.eggs_enemy:
                    continue
                
                value = 10
                if (r == 0 or r == self.map_size - 1) and (c == 0 or c == self.map_size - 1):
                    value = 50
                
                targets.append(((r, c), value))
        
        # Check if current loc is a target
        is_current_target = False
        for t, v in targets:
            if t == my_loc:
                is_current_target = True
                break
        
        if is_current_target:
            valid_moves = board.get_valid_moves()
            egg_moves = [m for m in valid_moves if m[1] == MoveType.EGG]
            if egg_moves:
                return self._pick_best_move_from_list(board, egg_moves)

        if not targets:
            return self._greedy_move(board)

        # 2. Dijkstra to find best target
        costs = {my_loc: 0}
        first_moves = {my_loc: None}
        queue = [(0, my_loc)] # cost, loc
        
        risks = np.zeros((self.map_size, self.map_size))
        for r in range(self.map_size):
            for c in range(self.map_size):
                p = (r + c) % 2
                risks[r, c] = self.trapdoor_probs[p][r, c]
                if (r, c) in board.found_trapdoors:
                    risks[r, c] = 1.0

        target_map = {t: v for t, v in targets}
        
        best_target_val = -float('inf')
        best_first_move = None
        
        while queue:
            cost, u = heapq.heappop(queue)
            
            if cost > costs.get(u, float('inf')):
                continue
            
            if u in target_map:
                score = target_map[u] - cost * 2
                if score > best_target_val:
                    best_target_val = score
                    best_first_move = first_moves[u]
            
            if cost > 20: 
                continue

            for d in Direction:
                try:
                    v = loc_after_direction(u, d)
                except:
                    continue
                
                if not board.is_valid_cell(v):
                    continue
                
                if v in board.turds_player or v in board.turds_enemy:
                    continue
                if v in board.eggs_enemy:
                    continue
                if board.is_cell_in_enemy_turd_zone(v):
                    continue
                
                risk = risks[v[0], v[1]]
                weight = 1 + risk * 100
                
                new_cost = cost + weight
                
                if new_cost < costs.get(v, float('inf')):
                    costs[v] = new_cost
                    if first_moves[u] is None:
                        first_moves[v] = (d, MoveType.PLAIN)
                    else:
                        first_moves[v] = first_moves[u]
                    heapq.heappush(queue, (new_cost, v))
                    
        if best_first_move:
            d, _ = best_first_move
            valid_moves = board.get_valid_moves()
            for vm in valid_moves:
                if vm[0] == d and vm[1] == MoveType.PLAIN:
                    return vm
            
        return self._greedy_move(board)

    def _pick_best_move_from_list(self, board, moves):
        best_m = None
        min_risk = float('inf')
        for m in moves:
            d, t = m
            new_loc = loc_after_direction(board.chicken_player.get_location(), d)
            p = (new_loc[0] + new_loc[1]) % 2
            risk = self.trapdoor_probs[p][new_loc]
            if risk < min_risk:
                min_risk = risk
                best_m = m
        return best_m if best_m else moves[0]

    def _greedy_move(self, board):
        valid_moves = board.get_valid_moves()
        if not valid_moves:
             return (Direction.UP, MoveType.PLAIN)
        return self._pick_best_move_from_list(board, valid_moves)
