"""
Bayesian trapdoor probability tracker for the Chicken Game.

Coordinate Convention:
- Game locations: (x, y) where x=column, y=row
- Internal arrays: [row][col] indexing
"""

from typing import List, Tuple, Set
from game.game_map import prob_hear, prob_feel


class TrapdoorTracker:
    """
    Tracks probability distributions for trapdoors using Bayesian inference.
    """
    
    def __init__(self, map_size: int = 8):
        """Initialize with prior distributions."""
        self.size = map_size
        
        # Probability distributions: [parity][row][col]
        # parity 0 = white (even), parity 1 = black (odd)
        self.priors = [
            self._init_prior_for_parity(parity=0),
            self._init_prior_for_parity(parity=1),
        ]
    
    def _init_prior_for_parity(self, parity: int) -> List[List[float]]:
        """
        Initialize prior probability distribution.
        
        Weight rules:
        - Edge (distance 0-1): weight 0
        - Inner ring (distance 2): weight 1
        - Center (distance 3+): weight 2
        """
        weights = [[0.0 for _ in range(self.size)] for _ in range(self.size)]
        total = 0.0
        
        for row in range(self.size):
            for col in range(self.size):
                if (row + col) % 2 != parity:
                    continue
                
                dist = min(row, col, self.size - 1 - row, self.size - 1 - col)
                
                if dist <= 1:
                    w = 0.0
                elif dist == 2:
                    w = 1.0
                else:
                    w = 2.0
                
                if w > 0:
                    weights[row][col] = w
                    total += w
        
        if total == 0:
            for row in range(self.size):
                for col in range(self.size):
                    if (row + col) % 2 == parity:
                        weights[row][col] = 1.0
                        total += 1.0
        
        for row in range(self.size):
            for col in range(self.size):
                weights[row][col] /= total
        
        return weights
    
    def update_with_sensors(self, player_loc: Tuple[int, int], 
                           sensor_data: List[Tuple[bool, bool]]):
        """
        Bayesian update using sensor readings.
        
        Args:
            player_loc: (x, y) = (column, row)
            sensor_data: [(heard_white, felt_white), (heard_black, felt_black)]
        """
        if not sensor_data or len(sensor_data) < 2:
            return
        
        my_x, my_y = player_loc
        
        for parity in (0, 1):
            heard, felt = sensor_data[parity]
            prior = self.priors[parity]
            posterior = [[0.0 for _ in range(self.size)] for _ in range(self.size)]
            total = 0.0
            
            for row in range(self.size):
                for col in range(self.size):
                    if (row + col) % 2 != parity:
                        continue
                    
                    p0 = prior[row][col]
                    if p0 == 0.0:
                        continue
                    
                    delta_x = abs(col - my_x)
                    delta_y = abs(row - my_y)
                    
                    p_h = prob_hear(delta_x, delta_y)
                    p_f = prob_feel(delta_x, delta_y)
                    
                    likelihood = (p_h if heard else (1.0 - p_h)) * \
                                (p_f if felt else (1.0 - p_f))
                    
                    val = p0 * likelihood
                    posterior[row][col] = val
                    total += val
            
            if total <= 0.0:
                continue
            
            for row in range(self.size):
                for col in range(self.size):
                    if (row + col) % 2 == parity:
                        posterior[row][col] /= total
            
            self.priors[parity] = posterior
    
    def update_with_found_trapdoors(self, found_trapdoors: Set[Tuple[int, int]]):
        """
        Set found trapdoors to 100% certainty.
        
        Args:
            found_trapdoors: Set of (x, y) locations
        """
        if not found_trapdoors:
            return
        
        for trap_loc in found_trapdoors:
            x, y = trap_loc
            parity = (x + y) % 2
            
            new_prior = [[0.0 for _ in range(self.size)] for _ in range(self.size)]
            new_prior[y][x] = 1.0
            self.priors[parity] = new_prior
    
    def get_probability(self, game_loc: Tuple[int, int]) -> float:
        """
        Get trapdoor probability at location.
        
        Args:
            game_loc: (x, y) = (column, row)
        
        Returns:
            Probability between 0.0 and 1.0
        """
        x, y = game_loc
        parity = (x + y) % 2
        return self.priors[parity][y][x]