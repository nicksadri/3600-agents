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
    Tracks probability distributions for trapdoors using:
    - Prior from trapdoor placement rules (center-weighted, parity-specific)
    - All sensor readings over time (Bayesian updates)
    - Movement history of BOTH players (safe squares)
    - Hard logical deduction when only one candidate remains for a parity
    """

    def __init__(self, map_size: int = 8):
        self.size = map_size

        # priors[parity][row][col]
        # parity 0 = even squares, parity 1 = odd squares
        self.priors: List[List[List[float]]] = [
            self._init_prior_for_parity(parity=0),
            self._init_prior_for_parity(parity=1),
        ]

        # Movement / visit history
        self.visited_us: Set[Tuple[int, int]] = set()
        self.visited_opp: Set[Tuple[int, int]] = set()

        # For debugging / sanity: store sensor observations (optional)
        # Each element: (loc=(x,y), sensor_data=[(heard_even,felt_even),(heard_odd,felt_odd)])
        self.observations: List[Tuple[Tuple[int, int], List[Tuple[bool, bool]]]] = []

    # -------------------------- Priors -------------------------- #

    def _init_prior_for_parity(self, parity: int) -> List[List[float]]:
        """
        Initialize prior probability distribution for one parity.

        Weight rules (mirror engine's trapdoor_manager logic):
        - Outer ring (dist_edge <= 1): weight 0 (no trapdoors there)
        - Next ring (dist_edge == 2): weight 1
        - Inner region (dist_edge >= 3): weight 2
        """
        weights = [[0.0 for _ in range(self.size)] for _ in range(self.size)]
        total = 0.0

        for row in range(self.size):
            for col in range(self.size):
                if (row + col) % 2 != parity:
                    continue

                dist_edge = min(row, col, self.size - 1 - row, self.size - 1 - col)

                if dist_edge <= 1:
                    w = 0.0
                elif dist_edge == 2:
                    w = 1.0
                else:
                    w = 2.0

                if w > 0:
                    weights[row][col] = w
                    total += w

        # Fallback in case everything was zero
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

    # -------------------------- Safe squares logic -------------------------- #

    def _apply_safe_squares(self, safe_squares: Set[Tuple[int, int]]):
        """
        Given a set of squares that CANNOT be trapdoors, zero out their probability
        and renormalize for each parity separately.
        """
        # Zero out safe squares
        for x, y in safe_squares:
            if not (0 <= x < self.size and 0 <= y < self.size):
                continue
            parity = (x + y) % 2
            self.priors[parity][y][x] = 0.0

        # Renormalize each parity
        for parity in (0, 1):
            grid = self.priors[parity]
            total = 0.0
            for r in range(self.size):
                for c in range(self.size):
                    total += grid[r][c]

            if total > 0.0:
                for r in range(self.size):
                    for c in range(self.size):
                        grid[r][c] /= total
            else:
                # If everything went to 0 (shouldn't happen), fall back to uniform over parity
                count = 0
                for r in range(self.size):
                    for c in range(self.size):
                        if (r + c) % 2 == parity:
                            grid[r][c] = 1.0
                            count += 1
                        else:
                            grid[r][c] = 0.0
                for r in range(self.size):
                    for c in range(self.size):
                        grid[r][c] /= count

    # -------------------------- Sensor update -------------------------- #

    def update_with_sensors(
        self,
        player_loc: Tuple[int, int],
        sensor_data: List[Tuple[bool, bool]],
    ):
        """
        Bayesian update using sensor readings at the given player location.

        Args:
            player_loc: (x, y) = (column, row)
            sensor_data: [(heard_even, felt_even), (heard_odd, felt_odd)]
        """
        if not sensor_data or len(sensor_data) < 2:
            return

        self.observations.append((player_loc, sensor_data))

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
                # If this happens, the reading was contradictory to all priors;
                # we simply skip updating this parity (keep the prior).
                continue

            for row in range(self.size):
                for col in range(self.size):
                    if (row + col) % 2 == parity:
                        posterior[row][col] /= total

            self.priors[parity] = posterior

    # -------------------------- Found trapdoors -------------------------- #

    def update_with_found_trapdoors(self, found_trapdoors: Set[Tuple[int, int]]):
        """
        If we *know* a trapdoor has been found at a location, set that cell to
        probability 1.0 for its parity and everything else to 0.

        This already does your "if we find a trapdoor, set that cell to 100%" logic.
        """
        if not found_trapdoors:
            return

        for trap_loc in found_trapdoors:
            x, y = trap_loc
            if not (0 <= x < self.size and 0 <= y < self.size):
                continue
            parity = (x + y) % 2

            new_prior = [[0.0 for _ in range(self.size)] for _ in range(self.size)]
            new_prior[y][x] = 1.0
            self.priors[parity] = new_prior

    # -------------------------- Logical snapping -------------------------- #

    def _snap_if_single_candidate(self):
        """
        If for a given parity there is exactly ONE cell with non-zero probability,
        snap that cell to probability 1.0 (and the rest to 0.0).

        This matches your intuition:
          If hearing/feeling + safe squares elimination leave only one possible
          cell for the trapdoor, then that cell *must* be it.
        """
        for parity in (0, 1):
            grid = self.priors[parity]
            candidates = [
                (r, c)
                for r in range(self.size)
                for c in range(self.size)
                if grid[r][c] > 0.0
            ]
            if len(candidates) == 1:
                r0, c0 = candidates[0]
                for r in range(self.size):
                    for c in range(self.size):
                        grid[r][c] = 0.0
                grid[r0][c0] = 1.0

    # -------------------------- High-level step API -------------------------- #

    def step(
        self,
        my_pos: Tuple[int, int],
        opp_pos: Tuple[int, int],
        sensor_data: List[Tuple[bool, bool]],
        eggs_player: Set[Tuple[int, int]],
        eggs_enemy: Set[Tuple[int, int]],
        turds_player: Set[Tuple[int, int]],
        turds_enemy: Set[Tuple[int, int]],
        found_trapdoors: Set[Tuple[int, int]],
    ):
        """
        One "turn" of belief update.

        Inputs:
          - my_pos:       where *we* are now
          - opp_pos:      where the opponent is now
          - sensor_data:  [(heard_even,felt_even), (heard_odd,felt_odd)]
          - eggs_player, eggs_enemy, turds_player, turds_enemy: board marks
          - found_trapdoors: set of locs confirmed as trapdoors

        Logic:
          1. Record that both players have visited their current cells.
          2. Mark all eggs / turds as safe squares.
          3. Apply "safe squares" to zeros out those probabilities.
          4. Apply Bayesian update for the new sensor reading.
          5. Re-apply safe squares (in case Bayes revived them numerically).
          6. If only one candidate remains for a parity, snap it to 100%.
        """
        # 1. Record movement history
        self.visited_us.add(my_pos)
        self.visited_opp.add(opp_pos)

        # 2. Found trapdoors -> collapse to certainty where known
        self.update_with_found_trapdoors(found_trapdoors)

        # 3. Build safe set:
        #    - any egg or turd square
        #    - any square we've stood on and not teleported from
        #    - any square opponent currently stands on or has left marks on
        safe_squares: Set[Tuple[int, int]] = set()
        safe_squares |= eggs_player
        safe_squares |= eggs_enemy
        safe_squares |= turds_player
        safe_squares |= turds_enemy
        safe_squares |= self.visited_us
        safe_squares |= self.visited_opp

        # 4. Apply safe squares (zero out impossible trap cells)
        self._apply_safe_squares(safe_squares)

        # 5. Bayesian sensor update for this turn
        self.update_with_sensors(my_pos, sensor_data)

        # 6. Re-apply safe squares (in case tiny probabilities reappeared numerically)
        self._apply_safe_squares(safe_squares)

        # 7. Hard logical snap if only one candidate left per parity
        self._snap_if_single_candidate()

    # -------------------------- Probability query -------------------------- #

    def get_probability(self, game_loc: Tuple[int, int]) -> float:
        """
        Get trapdoor probability at location.

        Returns:
            Probability between 0.0 and 1.0 for whichever parity this square has.
        """
        x, y = game_loc
        parity = (x + y) % 2
        if not (0 <= x < self.size and 0 <= y < self.size):
            return 0.0
        return self.priors[parity][y][x]

    def debug_print(self, precision: int = 2):
        """
        Pretty-print a single combined 8x8 probability grid.

        Each cell shows the probability of the trapdoor for its parity:
          - even (i+j even): from priors[0]
          - odd  (i+j odd):  from priors[1]
        """
        fmt = f"{{:.{precision}f}}"
        print("\n=== Combined Trapdoor Probability Grid ===")

        # column header
        header = "    " + " ".join(f"{c:>6}" for c in range(self.size))
        print(header)

        for r in range(self.size):
            row_vals = []
            for c in range(self.size):
                parity = (r + c) % 2
                p = self.priors[parity][r][c]
                row_vals.append(fmt.format(p))
            print(f"r={r:>2} " + " ".join(f"{v:>6}" for v in row_vals))

        total_even = sum(sum(row) for row in self.priors[0])
        total_odd  = sum(sum(row) for row in self.priors[1])
        print(f"\nTotal sum parity 0 (even): {total_even:.6f}")
        print(f"Total sum parity 1 (odd):  {total_odd:.6f}")
