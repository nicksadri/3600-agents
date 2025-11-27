#!/usr/bin/env python3
"""
Tournament simulator for CS3600 Chicken Game agents.
Runs multiple games between two agents and reports win statistics.
"""

import os
import pathlib
import sys
import time
import json
from collections import defaultdict
from typing import Dict, Tuple

# Add engine to path
engine_path = pathlib.Path(__file__).parent / "engine"
sys.path.insert(0, str(engine_path))

from board_utils import get_history_json
from gameplay import play_game
from game.enums import ResultArbiter, WinReason


class TournamentStats:
    """Track and display tournament statistics."""
    
    def __init__(self, player_a_name: str, player_b_name: str):
        self.player_a_name = player_a_name
        self.player_b_name = player_b_name
        self.games_played = 0
        
        # Win counters
        self.wins = {
            ResultArbiter.PLAYER_A: 0,
            ResultArbiter.PLAYER_B: 0,
            ResultArbiter.TIE: 0
        }
        
        # Win reasons
        self.win_reasons = defaultdict(int)
        
        # Performance metrics
        self.total_turns = 0
        self.total_time = 0
        self.a_total_eggs = 0
        self.b_total_eggs = 0
        self.a_timeouts = 0
        self.b_timeouts = 0
        self.a_crashes = 0
        self.b_crashes = 0
        self.trapdoor_triggers = 0
    
    def record_game(self, final_board, game_time: float, trapdoors, spawns, err_a: str, err_b: str):
        """Record results from a single game."""
        self.games_played += 1
        self.total_turns += final_board.turn_count
        self.total_time += game_time
        
        # Record winner
        winner = final_board.winner
        self.wins[winner] += 1
        
        # Record win reason
        reason = WinReason(final_board.win_reason).name
        self.win_reasons[reason] += 1
        
        # Get final egg counts
        # Note: board perspective might be flipped, so we need to check is_as_turn
        if final_board.is_as_turn:
            a_eggs = final_board.chicken_player.get_eggs_laid()
            b_eggs = final_board.chicken_enemy.get_eggs_laid()
        else:
            a_eggs = final_board.chicken_enemy.get_eggs_laid()
            b_eggs = final_board.chicken_player.get_eggs_laid()
        
        self.a_total_eggs += a_eggs
        self.b_total_eggs += b_eggs
        
        # Track errors
        if err_a:
            if "Timeout" in err_a or "timeout" in err_a.lower():
                self.a_timeouts += 1
            else:
                self.a_crashes += 1
        
        if err_b:
            if "Timeout" in err_b or "timeout" in err_b.lower():
                self.b_timeouts += 1
            else:
                self.b_crashes += 1
        
        # Count trapdoor triggers
        if hasattr(final_board, 'found_trapdoors'):
            self.trapdoor_triggers += len(final_board.found_trapdoors)
    
    def print_progress(self):
        """Print progress update."""
        if self.games_played % 10 == 0:
            a_pct = (self.wins[ResultArbiter.PLAYER_A] / self.games_played * 100) if self.games_played > 0 else 0
            b_pct = (self.wins[ResultArbiter.PLAYER_B] / self.games_played * 100) if self.games_played > 0 else 0
            print(f"[{self.games_played} games] {self.player_a_name}: {a_pct:.1f}% | {self.player_b_name}: {b_pct:.1f}%")
    
    def print_summary(self):
        """Print comprehensive tournament summary."""
        print("\n" + "="*70)
        print(f"TOURNAMENT RESULTS: {self.player_a_name} vs {self.player_b_name}")
        print("="*70)
        print(f"\nTotal Games Played: {self.games_played}")
        print(f"Total Time: {self.total_time:.2f} seconds ({self.total_time/60:.2f} minutes)")
        print(f"Average Game Time: {self.total_time/self.games_played:.2f} seconds")
        print(f"Average Turns per Game: {self.total_turns/self.games_played:.1f}")
        
        print("\n" + "-"*70)
        print("WIN STATISTICS")
        print("-"*70)
        
        a_wins = self.wins[ResultArbiter.PLAYER_A]
        b_wins = self.wins[ResultArbiter.PLAYER_B]
        ties = self.wins[ResultArbiter.TIE]
        
        a_pct = (a_wins / self.games_played * 100) if self.games_played > 0 else 0
        b_pct = (b_wins / self.games_played * 100) if self.games_played > 0 else 0
        tie_pct = (ties / self.games_played * 100) if self.games_played > 0 else 0
        
        print(f"{self.player_a_name:20s}: {a_wins:4d} wins ({a_pct:5.1f}%)")
        print(f"{self.player_b_name:20s}: {b_wins:4d} wins ({b_pct:5.1f}%)")
        print(f"{'Ties':20s}: {ties:4d}      ({tie_pct:5.1f}%)")
        
        print("\n" + "-"*70)
        print("WIN REASONS")
        print("-"*70)
        for reason, count in sorted(self.win_reasons.items(), key=lambda x: x[1], reverse=True):
            pct = (count / self.games_played * 100) if self.games_played > 0 else 0
            print(f"{reason:20s}: {count:4d} ({pct:5.1f}%)")
        
        print("\n" + "-"*70)
        print("PERFORMANCE METRICS")
        print("-"*70)
        avg_a_eggs = self.a_total_eggs / self.games_played if self.games_played > 0 else 0
        avg_b_eggs = self.b_total_eggs / self.games_played if self.games_played > 0 else 0
        
        print(f"{self.player_a_name} Average Eggs: {avg_a_eggs:.2f}")
        print(f"{self.player_b_name} Average Eggs: {avg_b_eggs:.2f}")
        print(f"{self.player_a_name} Timeouts: {self.a_timeouts}")
        print(f"{self.player_b_name} Timeouts: {self.b_timeouts}")
        print(f"{self.player_a_name} Crashes: {self.a_crashes}")
        print(f"{self.player_b_name} Crashes: {self.b_crashes}")
        print(f"Total Trapdoor Triggers: {self.trapdoor_triggers}")
        
        print("\n" + "="*70)
        
        # Determine overall winner
        if a_wins > b_wins:
            margin = a_wins - b_wins
            print(f"🏆 WINNER: {self.player_a_name} (by {margin} games)")
        elif b_wins > a_wins:
            margin = b_wins - a_wins
            print(f"🏆 WINNER: {self.player_b_name} (by {margin} games)")
        else:
            print(f"🤝 TOURNAMENT TIED")
        
        print("="*70 + "\n")
    
    def save_to_file(self, filename: str):
        """Save tournament results to JSON file."""
        data = {
            "player_a": self.player_a_name,
            "player_b": self.player_b_name,
            "games_played": self.games_played,
            "total_time": self.total_time,
            "total_turns": self.total_turns,
            "wins": {
                "player_a": self.wins[ResultArbiter.PLAYER_A],
                "player_b": self.wins[ResultArbiter.PLAYER_B],
                "ties": self.wins[ResultArbiter.TIE]
            },
            "win_reasons": dict(self.win_reasons),
            "metrics": {
                "a_total_eggs": self.a_total_eggs,
                "b_total_eggs": self.b_total_eggs,
                "a_avg_eggs": self.a_total_eggs / self.games_played if self.games_played > 0 else 0,
                "b_avg_eggs": self.b_total_eggs / self.games_played if self.games_played > 0 else 0,
                "a_timeouts": self.a_timeouts,
                "b_timeouts": self.b_timeouts,
                "a_crashes": self.a_crashes,
                "b_crashes": self.b_crashes,
                "trapdoor_triggers": self.trapdoor_triggers
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filename}")


def run_tournament(player_a_name: str, player_b_name: str, num_games: int = 200, 
                   save_games: bool = False, display_games: bool = False):
    """
    Run a tournament between two agents.
    
    Args:
        player_a_name: Name of first agent directory
        player_b_name: Name of second agent directory
        num_games: Number of games to play
        save_games: Whether to save individual game histories
        display_games: Whether to display games in terminal
    """
    top_level = pathlib.Path(__file__).parent.resolve()
    play_directory = os.path.join(top_level, "3600-agents")
    
    print(f"\n{'='*70}")
    print(f"Starting Tournament: {player_a_name} vs {player_b_name}")
    print(f"Games to play: {num_games}")
    print(f"{'='*70}\n")
    
    stats = TournamentStats(player_a_name, player_b_name)
    tournament_start = time.perf_counter()
    
    for game_num in range(num_games):
        game_start = time.perf_counter()
        
        try:
            # IMPORTANT FIX: Always set record=True to avoid history errors
            final_board, trapdoors, spawns, err_a, err_b = play_game(
                play_directory,
                play_directory,
                player_a_name,
                player_b_name,
                display_game=display_games,
                delay=0.0,
                clear_screen=False,
                record=True,  # Changed from save_games to True
                limit_resources=False,
            )
            
            game_time = time.perf_counter() - game_start
            stats.record_game(final_board, game_time, trapdoors, spawns, err_a, err_b)
            
            # Optionally save game history
            if save_games:
                records_dir = os.path.join(play_directory, "tournament_games")
                os.makedirs(records_dir, exist_ok=True)
                out_file = f"{player_a_name}_vs_{player_b_name}_game_{game_num}.json"
                out_path = os.path.join(records_dir, out_file)
                
                with open(out_path, "w") as fp:
                    fp.write(get_history_json(final_board, trapdoors, spawns, err_a, err_b))
            
            stats.print_progress()
            
        except KeyboardInterrupt:
            print("\n\nTournament interrupted by user.")
            break
        except Exception as e:
            print(f"\nError in game {game_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    tournament_time = time.perf_counter() - tournament_start
    
    # Print final summary
    stats.print_summary()
    
    # Save results
    results_dir = os.path.join(play_directory, "tournament_results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"{player_a_name}_vs_{player_b_name}_{timestamp}.json"
    results_path = os.path.join(results_dir, results_file)
    stats.save_to_file(results_path)
    
    return stats


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} <player_a_name> <player_b_name> [num_games] [--save-games] [--display]")
        print(f"\nExample: python3 {sys.argv[0]} MyAgent Yolanda 200")
        print(f"         python3 {sys.argv[0]} MyAgent Yolanda 100 --save-games")
        sys.exit(1)
    
    player_a_name = sys.argv[1]
    player_b_name = sys.argv[2]
    num_games = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 20
    save_games = "--save-games" in sys.argv
    display_games = "--display" in sys.argv
    
    run_tournament(player_a_name, player_b_name, num_games, save_games, display_games)


if __name__ == "__main__":
    main()