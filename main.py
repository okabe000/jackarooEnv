import numpy as np
# from loggerHandler import logger
from config import ACTION_MAP
from player import Player ,ACTION_SAPCE
from ball import Ball
# from displayManger import displayManger
import os 
import time
from datetime import datetime ,timedelta  
import argparse

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
from datetime import datetime, timedelta
import os
from torch.utils.tensorboard import SummaryWriter
from colorama import Fore, Style
import random
# from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import json
from colorama import Fore, Style
from collections import deque
from config import ACTION_SPACE


from jackaroo_gui import EnhancedDisplayManager, patch_player_feedback

class Game:
    def __init__(self):
        self.players = None  # Initialize as placeholder
        self.state_size = 0
        self.action_size = ACTION_SPACE
        self.reset()

        # poplate the players hands
        # self.deal_cards()

    def reset(self):
        # [#.1] board modeling as 2d Array, and the deck as array of one's
        self.action_map = ACTION_MAP
        self.current_player = 0
        self.GameID = 0
        self.device = 'cuda'
        self.done = False
        self.team_has_won = 0
        self.winning_team= None
        self.time_taken = None
        section =   19
        self.obsticles_map     =     [0 for i in range(4 * 19)]
        self.Board =                 np.array([0 for i in range(4* section)])
        self.Board2D =               self.Board.reshape(4,19)
        self.Deck =                  [1 for i in range(1, 52 +1 )]
        self.Balls =                 [[Ball(self, pos=-10, owner=i + 1) for _ in range(4)] for i in range(4)]
        self.safeCells =             [[0 for i in range(4)] for j in range(4)]
        self.tinyObMap =             [[0 for i in range(4)] + [1] for j in range(4)]
        self.free_balls =             [ball for row in self.Balls for ball in row if ball != -10 ]
        self.imprisoned_balls = {ball for row in self.Balls for ball in row if ball.pos == -10}
        self.Balls_TillObsticle =    np.array([[-1 for i in range(4)] for j in range(4)])
        self.Balls_TillWinGate  =    np.array([[-1 for i in range(4)] for j in range(4)])
        self.movable_balls =         np.array([[-10 for i in range(4)] for j in range(4)])
        # self.all_balls_at_base =     [ball.pos for player in self.game.players if player.id != self.id for ball in player.balls if ball.pos == player.base]
        # self.disManger = displayManger(self)
        self.round = 0
        self.currentlyPlaying = 0
        self.players = [
        Player(self,0, team=1,policy='ppo'),
        Player(self,1, team=2,policy='ppo'),
        Player(self,2, team=1,policy='ppo'),
        Player(self,3, team=2,policy='ppo'),]
        self.state_size = len(self.get_STATE())
        self.bases = [0, 19 ,38 , 57]  # Bases for each player
        self.win_gates = [self.bases[0]-2 , self.bases[1]-2 ,self.bases[2]-2 , self.bases[3]-2]  # Win gates for each player
        self.free_balls_cache = None  # Cache for free balls
        
        return self.get_STATE() 
    

    def update_all_distances(self):
        """Update distances for all players."""
        for player_id in range(len(self.players)):
            self.update_distances(player_id)
    
    
    def set_display_manager(self):
            """Initialize disManger only when needed in the main process."""
            from displayManger import displayManger
            self.disManger = displayManger(self)

    def display_game_state(self):
        """Display current game state, initializing disManger if needed."""
        if self.disManger is None:
            self.set_display_manager()
        self.disManger.display_game(self)

    # ... rest of the Game class methods unchanged (step, get_STATE, etc.) ...

    def __getstate__(self):
        """Exclude disManger from pickling."""
        state = self.__dict__.copy()
        state['disManger'] = None
        return state

    def __setstate__(self, state):
        """Restore state without disManger."""
        self.__dict__.update(state)
        self.disManger = None  # Ensure it remains None unless explicitly set

    def step(self, show_game=False):
        """Optimized version of step function with reduced overhead.
        For players using a DQN/PPO policy, collects extra variables from agent.act().
        """
        players = self.players
        # Container to collect agent outputs (only for players with a DQN/PPO policy)
        agent_outputs = []

        # Process each player's turn
        for player in players:
            player.reward = 0
            if player.hasWon:
                # Handle the case for a player who has already won by proxy
                teammate = players[player.teamMate_id]
                original_balls = player.balls  # Backup original balls
                original_id = player.id  # Backup original id
                # Temporarily assign teammate's balls to this player
                player.balls = teammate.balls
                for ball in self.Balls[teammate.id]:
                    ball.owner_2 = player.id + 1
                player.id = teammate.id
                self.place_balls_on_board()
                player.check_legal_actions(self)
                self.update_distances(player.id)
                self.current_player = original_id
                # Capture agent output if applicable
                result = player.play()
                if player.policy == 'ppo' and result is not None:
                    agent_outputs.append(result)
                # Restore original values
                player.balls = original_balls
                player.id = original_id
            else:
                # Normal turn logic
                self.place_balls_on_board()
                player.check_legal_actions(self)
                self.update_distances(player.id)
                if show_game:
                    self.display_game_state()
                result = player.play()
                if player.policy == 'ppo' and result is not None:
                    agent_outputs.append(result)

            # Check if a winning condition has been met
            if players[0].hasWon and players[2].hasWon:
                self.winning_team = 'A'
                self.team_has_won = 1
                self.done = True
                players[0].reward += 50 
                players[2].reward += 50
                players[1].reward -= 50 
                players[3].reward -= 50
                state = self.get_STATE()
                reward = self.players[0].reward
                return state, reward, self.done, agent_outputs
            elif players[1].hasWon and players[3].hasWon:
                self.winning_team = 'B'
                players[0].reward -= 50 
                players[2].reward -= 50
                players[1].reward += 50 
                players[3].reward += 50
                self.team_has_won = 2
                self.done = True
                state = self.get_STATE()
                reward = self.players[0].reward
                return state, reward, self.done, agent_outputs

        # If no winning condition is met, return current state, reward, done, and agent outputs
        state = self.get_STATE()
        reward = self.players[0].reward
        return state, reward, self.done, agent_outputs


    def get_STATE(self, player_id=0): 
        # Rotate the board perspective based on player_id
        relative_board = np.roll(self.Board, -player_id * 19)
        relative_team_board = np.zeros_like(self.Board)
        
        # Calculate team affiliations (team 1 is always "our team")
        for row_idx, row in enumerate(self.Balls):
            for ball in row:
                if ball.pos != -10:  # Only consider active balls
                    # Determine if ball belongs to our team or opponent
                    ball_team = self.players[row_idx].team
                    player_team = self.players[player_id].team
                    team_value = 1 if ball_team == player_team else 2
                    if 0 <= ball.pos < len(relative_team_board):
                        relative_team_board[ball.pos] = team_value
        
        # Roll the team board to match player perspective
        relative_team_board = np.roll(relative_team_board, -player_id * 19)
        
        # Normal board (without team markings)
        normal_board_state = relative_board.flatten()
        team_board_state = relative_team_board.flatten()

        # Get player-specific data
        current_player = self.players[player_id]
        teammates_id = current_player.teamMate_id
        
        # Track which players are burned
        burned_players = [(self.players[(player_id + i) % 4].isBurned) for i in range(4)]
        
        # Reorder balls to make current player's balls first
        ordered_balls = []
        for i in range(4):
            adjusted_idx = (player_id + i) % 4
            ordered_balls.extend(self.Balls[adjusted_idx])
        
        # Enhanced ball features from reordered perspective
        ball_features = []
        for i, ball in enumerate(ordered_balls):
            owner_idx = i // 4  # Which player owns this ball (0=current, 1=next, etc)
            owner_team = 1 if self.players[(player_id + owner_idx) % 4].team == current_player.team else 2
            in_safe = 1 if ball.pos in self.safeCells[(player_id + owner_idx) % 4] else 0
            
            features = [
                ball.pos,
                ball.distance_to_obstacle,
                ball.distance_to_win_gate,
                ball.distance_to_playerWinGateOb,
                int(ball.done),
                in_safe,
                owner_team
            ]
            ball_features.extend(features)
        ball_state = np.array(ball_features)
        
        # Team statistics from player perspective
        my_team_kills = opponent_team_kills = 0
        my_team_free = opponent_team_free = 0
        my_team_near_win = opponent_team_near_win = 0

        for p_idx, player in enumerate(self.players):
            is_teammate = player.team == current_player.team
            if is_teammate:
                my_team_kills += player.kill_count
                my_team_free += len([b for b in player.balls if b.pos != -10])
                my_team_near_win += sum(1 for ball in player.balls 
                                        if not ball.done and ball.distance_to_win_gate <= 13)
            else:
                opponent_team_kills += player.kill_count
                opponent_team_free += len([b for b in player.balls if b.pos != -10])
                opponent_team_near_win += sum(1 for ball in player.balls 
                                            if not ball.done and ball.distance_to_win_gate <= 13)

        # Create composite state
        STATE = np.concatenate([
            normal_board_state,
            team_board_state,
            ball_state,
            self.obsticles_map,
            self.movable_balls.flatten(),
            [my_team_kills, opponent_team_kills],
            [my_team_free, opponent_team_free],
            [my_team_near_win, opponent_team_near_win],
            burned_players,
            [1, 3],  # team players number
            np.array([self.round]),
            np.array(self.Deck),
        ])

        assert STATE is not None, "get_STATE() produced None"
        return STATE


    def place_balls_on_board(self):
        """Place balls on the board based on their position, using the 1D array."""
        # Reset the board (we can simply overwrite specific positions instead of reinitializing)
        # self.check_duplicate_ball_positions()
        self.Board.fill(0)  # Set all elements to zero

        # Reset the safeCells array in one line
        self.safeCells = [[0] * 4 for _ in range(4)]

        # Iterate over balls and place them in Board and safeCells
        for row_idx, ballRow in enumerate(self.Balls):
            for ball in ballRow:
                if ball.pos >= 0:
                    # Ball position is valid, place it on the board
                    self.Board[ball.pos] = ball.owner  # Use ball's owner as the value on the board
                elif -4 <= ball.pos < 0:
                    # For negative positions, place it in safeCells (inverted position mapping)
                    inversedPos = -ball.pos - 1
                    self.safeCells[row_idx][inversedPos] = ball.owner


    def invalidate_free_balls_cache(self):
        """Invalidate the free balls cache when ball positions change."""
        self.free_balls_cache = None

    def get_free_balls(self):
        """Get cached free balls or update cache if needed."""
        # if self.free_balls_cache is None:
        #     self.free_balls_cache = [ball for player in self.players for ball in player.balls if ball.pos != -10]
        return [ball for player in self.players for ball in player.balls if ball.pos != -10]
    
    def update_distances(self, current_player=None):
        """Update distances for all balls."""
        
        # Update only necessary parts if there were changes in the state
        if self.board_state_changed():  # Check if board state has changed (track relevant changes)
            self.place_balls_on_board()
        
        self.update_tinyObMap()
        if current_player is not None:
            self.update_obstacle_map(current_player)

        self.get_free_balls()

        # Perform ball updates only if their state has changed
        for player_balls in self.Balls:
            for ball in player_balls:
                if ball.has_state_changed():  # Check if the ball's state has changed
                    ball.isDone()
                    ball.update_till_obstacle()
                    ball.update_till_winGate()


    def board_state_changed(self):
        """Check if the board state has changed, triggering updates."""
        # Track changes to board and safeCells, optimize by comparing only the required parts
        return not np.array_equal(self.Board, self.compute_board_state())  # Efficient array comparison

    def compute_board_state(self):
        """Compute and return the current state of the board for comparison."""
        # Return a tuple representation of the board and safeCells for comparison
        state = {
            'Board': tuple(self.Board.flatten()),  # Use flattened to make it a 1D tuple for easy comparison
            'safeCells': tuple(tuple(row) for row in self.safeCells)
        }
        return state

    def update_tinyObMap(self):
        """Update obstacles based on safeCells and tinyObMap."""
        # Only update if safeCells have changed
        safeCells = self.safeCells
        tinyObMap = self.tinyObMap

        # Avoid resetting the entire tinyObMap unless necessary
        for i in range(len(safeCells)):
            for j in range(len(safeCells[i])):
                if safeCells[i][j] != tinyObMap[i][j]:  # Only update if changed
                    tinyObMap[i][j] = 1 if safeCells[i][j] != 0 else 0

        # Update the last column of tinyObMap
        for row in tinyObMap:
            row[-1] = 1


    def update_obstacle_map(self, current_player):
        """Update obstacle map including win gates and safe cells."""
        # Only update if relevant parts have changed
        self.obsticles_map = [0] * len(self.Board)
        currentWinGate = self.win_gates[current_player] + 1
        self.obsticles_map[currentWinGate] = -1

        bases = [i * 19 for i in range(4)]
        for base in bases:
            if self.Board[base] != 0:
                ballOcup = self.Board[base]
                base_index = bases.index(base)
                if ballOcup == base_index + 1:
                    self.obsticles_map[base] = 1


    def refill_deck(self):
        self.Deck = [1 for i in range(52)]

    def reduce_card(self, card_number):
        """Reduce the card number to a value between 1 and 13."""
        if not (1 <= card_number <= 52):
            raise ValueError("Card number must be between 1 and 52")

        # Determine suit and rank of the card
        suits = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
        rank = (card_number - 1) % 13 + 1  # 1 (Ace) to 13 (King)
        suit = suits[(card_number - 1) // 13]

        # Map the rank and suit to the corresponding class
        if rank == 1:  # Ace
            return 1
        elif 2 <= rank <= 10:  # Cards 2-10
            return rank
        elif rank == 11:  # Jack
            if suit in ['Clubs', 'Spades']:  # Black Jacks only
                return 11
            else:  # Red Jacks only
                return 14
        elif rank == 12:  # Queen
            return 12
        elif rank == 13:  # King
            return 13

    def deal_cards(self):
        """
        Deals cards to the players. The number of cards to deal changes after each round.
        """
        players = self.players
        # Increase the round number and track it
        self.round += 1
        # Refill the deck if all cards have been used (deck is all zeros)
        if all(self.Deck[i] == 0 for i in range(52)):
            self.refill_deck()  # Refill the deck
        
        # select randomly from the avaliable cards
        # flattened_Deck = [card for row in self.Deck for card in row]
        available_cards =  [i+1 for i, value in enumerate(self.Deck) if value == 1]

         # first deal in the round
            # deal for each 5 cards
        for player in players:
            player.hand.clear()  # Clear the player's hand before dealing new cards
        for x in range( 5 if self.round % 3 ==1 else 4):
            for player in players:
                random_card = random.choice(available_cards)
                choosen_card =  self.reduce_card(random_card)
                player.hand.append(choosen_card)
                self.Deck[random_card -1] = 0       # there is no card that is zero so to avoid the issue subtraced one after added it
                available_cards.remove(random_card)
       




    def check_duplicate_ball_positions(self):
        """
        Check for duplicate ball positions in the game.
        Raises descriptive errors for:
            - Collisions at positive or zero positions.
            - Overlapping player balls in negative safe cell positions.
        Ignores positions set to -10.
        """
        seen_positions = {}

        for player_idx, player_balls in enumerate(self.Balls):
            for ball_idx, ball in enumerate(player_balls):
                pos = ball.pos

                # Ignore positions set to -10
                if pos == -10:
                    continue

                # For negative positions, check duplicates within the player's own balls
                if pos < 0:
                    if pos in seen_positions and seen_positions[pos]['type'] == 'negative' and seen_positions[pos]['player_idx'] == player_idx:
                        raise ValueError(
                            f"Overlap Detected: Player {player_idx + 1} has multiple balls in the same safe cell at position {pos}. "
                            f"Balls: {seen_positions[pos]['ball']} and Ball {ball_idx + 1}."
                        )
                    seen_positions[pos] = {'type': 'negative', 'player_idx': player_idx, 'ball': f'Player {player_idx + 1} Ball {ball_idx + 1}'}

                # For positive or zero positions, check duplicates globally
                elif pos >= 0:
                    if pos in seen_positions and seen_positions[pos]['type'] == 'positive':
                        raise ValueError(
                            f"Collision Detected: Balls collided at position {pos}. "
                            f"Balls: {seen_positions[pos]['ball']} and Player {player_idx + 1} Ball {ball_idx + 1}."
                        )
                    seen_positions[pos] = {'type': 'positive', 'player_idx': player_idx, 'ball': f'Player {player_idx + 1} Ball {ball_idx + 1}'}


    def display_game_state(self):
        """Display current game state."""
        self.disManger.display_game(self)

def setUpGameForTesting(game,handsOnly=False):
    players = game.players
    players[0].hand = [1, 13, 7, 5]
    players[1].hand = [1, 1, 1, 1]
    players[2].hand = [1, 1, 1, 1]
    players[3].hand = [1, 1, 1, 1]

    if not handsOnly:
        ballPostions = [[0, -4, 8, -10], [5, 10, -10, -10], [13, 12, -3, -4], [1, -10, -10, -10]]
        for ballRow in game.Balls:
            for ball in ballRow:
                ball.pos = ballPostions[ball.owner - 1][ballRow.index(ball)]
        return game
    return game

def minmial_gameOutput(game):
    for row in game.safeCells:
        print(f'PLAYER :{row}') 

COLORS = {
    1: "\033[91m",  # Red
    2: "\033[92m",  # Green
    3: "\033[94m",  # Blue
    4: "\033[33m",  # Brown (Yellow)
    "reset": "\033[0m"  # Reset to default
}

def print_safe_cells(game):
    """
    Print the current game state and safeCells with a delay for better visibility.
    """
    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f'\n -- [ Round : {game.round:04d} ]--\n')
    for row in game.safeCells:
        colored_row = " ".join(
            f"{COLORS.get(num, COLORS['reset'])}{num}{COLORS['reset']}" for num in row
        )
        print(f"PLAYER: {colored_row}")

def main():
    game = Game()
    # game = setUpGameForTesting(game)
    game.display_manager = EnhancedDisplayManager(game)

    players = game.players
    for player in players:
        patch_player_feedback(player)
    # for round in range(100):
    # game_start_time = time.time()

    while True:
        # game = setUpGameForTesting(game, handsOnly=True)
        game.deal_cards()
        cards_this_round = 5 if game.round % 3 == 1 else 4
        for _ in range(cards_this_round):
            for player in players:
                if player.hasWon:
                    # Create a temporary proxy for the teammate's balls
                    teammate = players[player.teamMate_id]
                    original_balls = player.balls  # Backup original balls
                    original_id = player.id  # Backup original balls
                    # print(player.teamMate_id)
                    # Temporarily assign teammate's balls to this player
                    player.balls = teammate.balls
                    for ball in game.Balls[teammate.id]:
                        ball.owner_2 = player.id +1

                    # Execute the turn logic as if playing for the teammate
                    # changing the player id temprorly during his play, but doing it once 
                    player.id = teammate.id
                    # if player.changed_id:
                    #     player.changed_id = False
                    game.place_balls_on_board()
                    player.check_legal_actions(game)
                    game.update_distances(player.id)
                    game.display_manager.display_game(game, view_sections=None, GUI=True)
                    # game.display_game_state()
                    # time.sleep(0.5)

                    game.current_player = original_id

                    player.play()
                    # print(f"a player conneected. player_1 rewards after play: {players[0].reward} " )
                    # for player in players:
                    #     print(f"player {player.id} rewards after play: {player.reward} " )
                    player.balls = original_balls
                    player.id = original_id
                else:
                    # Normal turn logic for players who haven't won
                    game.place_balls_on_board()
                    player.check_legal_actions(game)
                    game.update_distances(player.id)
                    game.display_manager.display_game(game, view_sections=None, GUI=True)
                    # game.display_game_state()
                    # time.sleep(0.5)
                    game.current_player = player.id

                    player.play()


                if   players[0].hasWon and players[2].hasWon:
                    # print(f'team A wins!\n\nRound took:{game.round}') 
                    game.winning_team = 'A'
                    # game.time_taken =  time.time() - game_start_time
                    players[0].reward += 50 
                    players[2].reward += 50
                 
                    players[1].reward -= 50 
                    players[3].reward -= 50
                    # print(f"player 1 rewards end-game: {players[0].reward}\nKillCount:{players[0].kill_count}\nplays:{players[0].plays}. milsonte given{players[0].rewards_given} " )
                    for player in players:
                        print(f"player {player.id}:{player.number} rewards end-game: {player.reward} -- KillCount:{player.kill_count} -- plays:{players[0].plays} --  milsonte given{player.rewards_given} " )

                    # register_GameInfo(game)
                    return 1

                elif players[1].hasWon and players[3].hasWon: 
                    # print(f'team B wins!\n\nRound took:{game.round}') 
                    game.winning_team = 'B'
                    # game.time_taken =  time.time() - game_start_time
                    players[0].reward -= 50 
                    players[2].reward -= 50
                 
                    players[1].reward += 50 
                    players[3].reward += 50
                    
                    for player in players:
                        print(f"player {player.id}:{player.number} rewards end-game: {player.reward} -- KillCount:{player.kill_count} -- plays:{players[0].plays} --  milsonte given{player.rewards_given} " )

                    # register_GameInfo(game)
                    return 2


def register_GameInfo(info):
    with open('Games_log', 'a') as file:
        # Calculate total time in MM:SS
        minutes, seconds = divmod(int(info.time_taken), 60)
        avg_time_per_round = info.time_taken / info.round if info.round > 0 else 0
        for player in info.players:
            players_rewards = player.reward
        # Write formatted game information to the log
        file.write(
            f"reward for players{players_rewards}"
            f"\nWinning Team: {info.winning_team} | "
            f"Rounds Taken: {info.round:04d} | "
            f"Time Taken: {minutes:02d}:{seconds:02d} | "
            f"Avg Time/Round: {avg_time_per_round} s\n"
        )

if __name__ == "__main__":
    tests = range(1)
    # for test in tests:
    for test in tests:
        print(f"Running Test {test + 1}")
            
        N_games = 10
        Games=range(N_games) # 10 games
        games_result = []
        for gameVar in Games:
            # print(f'Game:{gameVar:04d}\n')
            game_result  = main()
            games_result.append(game_result)
            print(f'Game Result:{game_result}\n')
        print(f'All Games Completed, games:{N_games}')
        A_wins = games_result.count(1)
        B_wins = games_result.count(2)
        total_count = len(games_result)
        ones_percentage = (A_wins / total_count) * 100
        twos_percentage = (B_wins / total_count) * 100
        print(f"Team A wins: {A_wins} ({ones_percentage:.2f}%) <-> Team B wins: {B_wins} ({twos_percentage:.2f}%)")




def visualize_q_values_3d(q_value_history: list,
                          max_episodes: int = 500,
                          max_steps: int = 200,
                          smooth_sigma: float = 1.2,
                          scale_factor: float = 1.0,
                          figsize: tuple = (16, 9),  # Optional: Increase for better clarity
                          quality: int = 3,  # Higher quality
                          axis_mapping: dict = None,
                          save_folder: str = "figures",  # Save folder for the figure
                          save_filename: str = "q_values_3d_plot.png") -> None:
    if not q_value_history or not any(q_value_history):
        print("No Q-value data to visualize.")
        return

    if axis_mapping is None:
        axis_mapping = {'x': 'episodes', 'y': 'q_values', 'z': 'steps'}

    # Adjusted quality settings for better resolution
    quality_settings = {
        1: {'rcount': 50, 'ccount': 50, 'antialias': False},
        2: {'rcount': 100, 'ccount': 100, 'antialias': True},
        3: {'rcount': 300, 'ccount': 300, 'antialias': True},  # Improved resolution
    }
    render_params = quality_settings.get(quality, quality_settings[2])

    num_episodes = len(q_value_history)
    ep_step = max(1, num_episodes // max_episodes)
    effective_sigma = smooth_sigma * (num_episodes / max_episodes)**0.5

    sampled_episodes = []
    step_counts = [len(ep) for ep in q_value_history]
    max_original_steps = max(step_counts)
    step_stride = max(1, max_original_steps // max_steps)

    all_values = []
    for ep in q_value_history[::ep_step]:
        if len(ep) == 0:
            continue
        sampled_steps = ep[::step_stride]
        sampled_episodes.append(sampled_steps)
        all_values.extend([v for step in sampled_steps for v in step])

    if not all_values:
        print("No valid Q-values to visualize.")
        return

    # Softmax normalization for Q-values
    def softmax(values):
        values = np.array(values)
        exp_values = np.exp(values - np.max(values))
        return exp_values / np.sum(exp_values, axis=0, keepdims=True)

    final_episodes = len(sampled_episodes)
    final_steps = max(len(ep) for ep in sampled_episodes)
    num_actions = len(sampled_episodes[0][0]) if sampled_episodes[0] else 0
    q_array = np.full((final_episodes, final_steps, num_actions), np.nan)

    for ep_idx, episode in enumerate(sampled_episodes):
        for step_idx, step_values in enumerate(episode):
            if step_idx >= final_steps:
                continue
            q_array[ep_idx, step_idx] = softmax(step_values)

    if np.isnan(q_array).all():
        print("All Q-values are NaN. Visualization skipped.")
        return

    # Aggregated Q-values (taking max across actions)
    aggregated = np.nanmax(q_array, axis=2)
    smoothed = gaussian_filter(aggregated, sigma=[effective_sigma, 1], mode='nearest')

    axes_data = {
        'episodes': np.arange(0, num_episodes, ep_step),
        'steps': np.arange(0, max_original_steps, step_stride),
        'q_values': smoothed.T
    }

    try:
        X = axes_data[axis_mapping['x']]
        Y = axes_data[axis_mapping['y']]
        Z = axes_data[axis_mapping['z']]

        if axis_mapping['z'] == 'q_values':
            X, Y = np.meshgrid(X, np.arange(len(Z)))
        else:
            X, Z = np.meshgrid(X, np.arange(len(Z)))
            Y = smoothed.T

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface with higher resolution and anti-aliasing
        surf = ax.plot_surface(X, Y, Z,
                               cmap='viridis',
                               linewidth=0,
                               antialiased=render_params['antialias'],
                               rcount=render_params['rcount'],
                               ccount=render_params['ccount'])

        ax.set_xlabel(f'{axis_mapping["x"].capitalize()} (X-axis) →')
        ax.set_ylabel(f'{axis_mapping["y"].capitalize()} (Y-axis) →')
        ax.set_zlabel(f'{axis_mapping["z"].capitalize()} (Z-axis) →')
        ax.view_init(elev=30, azim=-135)

        # Optional: Add color bar for better understanding of Q-values
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        plt.tight_layout()

        # Save the figure
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, save_filename)
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved at: {save_path}")

        plt.show()
    except Exception as e:
        print(f"Error during visualization: {e}")



# def train_dqn_agent(num_episodes=1000, batch_size=512, parallel_episodes=2, 
#                    resume_training=True, log_to_tensorboard=False):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     import torch.cuda as cuda

#     # Initialize agent and TensorBoard writer if logging is enabled
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     unique_id = f"instance_{os.getpid()}_{timestamp}"  # Unique ID for the process

#     log_dir = f'runs/jackaroo_dqn_{unique_id}' if log_to_tensorboard else None
#     writer = SummaryWriter(log_dir) if log_to_tensorboard else None
#     training_start_time = time.time()
    
#     # # Start the user input thread
#     # input_thread = threading.Thread(target=user_input_thread, daemon=True)
#     # input_thread.start()
    
#     team_a_wins = 0
#     team_b_wins = 0
#     total_reward = 0
#     q_value_history = []  # Stores Q-values for each step in every episode
#     segment_size = num_episodes // 10  # Divide episodes into 5 segments (XX% each)
#     epsilon_reset_value = 1.0  # Reset value for epsilon at the start of each segment
#     moving_avg_rewards = []  # Track moving average rewards
#     moving_avg_losses = []  # Track moving average losses

#     agent_memory_fullness = 0  # Track memory usage percentage
#     # Tracking variables for epsilon adjustment
#     recent_rewards = []
#     last_adjustment_step = 0
#     recent_losses = []

#     # Tracking variables for adaptive tau
#     tau_value = 0
#     tau = 0.01  # Initial tau
#     tau_increase_step = 0.01  # How much to increase tau if needed
#     tau_max = 1.0  # Maximum value for ta


#     # Ensure the agent and networks are on GPU
#     device = torch.device('cuda' if cuda.is_available() else 'cpu')



#     game = Game()

#     # agent.q_network.to(device)
#     # agent.target_network.to(device)
#     print(f'state input   lenght: {len(game.get_STATE())}')
#     print(f'Action output lenght: {len(game.action_size)}')
#     agent = game.players[0].agent

#     # Determine starting episode and win counts
# # Load elapsed time correctly when resuming
#     if resume_training:
#         start_episode, team_a_wins, team_b_wins, saved_elapsed_time = load_training_state(agent)
#         training_start_time = time.time() - saved_elapsed_time  # Adjust the start time
#     else:
#         start_episode = 1
#         team_a_wins = 0
#         team_b_wins = 0
#         training_start_time = time.time()  # Start fresh


#     # Load latest model if resuming training
#     if resume_training:
#         load_latest_model(agent)

#     global_episode_counter = start_episode
#     for episode_batch_start in range(start_episode, start_episode + num_episodes, parallel_episodes):
#         # Apply interactive updates
#         if interactive_params["epsilon"] is not None:
#             agent.epsilon = interactive_params["epsilon"]
#             interactive_params["epsilon"] = None  # Reset to avoid applying repeatedly
        
#         if interactive_params["tau"] is not None:
#             agent.tau = interactive_params["tau"]
#             interactive_params["tau"] = None  # Reset to avoid applying repeatedly
        
#         if interactive_params["batch_size"] is not None:
#             agent.batch_size = interactive_params["batch_size"]
#             interactive_params["batch_size"] = None  # Reset to avoid applying repeatedly

#         batch_rewards = torch.zeros(parallel_episodes, device=device)
#         batch_done_flags = [False] * parallel_episodes
#         batch_games = [Game() for _ in range(parallel_episodes)]

#         # Initialize games and reset states
#         for game_instance in batch_games:
#             game_instance.reset()

#         while not all(batch_done_flags):
#             states = []
#             actions = []
#             rewards = []
#             next_states = []
#             dones = []

#             for idx, game_instance in enumerate(batch_games):
#                 if batch_done_flags[idx]:
#                     continue
                
#                 try :
#                     state = game_instance.get_STATE()
#                     game_instance.deal_cards(game_instance.players)
#                     cards_this_round = 5 if game_instance.round % 3 == 1 else 4
#                     for _ in range(cards_this_round):
#                         if not game_instance.done:
#                             next_state, reward, done = game_instance.step(show_game=False)
#                             action = agent.choosen_action
#                             # When storing experiences:
#                             agent.store_experience(
#                                 state,  # Store in CPU memory
#                                 action,
#                                 reward,
#                                 next_state,
#                                 done
#                             )                        
#                             states.append(state)
#                             actions.append(action)
#                             rewards.append(reward)
#                             next_states.append(next_state)
#                             dones.append(done)
#                             batch_rewards[idx] += reward
#                             if done:
#                                 batch_done_flags[idx] = True
#                 except Exception as e:
#                     print(f"Error during batch episode {episode_batch_start}: {e}, continuing to next game.")
#                     batch_done_flags[idx] = True

#             # Move all tensors to the device (GPU or CPU) where the model is located
#             if states:
#                 # states = np.array(states)
#                 # actions = np.array(actions)
#                 # rewards = np.array(rewards)
#                 # next_states = np.array(next_states)
#                 # dones = np.array(dones)
#                 # print(rewards)
                
#                 states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
#                 actions = torch.tensor(np.array(actions), dtype=torch.long, device=device)
#                 rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=device)
#                 next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
#                 dones = torch.tensor(np.array(dones), dtype=torch.float32, device=device)

#             # Batch process experiences
#             if agent.memory.n_entries >= agent.batch_size:
#                 loss, entropy = agent.replay(agent.tau)
#                 recent_losses.append(loss)
#                 if len(recent_losses) > 250:
#                     recent_losses.pop(0)
#             else:
#                 loss, entropy = 0.0, 0.0
#         # Update target network once per batch
#         agent.update_target_network_polyak(tau=agent.tau)

#         # Post-batch processing
#         for idx, game_instance in enumerate(batch_games):
#             if game_instance.team_has_won == 1:
#                 team_a_wins += 1
#             elif game_instance.team_has_won == 2:
#                 team_b_wins += 1
#         # for rewardy in batch_rewards:
#         #     print(rewardy)
#         # for reardy2 in rewards:
#         #     print(reardy2)
#         total_games = episode_batch_start + parallel_episodes
#         win_rate_a = (team_a_wins / total_games) * 100 if total_games > 0 else 0
#         win_rate_b = (team_b_wins / total_games) * 100 if total_games > 0 else 0

#         # If using TensorBoard, log the loss and reward
#         if log_to_tensorboard:
#             log_game_info(episode_batch_start, num_episodes, loss, win_rate_a, win_rate_b, 
#                           team_a_wins, team_b_wins, batch_rewards.sum().item(), agent.epsilon, entropy, steps=0)

#         recent_rewards.append(batch_rewards.sum().item())
#         # Save model every 100 or 1000 episodes
#         elapsed_time = time.time() - training_start_time
#         elapsed_time_str = str(timedelta(seconds=int(elapsed_time)))  # Use timedelta correctly

#         if (global_episode_counter - 1) % 100 < parallel_episodes:
#             save_training_state(agent, global_episode_counter, team_a_wins, team_b_wins, elapsed_time)

#         # Increment global episode counter
#         global_episode_counter += parallel_episodes


#         # Adaptive tau adjustment
#         if len(recent_losses) > 50:
#             moving_avg_loss = sum(recent_losses[-50:]) / len(recent_losses[-50:])
#             if len(moving_avg_losses) > 250:
#                 prev_avg_loss = moving_avg_losses[-1]
#                 if abs(moving_avg_loss - prev_avg_loss) < 0.01 and entropy < 0.5:  # Loss stagnation
#                     agent.tau = min(agent.tau + tau_increase_step, tau_max)  # Increase tau
#             moving_avg_losses.append(moving_avg_loss)

#         # Adaptive epsilon adjustment
#         recent_rewards.append(batch_rewards.sum().item())
#         if len(recent_rewards) > 250:
#             recent_rewards.pop(0)
#         if len(recent_rewards) == 250:
#             avg_reward = sum(recent_rewards) / 250
#             prev_avg_reward = moving_avg_rewards[-1] if moving_avg_rewards else 0
#             if avg_reward <= prev_avg_reward + 0.01:  # No significant improvement
#                 agent.epsilon = min(agent.epsilon + 0.05, 1.0)  # Increase epsilon

#         print(
#             f"{Fore.CYAN}[Batch Start {episode_batch_start:04}/{start_episode + num_episodes - 1}]{Style.RESET_ALL} "
#             f"{Fore.YELLOW}Loss: {loss:8.4f}{Style.RESET_ALL} "  # Fixed-width for loss
#             f"{Fore.GREEN}Batch Reward: {batch_rewards.sum().item():10.2f}{Style.RESET_ALL} "  # Fixed-width for reward
#             f"{Fore.MAGENTA}Epsilon: {agent.epsilon:6.4f}{Style.RESET_ALL} "  # Fixed-width for epsilon"
#             f"{Fore.LIGHTGREEN_EX if win_rate_a >= win_rate_b else Fore.LIGHTRED_EX}"
#             f"Win Rate A: {win_rate_a:6.2f}%{Style.RESET_ALL} "
#             f"{Fore.LIGHTGREEN_EX if win_rate_b >= win_rate_a else Fore.LIGHTRED_EX}"
#             f"Win Rate B: {win_rate_b:6.2f}%{Style.RESET_ALL} "
#             f"{Fore.CYAN}Elapsed: {elapsed_time_str}{Style.RESET_ALL}"
#         )

#         moving_avg_reward = sum(recent_rewards) / len(recent_rewards)
#         moving_avg_rewards.append(moving_avg_reward)
#         agent_memory_fullness = (agent.memory.n_entries / agent.memory.capacity) * 100  # Now uses SumTree's capacity

#         if log_to_tensorboard:
#             writer.add_scalar("Loss", loss, episode_batch_start)
#             if moving_avg_losses:  # Ensure there is at least one moving average value
#                 writer.add_scalar("Moving Avg. Loss", moving_avg_losses[-1], episode_batch_start)            
#             writer.add_scalar("Batch Reward", batch_rewards.sum().item(), episode_batch_start)
#             writer.add_scalar("Epsilon", agent.epsilon, episode_batch_start)
#             writer.add_scalar("TAU", agent.tau, episode_batch_start)
#             writer.add_scalar("Steps", game_instance.round, episode_batch_start)
#             writer.add_scalar("Entropy", entropy, episode_batch_start)
#             writer.add_scalar("Win Rate Team A", win_rate_a, episode_batch_start)
#             writer.add_scalar("Memory Fullness (%)", agent_memory_fullness, episode_batch_start)
#             writer.add_scalar("Moving Average Reward (100 episodes)", moving_avg_reward, episode_batch_start)

#     torch.save(agent.q_network.state_dict(), 'dqn_agent_model_final.pth')
#     if log_to_tensorboard:
#         writer.close()



# def save_training_state(agent, episode, team_a_wins, team_b_wins, elapsed_time):
#     # Save the training state to a JSON file
#     state = {
#         "last_episode": episode,
#         "elapsed_time": elapsed_time,  # Store updated elapsed time
#         "team_a_wins": team_a_wins,
#         "team_b_wins": team_b_wins
#     }
#     with open('training_state.json', 'w') as f:
#         json.dump(state, f)
#     print(f"Training state saved at episode {episode}.")
    
#     # Save the model
#     save_model(agent, episode)

# import torch
# import os

# def save_model(agent, episode, directory="models"):
#     """Saves the DQN agent's model with the episode number in the filename."""
#     os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
#     model_path = os.path.join(directory, f"dqn_model_episode_{episode}.pth")
#     torch.save(agent.q_network.state_dict(), model_path)
#     print(f"Model saved at {model_path}")

# def load_training_state(agent):
#     try:
#         # Load the training state from the JSON file
#         with open('training_state.json', 'r') as f:
#             state = json.load(f)

#         # Load model from the latest episode
#         episode = state.get('last_episode', 1)
#         E_time = state.get('elapsed_time', 0)
#         team_a_wins = state.get('team_a_wins', 0)
#         team_b_wins = state.get('team_b_wins', 0)

#         return episode, team_a_wins, team_b_wins ,E_time

#     except FileNotFoundError:
#         # If the state file is not found, return defaults
#         return 1, 0, 0, 0
    


# def load_latest_model(agent):
#     models = glob.glob('models/dqn_model_episode_*.pth')
#     if models:
#         latest_model = max(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
#         agent.q_network.load_state_dict(torch.load(latest_model))
#         print(f"Loaded model from {latest_model}")
#     else:
#         print("No model found to load.")

# def log_game_info(episode, total_episodes, loss, win_rate_a, win_rate_b, team_a_wins, team_b_wins, total_reward, epsilon, entropy, steps):
#     with open('gamesHistory.log', 'a') as f:
#         f.write(f"\033[94m[Episode {episode:04}/{total_episodes}]\033[0m")
#         f.write(f" -- Loss: {loss:.6f}")
#         f.write(f" -- Reward: {total_reward}")
#         # f.write(f" -- AvgReward: {avg_reward}")
#         f.write(f" -- Entropy: {entropy:.2f}")
#         f.write(f" -- Epsilon: {epsilon:.4f}")
#         f.write(f" -- Teams Winning Rates: {win_rate_a:.2f}% : {win_rate_b:.2f}%")
#         f.write(f" -- Steps: {steps}")
#         f.write(f" -- Teams Wins: {team_a_wins} : {team_b_wins}")
#         f.write("\n")


# # Shared dictionary for updating parameters
# interactive_params = {
#     "epsilon": None,
#     "tau": None,
#     "batch_size": None
# }

# # Function to handle interactive parameter adjustments
# def interactive_adjustments():
#     while True:
#         keyboard.wait('ctrl+g')  # Wait for Ctrl+G to be pressed
#         print("\n--- Modify Hyperparameters ---")
#         print("1: Update Epsilon")
#         print("2: Update Tau")
#         print("3: Update Batch Size")
#         print("4: Exit Interactive Mode")
#         try:
#             option = int(input("Choose an option: ").strip())
#             if option == 1:
#                 new_value = float(input("Enter new epsilon value (0 to 1): ").strip())
#                 interactive_params["epsilon"] = new_value
#                 print(f"Updating epsilon to {new_value}")
#             elif option == 2:
#                 new_value = float(input("Enter new tau value (0 to 1): ").strip())
#                 interactive_params["tau"] = new_value
#                 print(f"Updating tau to {new_value}")
#             elif option == 3:
#                 new_value = int(input("Enter new batch size: ").strip())
#                 interactive_params["batch_size"] = new_value
#                 print(f"Updating batch size to {new_value}")
#             elif option == 4:
#                 print("Exiting interactive mode.")
#                 break
#             else:
#                 print("Invalid option. Please try again.")
#         except ValueError:
#             print("Invalid input. Please enter a valid number.")

# # Start the interactive adjustments thread
# adjustments_thread = threading.Thread(target=interactive_adjustments, daemon=True)
# adjustments_thread.start()


# def main():
#     # Create argument parser for better command-line interface
#     parser = argparse.ArgumentParser(description="Train or play the DQN agent in the Jackaroo game.")
    
#     # 'mode' argument (train/play) with default value of 'train'
#     parser.add_argument(
#         '--mode', type=str, choices=['train', 'play'], default='train',
#         help="Choose 'train' to train the DQN agent or 'play' to play the game. Default is 'train'."
#     )
    
#     # 'episodes' argument with default value of 10000
#     parser.add_argument(
#         '--ep', type=int, default=10000,
#         help="Number of episodes for training. Default is 10000."
#     )
    
#     # 'resume' argument to specify if training should resume from saved state
#     parser.add_argument(
#         '--resume', action='store_true',
#         help="Resume training from the latest saved model if available."
#     )

#     # Parse arguments
#     args = parser.parse_args()

#     # Start training or playing based on the 'mode' argument
#     if args.mode == 'train':
#         print(f"Starting training session...")
#         print(f"Number of episodes: {args.ep}")
#         print(f"Resume training: {'Yes' if args.resume else 'No'}")
#         PPOAgent.train_ppo_agent(Game,num_episodes=args.ep,log_to_tensorboard=True ,resume_training=args.resume)
#         # train_dqn_agent(num_episodes=args.ep, resume_training=args.resume,log_to_tensorboard=True)
#     elif args.mode == 'play':
#         print("Starting the game in play mode...")
#         print('Out of service! WIP')
#     else:
#         print("Invalid mode. Use '--mode train' or '--mode play'. Exiting.")

# if __name__ == '__main__':
#     main()