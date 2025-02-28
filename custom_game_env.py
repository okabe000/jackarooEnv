import gymnasium as gym
from gymnasium import spaces
import numpy as np
from main import Game
from config import ACTION_SPACE

class CustomGameEnv(gym.Env):
    def __init__(self):
        super(CustomGameEnv, self).__init__()
        self.game = Game()
        self.action_space = spaces.Discrete(len(ACTION_SPACE))
        state_example = self.game.get_STATE()
        self.observation_space = spaces.Dict({
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=state_example.shape, dtype=np.float32)
        })
        self.current_player = 0
        self.current_action_mask = None
        self.deck = list(range(1, 53))
        self.rounds_completed = 0
        self.step_count = 0
        self.invalid_action_count = 0
        self.total_action_count = 0
        self.episode_invalid_actions = 0
        self.episode_total_actions = 0
        print(f"Env. Action size: {len(ACTION_SPACE)}")

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.game.reset()
        self.current_player = 0
        self.rounds_completed = 0
        self.deck = list(range(1, 53))
        self.step_count = 0
        self.episode_invalid_actions = 0
        self.episode_total_actions = 0
        
        # Make sure to deal cards to initialize the game properly
        self.game.deal_cards()
        
        # Update ball positions and distances immediately
        self.game.place_balls_on_board()
        self.game.update_all_distances()
        
        obs = self._get_observation()
        return obs, {}

    def update_all_distances(self):
        """Update distances for all players"""
        for player_id in range(4):
            self.game.update_distances(player_id)

    def _get_observation(self):
        player = self.game.players[self.current_player]
        state = np.array(self.game.get_STATE(player_id=self.current_player), dtype=np.float32)
        state = np.clip(state, -1e6, 1e6)  # Prevent extreme values
        
        # Re-check legal actions right before returning observation
        player.check_legal_actions(self.game)
        valid_action_indices = player.map_actions_to_indices(
            player.actions, ACTION_SPACE, player.get_ball_indices
        )
        
        action_mask = np.zeros(len(ACTION_SPACE), dtype=np.float32)
        if valid_action_indices:
            action_mask[valid_action_indices] = 1.0
        else:
            # If no valid actions, look for BURN_CARD actions
            print("Warnning: the agent was asked to play, but no valid actions was provided, ",valid_action_indices)
            burn_indices = [i for i, action in enumerate(ACTION_SPACE)
                            if action['verb'] == 'BURN_CARD' and
                            any(valid_action['card_value'] == action['card_value']
                                for valid_action in player.actions)]
            if burn_indices:
                action_mask[burn_indices] = 1.0
            else:
                # Last resort - enable the pass action (assuming index 0 is pass)
                action_mask[0] = 1.0
        
        # Ensure at least one action is valid (sanity check)
        if not np.any(action_mask):
            action_mask[0] = 1.0  # Enable pass action as fallback
            print(f"WARNING: No valid actions for player {self.current_player}, enabling pass action")
        
        self.current_action_mask = action_mask
        return {'state': state}

    def step(self, action):
        if action is None:
            raise ValueError("Received None as action in step().")
        
        if isinstance(action, (list, np.ndarray)):
            action = action[0]
        
        self.total_action_count += 1
        self.episode_total_actions += 1
        
        agent = self.game.players[self.current_player]
        agent.check_legal_actions(self.game)
        valid_indices = agent.map_actions_to_indices(agent.actions, ACTION_SPACE, agent.get_ball_indices)
        
        # Store the previous reward to calculate incremental reward later
        previous_reward = agent.reward
        
        # Handle invalid actions by selecting a valid one
        if not self.current_action_mask[action]:
            self.invalid_action_count += 1
            self.episode_invalid_actions += 1
            print(f"WARNING: Invalid action {action} for player {self.current_player}")
            valid_indices_array = np.where(self.current_action_mask == 1)[0]
            if len(valid_indices_array) > 0:
                action = np.random.choice(valid_indices_array)
            else:
                # Force a pass if no valid actions
                action = 0
        
        # Decode and execute the selected action
        selected_action_dict = agent.decode_action(action)
        success = agent.play(selected_action_dict)
        
        # Handle action execution failure
        if not success:
            print(f"Action failed: {selected_action_dict}")
            valid_indices_array = np.where(self.current_action_mask == 1)[0]
            if len(valid_indices_array) > 1:
                valid_indices_array = valid_indices_array[valid_indices_array != action]
                action = np.random.choice(valid_indices_array)
                selected_action_dict = agent.decode_action(action)
                success = agent.play(selected_action_dict)
                if not success:
                    # If retry fails, force a pass
                    print(f"Retry failed: {selected_action_dict} - forcing pass")
                    pass_action = 0  # Assuming 0 is pass
                    pass_action_dict = agent.decode_action(pass_action)
                    success = agent.play(pass_action_dict)
                    if not success:
                        print("Even pass action failed - advancing turn anyway")
            else:
                print("No valid alternatives available - advancing turn")
        
        # Update game state after action
        self.game.place_balls_on_board()
        self.game.update_distances(self.current_player)
        self.step_count += 1
        
        # Check for end of round and deal new cards if needed
        if all(not player.hand for player in self.game.players):
            self.game.deal_cards()
            self.rounds_completed += 1
            # print(f"Round {self.rounds_completed} completed")
        
        # Check win conditions more aggressively
        self._check_win_conditions()
        
        # Calculate reward
        incremental_reward = agent.reward - previous_reward
        
        # Adjust reward for team A (assuming it's the model's team)
        if self.current_player in [0, 2]:
            # Enhanced reward for team A
            incremental_reward *= 1.5
            
            # Additional reward for progress toward winning
            if self.game.players[self.current_player].hasWon:
                incremental_reward += 25
            
            # Add a small step penalty to encourage faster solutions
            incremental_reward -= 0.01
        
        # Clip reward to prevent numerical issues
        reward = np.clip(incremental_reward, -1e6, 1e6)
        
        # Move to next player
        self.current_player = (self.current_player + 1) % 4
        
        # Get observation for next player
        obs = self._get_observation()
        
        # Determine if game has ended
        terminated = self.game.done
        truncated = self.step_count >= 1000  # Force truncation at max steps
        
        # Force game end if taking too long
        if self.step_count >= 900 and not terminated:
            # Check if any team is close to winning, and if so, award them the win
            team_a_progress = sum(1 for p_id in [0, 2] 
                                for ball in self.game.players[p_id].balls 
                                if ball.done)
            team_b_progress = sum(1 for p_id in [1, 3] 
                                for ball in self.game.players[p_id].balls 
                                if ball.done)
            
            if team_a_progress > team_b_progress:
                # Force team A win
                self.game.players[0].hasWon = True
                self.game.players[2].hasWon = True
                self._check_win_conditions()
            elif team_b_progress > team_a_progress:
                # Force team B win
                self.game.players[1].hasWon = True
                self.game.players[3].hasWon = True
                self._check_win_conditions()
            elif self.step_count >= 950:
                # Random winner if no clear advantage and almost at max steps
                team = np.random.choice([1, 2])
                if team == 1:
                    self.game.players[0].hasWon = True
                    self.game.players[2].hasWon = True
                else:
                    self.game.players[1].hasWon = True
                    self.game.players[3].hasWon = True
                self._check_win_conditions()
        
        info = {
            "step_count": self.step_count,
            "reward": reward,
            "rounds_completed": self.rounds_completed,
            "invalid_actions": self.episode_invalid_actions / (self.episode_total_actions + 1e-8),
            "done": terminated or truncated  # Include explicit done flag
        }
        
        if terminated:
            info.update({
                "team_has_won": self.game.team_has_won,
                "winning_team": self.game.winning_team,
                "episode": {"r": reward, "l": self.step_count}
            })
        
        return obs, reward, terminated, truncated, info

    def _check_win_conditions(self):
        """Check if either team has won the game"""
        # Check for team A win (players 0 and 2)
        if self.game.players[0].hasWon and self.game.players[2].hasWon:
            self.game.winning_team = 'A'
            self.game.team_has_won = 1
            self.game.done = True
            # Scale reward based on game length (faster = better)
            win_reward = 50 + (700 - self.step_count) * 0.1
            win_reward = max(50, win_reward)  # Ensure minimum reward
            self.game.players[0].reward += win_reward
            self.game.players[2].reward += win_reward
            self.game.players[1].reward -= win_reward
            self.game.players[3].reward -= win_reward
            # print(f"Team A wins after {self.step_count} steps with reward {win_reward}")
            
        # Check for team B win (players 1 and 3)
        elif self.game.players[1].hasWon and self.game.players[3].hasWon:
            self.game.winning_team = 'B'
            self.game.team_has_won = 2
            self.game.done = True
            win_reward = 50 + (700 - self.step_count) * 0.1
            win_reward = max(50, win_reward)  # Ensure minimum reward
            self.game.players[0].reward -= win_reward
            self.game.players[2].reward -= win_reward
            self.game.players[1].reward += win_reward
            self.game.players[3].reward += win_reward
            # print(f"Team B wins after {self.step_count} steps with reward {win_reward}")

    def render(self, mode="human"):
        """Render the game using the display manager"""
        if self.game.disManger is None:
            self.game.set_display_manager()
        self.game.disManger.display_game(self.game)

    def close(self):
        """Clean up resources"""
        pass

def get_action_mask(env):
    """Return the current action mask for the environment"""
    # Make sure to recalculate action mask if it's None
    if env.current_action_mask is None:
        env._get_observation()
    return env.current_action_mask