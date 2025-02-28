# from turtle import pos

# from pyglet import value
# from dqnAgnet import DQNAgent, PPOAgent
# from loggerHandler import logger
from config import ACTION_MAP
import random
from ball import Ball
import dqnAgnet 
import itertools
from dqnAgnet import PPOAgent
import numpy as np

#DQN agent stuff start


# Define grid dimensions
GRID_SIZE = 4


def generate_all_potential_action_map(action_map, grid_size):
    """Generates ALL_POTENTIAL_ACTION_MAP by expanding ball_pos fields."""
    all_actions = []
    actions_with_ball_pos  =0 
    actions_with_ball_idx = 0
    actions_with_two_pos =0
    actions_with_two_pos_flex = 0
    actions_with_two_pos_swap = 0
    other_actions = 0
    # Generate all possible ball positions
    all_positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    player_1_balls_positions = [(0, i) for i in range(grid_size)]
    # print(player_1_balls_positions)
    for action in action_map:
        # Expand actions that depend on ball_pos or similar fields
        if 'ball_pos' in action and action['ball_pos'] is None:
            if action['verb'] == 'MOVEANY' :
                for pos in all_positions:
                    # actions_with_ball_pos +=1
                    new_action = action.copy()
                    new_action['ball_pos'] = pos
                    all_actions.append(new_action)
            else:    
                for pos in player_1_balls_positions:
                    # actions_with_ball_pos +=1
                    new_action = action.copy()
                    new_action['ball_pos'] = pos
                    all_actions.append(new_action)
        elif 'ball_idx' in action and action['ball_idx'] is None:

            for idx in player_1_balls_positions:
                # actions_with_ball_idx +=1
                new_action = action.copy()
                new_action['ball_idx'] = idx
                all_actions.append(new_action)
        elif 'ball_pos1' in action and action['ball_pos1'] is None and 'ball_pos2' in action and action['ball_pos2'] is None:
            if 'verb'in action and action['verb'] == 'FLEXMOVE':
                for pos1, pos2 in itertools.product(player_1_balls_positions, repeat=2):
                    # actions_with_two_pos_flex +=1
                    new_action = action.copy()
                    new_action['ball_pos1'] = pos1
                    new_action['ball_pos2'] = pos2
                    all_actions.append(new_action)
            else:
                for pos1, pos2 in itertools.product(all_positions, repeat=2):
                    # actions_with_two_pos_swap +=1

                    # actions_with_two_pos +=1
                    new_action = action.copy()
                    new_action['ball_pos1'] = pos1
                    new_action['ball_pos2'] = pos2
                    all_actions.append(new_action)
        else:
            # Actions without ball_pos or similar fields
            # other_actions +=1
            all_actions.append(action)
    # total_summed_action =    actions_with_ball_pos + actions_with_ball_idx + actions_with_two_pos_flex +actions_with_two_pos_swap + other_actions 
    # print(f"ball_pos action\t\t:{actions_with_ball_pos}\naction of ball_idx\t\t:{actions_with_ball_idx}\nactions with two balls flex:swap:\t\t:{actions_with_two_pos_flex} : {actions_with_two_pos_swap}\nother actions\t\t:{other_actions}\nTotal\t\t;{total_summed_action} ")
    # print(len(all_actions))
    return all_actions


def encode_action_map(all_actions):
    """Encodes all actions into a dictionary mapping indices to actions."""
    return {idx: action for idx, action in enumerate(all_actions)}


# Define grid dimensions
GRID_SIZE = 4

# Generate ALL_POTENTIAL_ACTION_MAP
ALL_POTENTIAL_ACTION_MAP = generate_all_potential_action_map(ACTION_MAP, GRID_SIZE)
ACTION_SAPCE = len(ALL_POTENTIAL_ACTION_MAP)



#DQN agent stuff stop
class BallNotFoundError(Exception):
    """Custom exception raised when the ball is not found."""
    pass

class InvalidBallsStructureError(Exception):
    """Custom exception raised when the balls structure is invalid."""
    pass

class MissingPlayerRowError(Exception):
    """Custom exception raised when a specified player's row cannot be found."""
    pass
# Mapping for teammate based on player ID
TEAM_MATE_MAPPING = {
    0: 2,
    1: 3,
    2: 0,
    3: 1
}
ACTION_SIZE = len(ALL_POTENTIAL_ACTION_MAP)

# STATE_SIZE = 220
class Player:
    def __init__(self, game, player_id, team, policy='random'):
        self.game = game 
        self.id = player_id
        self.kill_count = 0
        self.changed_id = False
        self.plays = 0
        self.number = self.id + 1
        self.isBurned = False
        self.hand = [0] * 4
        self.rewards_given = [0,0,0,0]
        self.balls = game.Balls[player_id]
        self.team = team
        self.teamMate_id =  (self.id + 2) % 4 # 4 number of players
        self.policy = policy
        self.actions = []
        self.base = self.id * 19
        self.free_balls =  [ball for ball in self.balls if ball.pos != -10]
        self.mySafeCells = game.safeCells[self.id]
        self.hasWon = False
        self.STATE_SIZE = self.game.state_size
        if self.policy == 'ppo':
            self.agent = PPOAgent(state_size=self.STATE_SIZE, action_size=ACTION_SIZE)
            
        self.reward = 0
        self.completed_balls = [0,0,0,0]

    def get_state(self):
        """Get the game state from this player's perspective"""
        return self.game.get_STATE(self.id)
    
    def map_actions_to_indices(self,actions, all_potential_action_map, get_ball_indices):
        """
        Maps actions to their corresponding indices in ALL_POTENTIAL_ACTION_MAP.

        :param actions: List of dictionaries representing actions with `ball_pos` as an integer.
        :param all_potential_action_map: The list containing all potential actions with full `ball_pos` details.
        :param get_ball_indices: Function to convert ball positions into (row, col) indices for a 4x4 grid.
        :return: A list of indices corresponding to the mapped actions in ALL_POTENTIAL_ACTION_MAP.
        """
        indices = []

        for action in actions:
            action_id = action['action_id']
            verb = action['verb']
            offset = action.get('offset')
            offset1 = action.get('offset1')
            offset2 = action.get('offset2')
            card_value= action.get('card_value')
            
            # Handle actions with ball_pos
            if 'ball_pos' in action and action['ball_pos'] is not None:
                ball_indices = get_ball_indices(self.game.Balls, action['ball_pos'],self.id)
                if self.hasWon and  action['verb'] != 'MOVEANY':
                    ball_indices_list = list(ball_indices)
                    ball_indices_list[0] = 0
                    ball_indices = tuple(ball_indices_list)
                
                for idx, potential_action in enumerate(all_potential_action_map):
                    if (
                        # potential_action['action_id'] == action_id and
                        potential_action.get('ball_pos') == ball_indices and
                        potential_action['verb'] == verb and
                        potential_action.get('offset') == offset and 
                        potential_action.get('card_value') == card_value 

                    ):
                        indices.append(idx)
                        break
            # Handle actions with ball_idx (jailbreak)
            elif 'ball_idx' in action and action['ball_idx'] is not None:
                ball_idx = action['ball_idx'] # jailbreak indcies for the player choosen ball
                ball_indices = 0 , ball_idx
                for idx, potential_action in enumerate(all_potential_action_map):
                    if (
                        # potential_action['action_id'] == action_id and
                        potential_action['verb'] == verb and
                        potential_action.get('offset') == offset and
                        potential_action.get('ball_idx') == ball_indices and
                        potential_action.get('card_value') == card_value 
                    ):
                        indices.append(idx)
                        break
            # Handle actions with ball_pos1 and ball_pos2 (e.g., FLEXMOVE or SWAP)
            elif 'ball_pos1' in action and 'ball_pos2' in action:
                ball_indices1 = get_ball_indices(self.game.Balls,action['ball_pos1'],self.id) 
                ball_indices2 = get_ball_indices(self.game.Balls,action['ball_pos2'],self.id) 
                if self.hasWon and action['verb'] == 'FLEXMOVE':
                    ball_indices_list1 = list(ball_indices1)
                    ball_indices_list2 = list(ball_indices2)
                    ball_indices_list1[0] = 0
                    ball_indices_list2[0] = 0
                    ball_indices1 = tuple(ball_indices_list1)
                    ball_indices2 = tuple(ball_indices_list2)
                
          
                for idx, potential_action in enumerate(all_potential_action_map):
                    if (
                        # potential_action['action_id'] == action_id and
                        potential_action['verb'] == verb and
                        potential_action.get('offset1') == offset1 and
                        potential_action.get('offset2') == offset2 and
                        potential_action.get('ball_pos1') == ball_indices1 and
                        potential_action.get('ball_pos2') == ball_indices2
                    ):
                        indices.append(idx)
                        break

            # Handle actions without ball_pos
            else:
                for idx, potential_action in enumerate(all_potential_action_map):
                    if (
                        potential_action['action_id'] == action_id and
                        potential_action['verb'] == verb and
                        potential_action.get('offset') == offset
                    ):
                        indices.append(idx)
                        break
        
        # remove dublicate indices
        indices = list(set(indices))
        # if indices == []:
        #     raise ValueError(f"Mapping failed: {len(actions)} actions provided, but no indices were mapped.\nactions:{[action for action in actions]}")
        # if len(indices) != len(actions): 
            # raise ValueError(f"Mapping mismatch: {len(actions)} actions provided, but {len(indices)} indices were mapped.")
        return indices

    
    # def encode_valid_action(self, actions):
    #     """Encodes valid actions to their indices in ALL_POTENTIAL_ACTION_MAP."""
    #     # print(f'unencoded actions{actions}')
    #     encoded_indices = []

    #     for original_action in actions:
    #         # Work with a copy of the action to avoid modifying the original
    #         action = original_action.copy()

    #         # If the action has a 'ball_pos', verify it exists in ALL_POTENTIAL_ACTION_MAP
    #         if 'ball_pos' in action:
    #             try:
    #                 # Transform the action to match the ALL_POTENTIAL_ACTION_MAP format
    #                 ball_pos = action['ball_pos']  # Retain the original ball_pos
    #                 del action['ball_pos']         # Remove ball_pos temporarily to match the map structure
    #             except BallNotFoundError:
    #                 continue  # Skip invalid actions if ball position is not found

    #         # Find the corresponding index in ALL_POTENTIAL_ACTION_MAP
    #         found = False
    #         for idx, potential_action in enumerate(ALL_POTENTIAL_ACTION_MAP):
    #             # Add ball_pos back to the copied action for comparison
    #             if 'ball_pos' in potential_action:
    #                 potential_action_copy = potential_action.copy()
    #                 potential_action_copy['ball_pos'] = ball_pos
    #             else:
    #                 potential_action_copy = potential_action

    #             if potential_action_copy == action:
    #                 encoded_indices.append(idx)
    #                 found = True
    #                 break

    #         # Raise a clear error if the action is not found
    #         if not found:
    #             print(f' looking for action:{action} with ball_pos:{ball_pos}')
    #             raise ValueError(
    #                 f"Searched for: {action} with ball_pos: {ball_pos}\n"
    #                 f"Original action: {original_action}\n"
    #                 f"Did not find a match in ALL_POTENTIAL_ACTION_MAP."
    #             )

    #     # Ensure the lengths match
    #     if len(encoded_indices) != len(actions):
    #         raise ValueError(f"Encoding mismatch: {len(actions)} actions provided, but only {len(encoded_indices)} were encoded.")

    #     return encoded_indices


    def decode_action(self, action_index):
        """Decodes an action index into the corresponding action."""
        # action_index = action_tuple[0]  # Extract the first element (assumed to be the index)

        raw_action = ALL_POTENTIAL_ACTION_MAP[action_index]
        decoded_action = raw_action.copy()

        # Resolve ball positions using self.game.Balls
        if 'ball_pos' in raw_action:
            i, j = raw_action['ball_pos']
            if self.hasWon and raw_action['verb'] not in ['SWAP','MOVEANY']: i = 2
            decoded_action['ball_pos'] = self.game.Balls[i][j].pos
        if 'ball_idx' in raw_action:
            # ball_idx = raw_action['ball_idx']
            # i,j = self.id , ball_idx
            decoded_action['ball_idx'] =  raw_action['ball_idx'][1] # taking the seond item of the indices, first one hard-coded to 0
        if 'ball_pos1' in raw_action :
            i1, j1 = raw_action['ball_pos1']
            if self.hasWon and raw_action['verb'] not in ['SWAP','MOVEANY']: i1 = 2
            decoded_action['ball_pos1'] = self.game.Balls[i1][j1].pos
            i2, j2 = raw_action['ball_pos2']
            if self.hasWon and raw_action['verb'] not in ['SWAP','MOVEANY']: i2 = 2
            decoded_action['ball_pos2'] = self.game.Balls[i2][j2].pos

        return decoded_action 
    
    def agentTakeAction(self):
        """Agent takes an action based on current state."""
        state = self.game.get_STATE()
        if state is None:
            raise ValueError("get_STATE() returned None!")
        else:
            pass
            # print("DEBUG: get_STATE() returned state of shape", np.shape(state))


        valid_actions_indeices = self.map_actions_to_indices(self.actions,ALL_POTENTIAL_ACTION_MAP,self.get_ball_indices) # encode them
        raw_action = self.agent.act(state,valid_actions_indeices)
        if 0 > raw_action[0] > ACTION_SIZE:
            raise ValueError("agent selecting incorrect action/index :",raw_action)
        action_decoded ,action_tuple = self.decode_action(raw_action)
        self.agent.choosen_action = raw_action
        return action_decoded ,action_tuple








































    def check_player_hasWon(self):
        if all([ball.isDone() for ball in self.balls]):
            self.hasWon = True
            return True

    def movable_ditance_into_safeCells(self):
        MAX_INSDIE = 4
        counter = 0
        for value in self.game.safeCells[self.id]:
            if value != 0:
                return counter
            counter += 1
        return MAX_INSDIE 

    
    def __repr__(self):
        return f"Player {self.number} (Team {self.team})"

    def get_player_free_balls(self):
        self.free_balls = [ball for ball in self.balls if ball.pos != -10]
        return self.free_balls

                
    def getAction(self, action_map=ACTION_MAP):
        """Sets actions property based on player's hand."""
        # print(len(self.game.get_STATE()))
        # print(len(self.game.get_STATE()))
        self.actions = []
        for card_value in self.hand:
            card_actions = [action for action in action_map if action['card_value'] == card_value]
            self.actions += card_actions

        if self.isBurned:
            self.actions = [action for action in self.actions if action['verb'] == 'BURN_CARD']

        return self.actions if self.actions is not None else []

    def actionsSummary(self):
        """Create a concise single-line summary for each action with color formatting."""
        action_summaries = []
        for idx, action in enumerate(self.actions):
            if not action:
                continue

            # Base summary for the action
            summary = f"\033[96m{idx}\033[0m: {action.get('verb')}({action.get('card_value')})"
            
            # Handle MOVE and ball-related changes
            if 'ball_pos' in action:
                current_pos = action.get('ball_pos')
                new_pos = (current_pos + action.get('offset', 0)) % len(self.game.Board)
                summary += f" \033[93mBall@{current_pos}→{new_pos}\033[0m"
            elif 'ball_pos1' in action and 'ball_pos2' in action:
                if 'offset1' in action and 'offset2' in action:
                    offset1 = action.get('offset1', 0)
                    offset2 = action.get('offset2', 0)
                    summary += f" \033[93mBalls@{action['ball_pos1']}↔{action['ball_pos2']} Offfsets[{offset1}<->{offset2}]\033[0m"
                else:
                    summary += f" \033[93mBalls@{action['ball_pos1']}↔{action['ball_pos2']}\033[0m"


            # Handle BURN actions
            elif action['verb'] == 'BURN':
                summary += f" \033[91m→Player {(self.id + 1) % 4 + 1}\033[0m"

            # Handle FLEXMOVE with dual offsets
            elif action['verb'] == 'FLEXMOVE':
                offset1 = action.get('offset1', 0)
                offset2 = action.get('offset2', 0)
                summary += f" \033[94mOffsets({offset1}, {offset2})\033[0m"

            # Add to action summaries
            action_summaries.append(summary)
        
        return "\n".join(action_summaries) if action_summaries else "No actions"

    def humanFeedBack(self):
        """Handles human player input for choosing actions."""

        print(f"\033[95mPlayer {self.number}'s hand:\033[0m {self.hand}")
        print(f"\033[95mAvailable actions:\033[0m\n{self.actionsSummary()}")
        try:
            user_input = input("\033[92mChoose an action by index (default is 0):\033[0m ")
            chosen_action_index = int(user_input) if user_input else 0
            if 0 <= chosen_action_index < len(self.actions):
                chosen_action = self.actions[chosen_action_index]
                return chosen_action, None, None
        except ValueError:
            pass
            
        print("\033[91mInvalid action chosen. Skipping turn.\033[0m")
        return None, None, None
   
    def genrate_burn_card_actions(self, card_to_burn=None):
        """Generate BURN_CARD action for specific card."""
        if card_to_burn in self.hand:
            burn_action = {
                'card_value': card_to_burn,
                'action_id': 23 + card_to_burn,
                'verb': 'BURN_CARD'
            }
            return burn_action
        return None
    
    
    def check_legal_actions(self, game):
        playerActions = self.getAction(game.action_map)
        # Precompute reusable values
        player_base_pos = self.id * 19
        MOVE_SET_wo_MoveAny = {'MOVE', 'SUPER_MOVE', 'FLEXMOVE'}
        
        # Precompute balls that are relevant
        playersFreeBalls = [ball for ball in self.game.Balls[self.id] if ball.pos != -10]
        movable_any_balls = [ball for ball in game.free_balls if ball.pos != -10 and ball.pos != (ball.owner - 1) * 19]
        player_base_balls = [ball for ball in self.balls if ball.pos == self.base and ball.owner == self.id + 1]
        movable_any_balls_wPlayerBaseBall = movable_any_balls + player_base_balls if player_base_balls else movable_any_balls

        # Remove 'BURN_CARD' actions
        playerActions = [action for action in playerActions if action['verb'] != 'BURN_CARD']
        
        # Filter actions based on ball availability
        if len(playersFreeBalls) == 0:
            playerActions = [action for action in playerActions if action['verb'] not in MOVE_SET_wo_MoveAny]
        
        if len(movable_any_balls_wPlayerBaseBall) == 0:
            playerActions = [action for action in playerActions if action['verb'] != 'MOVEANY']
        
        if len(movable_any_balls) < 2:
            playerActions = [action for action in playerActions if action['verb'] != 'SWAP']
        
        if any(ball.pos == player_base_pos for ball in self.balls):
            playerActions = [action for action in playerActions if action['verb'] != 'JAILBREAK']

        # Handle 'BURN_CARD' if no other actions
        if self.isBurned and self.hand:
            card_to_burn = random.choice(self.hand)
            burn_action = {
                'card_value': card_to_burn,
                'action_id': 23 + card_to_burn,
                'verb': 'BURN_CARD'
            }
            self.actions = [burn_action]
            return

        self.actions = playerActions


        self.actions = playerActions
        self.actions = self.expand_actions_for_valid_balls(self.actions)
        if self.actions == []:
            burnActionForAllCards = [self.genrate_burn_card_actions(card) for card in self.hand]
            self.actions = burnActionForAllCards
    
    def isAllowed_2_enter_safeCells(self, ball_idx, ball_idy, offset):
        ball = self.game.Balls[ball_idx][ball_idy]
        if ball.pos < 0:
            return False  # Early exit if the ball's position is invalid
        
        # Cache player and ball properties to avoid redundant accesses
        player = self.game.players[ball_idx]
        max_movable_distance = player.movable_ditance_into_safeCells()
        total_distance_needed = max_movable_distance + ball.distance_to_win_gate

        # Check if the offset is within the allowed range
        isAllowed = offset > ball.distance_to_win_gate and offset <= total_distance_needed
        return isAllowed

    
    def _choose_random_action(self, actions):
        """Select a random action from available actions."""
        return random.choice(actions) if actions else None

    def _choose_prioritized_action(self, actions):
        """Select action based on priorities:
        1. Move balls in safe cells
        2. Jailbreak imprisoned balls
        3. Regular moves
        """
        # First priority: Move balls in safe cells (-4 to -1)
        safe_cell_actions = [
            action for action in actions
            if action['verb'] in ['MOVE', 'FLEXMOVE'] and
            isinstance(action.get('ball_pos'), int) and
            -4 <= action.get('ball_pos', 0) <= -1
        ]
        if safe_cell_actions:
            return random.choice(safe_cell_actions)

        # Second priority: Jailbreak actions
        jailbreak_actions = [
            action for action in actions
            if action['verb'] == 'JAILBREAK'
        ]
        if jailbreak_actions:
            return random.choice(jailbreak_actions)

        # Default: Random action
        return self._choose_random_action(actions)

    def play(self, action=None, ball_idx=None):
        """Executes an action and handles turn order.
           For a DQN/PPO policy, returns (action, log_prob, value, hidden_state) 
           so that extra variables can be used in training.
        """
        # Apply a small negative reward per play (as in your original code)
        self.reward -= 0.01 
        self.plays += 1 
        self.game.update_distances()

        if self.isBurned:
            if not self.hand:
                print('Burned')
                randomCard = random.choice(self.hand)
                burn_action = self.genrate_burn_card_actions(randomCard)
                self.actions = [burn_action]  
                self.isBurned = False  
        self.isBurned = False

        if self.actions == []:
            burnActionForAllCards = [self.genrate_burn_card_actions(card) for card in self.hand]
            self.actions = burnActionForAllCards

        # Choose action based on policy
        if not action:
            if self.policy == 'human':
                feedback = self.humanFeedBack()
                if feedback[0] is None:
                    return None  # or handle appropriately
                action = feedback[0]
                ball_idx = feedback[1]
            elif self.policy == 'random':
                action = self._choose_random_action(self.actions)
            elif self.policy == 'smart_random':
                action = self._choose_prioritized_action(self.actions)
            elif self.policy == 'ppo':
                # For a DQN/PPO policy, call agentTakeAction() which should return extra vars.
                action, action_details = self.agentTakeAction()
                action_taken, log_prob, value = action_details
            else:
                raise ValueError(f"Invalid policy: {self.policy}")

        # Execute the chosen action
        success = self._execute_action(action, ball_idx)
        if success:
            self.remove_card(action)
            # For DQN/PPO, return the extra variables for training
            # if self.policy == 'ppo':
            #     return action_details
            # else:
            return success
        else:
            raise ValueError(f"Action was not successful: Invalid action: {action}, player: {self.number}, ball_idx: {ball_idx}")

    def _execute_action(self, action, ball_idx=None):
        """Executes the action and returns success state."""
        if action == None:
            raise ValueError("No action provided. Cannot execute action.")
        verb = action.get('verb')
        
        if verb == 'JAILBREAK':
            ball_idx = action.get('ball_idx')
            return self.jailbreak(self.id, ball_idx)
            
        elif verb == 'MOVE':
            ball_idx = self.balls.index(next(ball for ball in self.balls if ball.pos == action['ball_pos']))
            return self.move_ball(self.id, ball_idx, action.get('offset', 0))
            
        elif verb == 'MOVEANY':
            return self.move_any(action)
            
        elif verb == 'SUPER_MOVE':
            ball_idx = self.balls.index(next(ball for ball in self.balls if ball.pos == action['ball_pos']))
            return self.super_move(self.id, ball_idx, action.get('offset', 0))
            
        elif verb == 'FLEXMOVE':
            ball_idx1 = self.balls.index(next(ball for ball in self.balls if ball.pos == action['ball_pos1']))
            ball_idx2 = self.balls.index(next(ball for ball in self.balls if ball.pos == action['ball_pos2']))
            return self.flex_move(action, ball_idx1, ball_idx2)
            
        elif verb == 'BURN':
            return self.burnNextPlayer()
            
        elif verb == 'BURN_CARD':
            return True
            
        elif verb == 'SWAP':
            if 'ball_pos1' in action and 'ball_pos2' in action:
                return self.swap_positions(action['ball_pos1'], action['ball_pos2'])
                
        return False

    def burnNextPlayer(self):
        """Burn the next player in the game."""
        nextplayerId = (self.id + 1) % len(self.game.players)
        next_player = self.game.players[nextplayerId]
        next_player.isBurned = True
        return True

    def flex_move(self, action, ball_idx1, ball_idx2):
        """Move two balls with offsets specified in action."""
        movable_balls = [ball for ball in self.game.Balls[self.id] if ball.pos != -10]
        # self.game.update_distances()

        # if len(movable_balls) < 2:
        #     return False
        
        # if len(movable_balls) == 1 :
        #     ball = movable_balls[0]
        #     offset = 7
        #     success = self.move_ball(self.id, ball, offset)

        #     return success
        

        if len(movable_balls) == 0:
            raise ValueError('error:filiteration player has no moveable balls yet this action passed filiteration!')
        elif len(movable_balls) == 1:
            raise ValueError('error:filiteration player has only one moveable balls yet this action passed filiteration!')
        else:
            ball_pos1 = action.get('ball_pos1')
            ball_pos2 = action.get('ball_pos2')
            offset1 = action.get('offset1', 0)
            offset2 = action.get('offset2', 0)
            ball_idx1,ball_idy1=  self.get_ball_indices(self.game.Balls,ball_pos1,self.id)
            ball_idx2,ball_idy2=  self.get_ball_indices(self.game.Balls,ball_pos2,self.id)
            ball_1 = self.game.Balls[ball_idx1][ball_idy1]
            ball_2 = self.game.Balls[ball_idx2][ball_idy2]
            # print(f'Ball 1: pos:{ball_1.pos}, ob:{ball_1.distance_to_obstacle} ,offset:{offset1}\n')
            # print(f'Ball 2: pos:{ball_2.pos}, ob:{ball_2.distance_to_obstacle} ,offset:{offset2}\n')


        if (offset1 > ball_1.distance_to_obstacle or 
            offset2 > ball_2.distance_to_obstacle):
            raise ValueError(f"\033[91mERROR:Filt:FLEXMOVE:OB Cannot move beyond the distance to the nearest obstacle" +
                             "\nball"
                             
                             ".\033[0m")
        if ball_1.pos == -10 or ball_2.pos == -10:
            raise ValueError("Cannot move a ball that is imprisoned.")
        success1 = self.move_ball(ball_idx1, ball_idy1, offset1)
        # if the ball hit the second ball, the action is invalid ,should be filtered out 
        self.game.update_distances(self.id)

        if ball_2.pos == -10:
            raise ValueError(f"FlexMove: first ball hitten the second one. Cannot move a ball that is imprisoned.\nDetails:{ball_pos1},ball2:{ball_pos2} successes:{success1} offsets:{offset1,offset2}")
        success2 = self.move_ball(ball_idx2, ball_idy2, offset2)
        if success1 and success2 != True:
            raise ValueError(f'ERROR:FLEXMOVE: the first ball moved successfully but the second ball failed to move\nball1:{ball_pos1},ball2:{ball_pos2} successes:{success1,success2} offsets:{offset1,offset2}')
            # a ball moved on top the other causing to be jailed so it can't be moved
            # if ball_1.pos == ball_2.pos:
            #     ball_1.pos = -10
        return success1 and success2

    def super_move(self, player_id, ball_idx, offset):
        ball = self.game.Balls[player_id][ball_idx]
        if ball.pos < 0:
            raise ValueError("Cannot move a ball that is imprisoned or in the safe zone.")

        self.move_ball(self.id,ball_idx,offset)
        return True

    def move_ball(self, player_id, ball_idx, offset):
        ball = self.game.Balls[player_id][ball_idx]
        frendlyMove = self.game.players[ball.owner -1].team == self.team

        # self.game.update_distances()


        # trying to move a prisoned ball
        if ball.pos == -10:
            return False
            raise ValueError("Cannot move a ball that is imprisoned.")
        # allowed_2_enter_safeCells = offset > ball.distance_to_win_gate and offset <= moveable_amount_Winthin_safeCells + ball.distance_to_win_gate
        isAllowed = self.isAllowed_2_enter_safeCells(player_id,ball_idx,offset)
        if frendlyMove:
            if isAllowed:
                howMuchTomonveinside = offset - ball.distance_to_win_gate
                ball.pos = howMuchTomonveinside * -1
                return True

            else:
                if offset > ball.distance_to_obstacle :
                    raise ValueError(f"\033[91mERROR:M_OB01 Cannot move beyond the distance to the nearest obstacle.\npos:{ball.pos}, offset:{offset},hand:{self.hand}\033[0m ")
                elif offset <= ball.distance_to_obstacle:
                    if ball.pos < 0 and ball.pos != -10:
                        for pos in range(ball.pos - 1, ball.pos + offset * -1, -1):
                            inv_pos=pos *-1
                            if inv_pos > 4 or inv_pos < 0 : raise ValueError(f'ERROR:M_SF_POS: the ball has invalid pos :{ball.pos},invPos:{inv_pos}')
                            if self.game.tinyObMap[self.id][(inv_pos)] == 1:

                                raise ValueError(f'Error: a ball inside the safe cells tryed bypassing an obsticle\npos:{ball.pos}, offset:{offset},hand:{self.hand}')
                                return False
                            
                        oldPos = ball.pos
                        newPos = ball.pos - offset  
                        indexofBall = self.game.safeCells[self.id][(newPos*-1)-1] 
                        #if hte newPos is has a ball, raise an error
                        if isinstance( self.game.safeCells[self.id][indexofBall] , Ball):
                            raise ValueError(f'ERROR:M_SF: moving on top of a ball inside the safeCells') 
                        ball.pos = newPos # thats enough if ball inside the sfaeCells

                        if ball.pos > 0:
                            raise ValueError(f'ERROR:M_SF ball in the safeCell escaped!,it was negitive pos then become positive\nINFO:\npos:{ball.pos}, \nowner:{ball.owner}, \noffset:{offset} \nOB:{ball.distance_to_obstacle}')
                        if oldPos < newPos:
                            raise ValueError('ERROR in safe cell movements , a ball tried moving backwards inside a safe cell')
                        if ball.pos not in [-1,-2,-3,-4]: raise ValueError('ERROR:M_SF: the ball got out of the safeCell')
                        return True
                    elif ball.pos >= 0:
                        newPos = (ball.pos + offset) % len(self.game.Board)
                        self.finalize_move(ball,newPos)
                        return True
                    
                # raise ValueError("The ball cannot enter its safe cells.") # the action should've been filiterd out from the start
        elif not frendlyMove: # not friendly move, allow the oppenent team can move the ball in the board
            if offset <= ball.distance_to_obstacle:
                newPos = (ball.pos + offset) % len(self.game.Board)
                self.finalize_move(ball,newPos)
                return True
            else:
                # should have been filiterd out from the start
                raise ValueError("\033[91mError: Cannot move beyond the distance to the nearest obstacle. This action should have been filtered out.\033[0m")



    def move_any(self, action):
        """Move 5 spaces using any ball from the movable balls."""
        # self.update_balls_positions()
        chosen_ball_pos = action.get('ball_pos')
        
        # Find the ball indices [player_idx, ball_idx]
        ball_indices = None
        for player_idx, player_balls in enumerate(self.game.Balls):
            for ball_idx, ball in enumerate(player_balls):
                if ball.pos == chosen_ball_pos:
                    ball_indices = [player_idx, ball_idx]
                    break
            if ball_indices:
                break
                
        if ball_indices:
            return self.move_ball(ball_indices[0], ball_indices[1], action['offset'])
                
        raise ValueError(f"No movable ball found at position {chosen_ball_pos}.")


    def jailbreak(self, player_id, ball_idx):
        """Free a specific imprisoned ball."""
        ball = self.game.Balls[self.id][ball_idx]
        playerBsee =  (ball.owner - 1) * 19


        if self.game.Board[playerBsee] != 0:
            playerID_at_newPos = self.game.Board[playerBsee]  
            for ball_in_list in self.game.Balls[playerID_at_newPos-1]:
                if ball_in_list.pos == playerBsee:
                    playerBall_to_impersonate = ball_in_list
                    playerBall_to_impersonate.pos =-10

        if ball.pos == -10:
            ball.pos = playerBsee
            self.actions = []
            self.check_legal_actions(self.game)

            return True
        return False
    
    def swap_positions(self, ball_pos1, ball_pos2):
        """Swap the positions of any two movable balls."""
        ball_1 = next(ball for sublist in self.game.Balls for ball in sublist if ball.pos == ball_pos1)
        ball_2 = next(ball for sublist in self.game.Balls for ball in sublist if ball.pos == ball_pos2)

        ball_1.pos, ball_2.pos = ball_2.pos, ball_1.pos
        return True
    

    def remove_card(self, action_or_card):
        """Remove a card from the player's hand after it has been played."""
        if isinstance(action_or_card, dict):
            card = action_or_card.get('card_value')
        else:
            card = action_or_card

        if card in self.hand:
            self.hand.remove(card)
        else:
            raise ValueError(f"Card {card} not found in player's hand. attempting to remove a card that is not in the hand")


    def expand_actions_for_valid_balls(self, filtered_actions):
        """
        Expands each action to create specific actions for each valid ball it can affect.
        Returns expanded actions list with all ball positions filled in.
        """
        expanded_actions = []
        self.game.update_distances(self.id)

        player_free_balls = [ball for ball in self.balls if ball.pos != -10]
        player_imprisoned_balls = [ball for ball in self.balls if ball.pos == -10]


        # playersFreeBalls = [ball for ball in self.balls if ball.pos != -10]
        # movable_balls = [ball for ball in self.game.get_free_balls() 
        #                 if ball.pos >= 0 and ball.pos != (ball.owner - 1) * 19]
        movable_any_balls = [ball for ball in self.game.get_free_balls() 
                             if ball.pos >= 0 and ball.pos != (ball.owner - 1) * 19]
        player_base_balls = [ball for ball in self.balls if ball.pos == self.base and ball.owner == self.id+1]
        movable_any_balls_wPlayerBaseBall = movable_any_balls + player_base_balls if player_base_balls else movable_any_balls
        # print(f' moveable balls: {len(movable_balls)}\n player base balls: {len(player_base_balls)}\n player base balls + moveable balls: {len(movable_any_balls_wPlayerBaseBall)}')

        action_groups = {
            'MOVE': [],
            'MOVEANY': [],
            'JAILBREAK': [],
            'SWAP': [],
            'SUPER_MOVE': [],
            'FLEXMOVE': [],
            'BURN': [],
            'BURN_CARD': []
        }

        for action in filtered_actions:
            verb = action['verb']
            if verb in action_groups:
                action_groups[verb].append(action)

        for verb, actions in action_groups.items():
            for action in actions:
                if verb in ['MOVE', 'SUPER_MOVE']:
                    expanded_actions.extend(
                        self._expand_single_ball_action(action, player_free_balls)
                    )
                elif verb == 'MOVEANY':
                    expanded_actions.extend(
                        self._expand_single_ball_action(action, movable_any_balls_wPlayerBaseBall)
                    )
                elif verb == 'SWAP':
                    expanded_actions.extend(
                        self._expand_two_ball_action(action, movable_any_balls)
                    )
                elif verb == 'JAILBREAK':
                    expanded_actions.extend(
                        self._expand_single_ball_action(action, player_imprisoned_balls)
                    )
                elif verb == 'FLEXMOVE':
                    expanded_actions.extend(
                        self._expand_flex_move_action(action, player_free_balls)
                    )
                else:
                    expanded_actions.append(action)

        jailbreak_actions = []
        for action in action_groups['JAILBREAK']:
            for idx, ball in enumerate(self.balls):
                if ball.pos == -10:
                    new_action = action.copy()
                    new_action['ball_idx'] = idx
                    jailbreak_actions.append(new_action)
        
        expanded_actions.extend(jailbreak_actions)

        valid_actions = self._filter_valid_moves(expanded_actions)
        
        final_actions = [action for action in valid_actions 
                        if not any(v is None for v in action.values())]

        self.actions = final_actions
        return final_actions

    def _expand_single_ball_action(self, action, ball_set):
        """Expands an action that affects a single ball."""
        expanded = []
        for ball in ball_set:
            new_action = action.copy()
            new_action['ball_pos'] = ball.pos
            expanded.append(new_action)
        return expanded

    def _expand_two_ball_action(self, action, ball_set):
        """Expands an action that affects two balls (like SWAP)."""
        expanded = []
        for i, ball1 in enumerate(ball_set):
            for ball2 in ball_set[i+1:]:
                new_action = action.copy()
                new_action['ball_pos1'] = ball1.pos
                new_action['ball_pos2'] = ball2.pos
                expanded.append(new_action)
        return expanded

    def _expand_flex_move_action(self, action, ball_set):
        """Expands FLEXMOVE actions with proper offsets and ball pairs."""
        expanded = []
        if len(ball_set) < 2:
            return expanded

        offset_pairs = [(6, 1), (5, 2), (4, 3), (3, 4), (2, 5), (1, 6)]
        
        for i, ball1 in enumerate(ball_set):
            for ball2 in ball_set[i+1:]:
                for offset1, offset2 in offset_pairs:
                    new_action = action.copy()
                    new_action['ball_pos1'] = ball1.pos
                    new_action['ball_pos2'] = ball2.pos
                    new_action['offset1'] = offset1
                    new_action['offset2'] = offset2
                    if (ball1.distance_to_obstacle > offset1 and 
                        ball2.distance_to_obstacle > offset2):
                        expanded.append(new_action)
        
        return expanded
    

    
    def _filter_valid_moves(self, actions):
        """Filters out moves that would hit obstacles."""
        valid_actions = []
    
        self.game.update_distances(self.id)
        
        # Cache to store results of isAllowed_2_enter_safeCells function
        allowed_cache = {}

        def is_allowed_to_enter(ball_idx, ball_idy, offset):
            """Check if the ball can enter safeCells, using cache to optimize repeated calls"""
            # Check if this ball and offset have been checked before
            if (ball_idx, ball_idy, offset) not in allowed_cache:
                allowed_cache[(ball_idx, ball_idy, offset)] = self.isAllowed_2_enter_safeCells(ball_idx, ball_idy, offset)
            return allowed_cache[(ball_idx, ball_idy, offset)]

        for action in actions:
            verb = action.get('verb')
            if verb in ['MOVE', 'MOVEANY', 'SUPER_MOVE']:
                ball_pos = action['ball_pos']
                if ball_pos < 0:
                    if verb == 'MOVEANY':
                        continue

                    indices = self.get_ball_indices(self.game.Balls, ball_pos, self.id)
                else:
                    indices = self.get_ball_indices(self.game.Balls, ball_pos, self.id)
                    
                if indices is None:
                    # Handle case where no match is found
                    raise ValueError(f'Ball was not found, pos:{ball_pos}')
                
                ball_idx, ball_idy = indices
                ball = self.game.Balls[ball_idx][ball_idy]
                frendlyMove = self.game.players[ball.owner - 1].team == self.team
                offset = action['offset']
                
                if offset == -4:
                    if ball.pos < 0:
                        continue
                    elif any(val == 1 for val in self.game.obsticles_map[ball.pos + offset - 1:ball.pos]):  # Check for obstacles
                        continue

                if verb == 'SUPER_MOVE' and offset > ball.distance_to_obstacle and not is_allowed_to_enter(ball_idx, ball_idy, offset):
                    pass  # Skip action if not allowed to enter

                if is_allowed_to_enter(ball_idx, ball_idy, offset):
                    if frendlyMove:
                        if verb == 'MOVE' and offset < 0:  # Prevent backward movement to safeCell
                            continue
                        if not ball.pos < 0:
                            if offset > self.movable_ditance_into_safeCells() + ball.distance_to_win_gate:  # Skip if move exceeds limit
                                continue
                        
                        if verb == 'SUPER_MOVE' and offset < ball.distance_to_obstacle:
                            if ball.distance_to_win_gate not in [12, 11, 10, 9]:
                                raise ValueError('Not playable, but passed filtration')
                        valid_actions.append(action)
                    
                    elif offset > ball.distance_to_obstacle:
                        continue
                    else:
                        valid_actions.append(action)
                    
                elif offset > ball.distance_to_obstacle:
                    continue
                else:
                    valid_actions.append(action)

                if verb == 'SUPER_MOVE' and offset > ball.distance_to_obstacle:
                    continue  # Skip invalid super move

            # Process FLEXMOVE actions
            elif verb == 'FLEXMOVE':
                ball_pos1 = action['ball_pos1']
                ball_pos2 = action['ball_pos2']
                offset1 = action['offset1']
                offset2 = action['offset2']

                # Find indices for the balls
                ball_idx1, ball_idy1 = self.get_ball_indices(self.game.Balls, ball_pos1, self.id)
                ball_idx2, ball_idy2 = self.get_ball_indices(self.game.Balls, ball_pos2, self.id)

                ball1 = self.game.Balls[ball_idx1][ball_idy1]
                ball2 = self.game.Balls[ball_idx2][ball_idy2]

                # If ball is in safeCells, restrict movement beyond safeCell limit
                if ball1.pos < 0 and ball1.pos > -4:
                    if offset1 > 3 or offset1 > (ball1.pos + 4) + 1:
                        continue

                if ball2.pos < 0 and ball2.pos > -4:
                    if offset2 > 3 or offset2 > (ball2.pos + 4) + 1:
                        continue

                # Check obstacle constraints
                if offset1 > ball1.distance_to_obstacle or offset2 > ball2.distance_to_obstacle:
                    continue  # Skip invalid moves
                # special anoying case
                if ball_pos1 == 71 and ball_pos2 == 0 and offset1 == 4 and offset2 == 3:
                    continue
                # Use cached results for isAllowed_2_enter_safeCells
                allowed1 = is_allowed_to_enter(ball_idx1, ball_idy1, offset1)
                allowed2 = is_allowed_to_enter(ball_idx2, ball_idy2, offset2)


                # # fake for testing
                # ball1.pos = 73
                # ball2.pos = 2
                # offset1 = 5
                # offset2 = 2

                board_size = len(self.game.Board)
                # Positions of the two balls
                pos1 = ball1.pos % board_size
                pos2 = ball2.pos % board_size
                # Calculate the absolute difference
                diff_abs = abs(pos1 - pos2)

                # Find the shortest difference considering the board wraps around
                # Determine the shortest difference and direction
                if diff_abs <= (board_size - diff_abs):
                    diff = diff_abs
                    is_front = True  # Direct distance is shorter
                else:
                    diff = board_size - diff_abs
                    is_front = False  # Wrapping around is shorter

                # diff = (ball1.pos)% len(self.game.Board) - (ball2.pos)% len(self.game.Board)
                if diff <= 6:
                    # if diff_abs >=10 and pos1 > 0 and pos2 > 0:
                    #     print("test")
                    if is_front:
                        if diff  ==  offset1 +1:# if the difference between the balls position is equal to the offset of the first ball,then the second ball can't be moved since its impresoned causing a crash
                            continue
                    else:
                        if diff == offset1:
                            continue
                # if allowed1 or allowed2:
                # Check for ball collisions
                if ball1.pos + offset1 == ball2.pos:
                    continue
                if ball2.pos + offset2 == ball1.pos:
                    continue

                # blocking ball_2 by ball_1
                if (ball_pos1 + offset1 ) % len(self.game.Board) == self.base:
                    # second ball stuck if it cant enter and offset reach or pass base
                    if not allowed2:
                        if (ball_pos2  + offset2 ) % len(self.game.Board) >= self.base:
                            continue
                if (ball_pos1 + offset1 ) % len(self.game.Board) == (ball_pos2 + offset2 ) % len(self.game.Board):
                    continue
                    # then if ball two going to be stuck by it
                # Validity condition: At least one ball can enter its safe cell or move without issues
                # if not allowed1 and not allowed2:
                #     continue  # Both moves invalid

                # If one ball can move but the other faces an obstacle
                if allowed1 and not allowed2:
                    if offset2 > ball2.distance_to_obstacle:
                        continue

                elif allowed2 and not allowed1:
                    if offset1 > ball1.distance_to_obstacle:
                        continue
                elif allowed1 and allowed2: # if both balls can enter ,very rare so test it in-depth
                    # dont filiter them but, make sure no issues happens
                    pass

                # If all checks pass, add the action to the valid list
                valid_actions.append(action)

            elif verb in ['SWAP', 'JAILBREAK', 'BURN', 'BURN_CARD']:
                valid_actions.append(action)
            else:
                valid_actions.append(action)
        
        return valid_actions
    
    def finalize_move(self, ball, newPos, isSuper=False):
        """
        Finalizes a move by checking for collisions, updating positions, and handling special moves.
        
        Args:
            ball (Ball): The ball object being moved.
            newPos (int): The new position to move the ball to.
            isSuper (bool): If True, the move is a 'super move' that can eliminate balls in its path.
        """
        board_length = len(self.game.Board)

        # Calculate move offset and ensure board wrapping
        offset = (newPos - ball.pos) % board_length

        # Special handling for moves of exactly 4 spaces
        if newPos - ball.pos == 4:
            newPos = (newPos + 1) % board_length

        # Handle "super move" (offset 13)
        if offset == 13:
            for step in range(1, offset + 1):
                target_idx = (ball.pos + step) % board_length
                if self.game.Board[target_idx] != 0:  # If there's a ball at this position
                    try:
                        # Get ball indices and object
                        # self.print_board_with_indices(self.game.Board)

                        ball_idx, ball_idy = self.get_ball_indices(self.game.Balls, target_idx)
                        found_ball = self.game.Balls[ball_idx][ball_idy]

                        # # Debug: Log ball and ownership
                        # print(f"Processing target_idx={target_idx}, Board[target_idx]={self.game.Board[target_idx]}")
                        # print(f"Found ball at ball_idx={ball_idx}, ball_idy={ball_idy}, ball.owner={found_ball.owner}")

                        # Validate ownership
                        if (ball_idx + 1) != self.game.Board[target_idx]:
                            raise ValueError(f"Ball ownership mismatch for ball at index {target_idx}.")

                        # Mark ball as eliminated and clear board position
                        found_ball.pos = -10
                        self.game.Board[target_idx] = 0
                    except Exception as e:
                        print(f"Error occurred while processing target_idx={target_idx}: {str(e)}")
                        raise RuntimeError(f"Error processing collision at index {target_idx}: {str(e)}")

        # For non-super moves, check and handle collisions at newPos
        else:
            if self.game.Board[newPos] != 0:  # If there's a ball at newPos
                try:
                    # Get ball indices and object
                    ball_idx, ball_idy = self.get_ball_indices(self.game.Balls, newPos)
                    found_ball = self.game.Balls[ball_idx][ball_idy]

                    if (ball_idx + 1) != self.game.Board[newPos]:
                        raise ValueError(f"Ball ownership mismatch for ball at index {newPos}.")

                    # Mark ball as eliminated and clear board position
                    found_ball.pos = -10
                    self.game.Board[newPos] = 0

                    #rewarding
                    if self.team == self.game.players[ball.owner -1].team:
                        self.reward -= 0.25
                        self.kill_count += 1
                        hitten_player_id = found_ball.owner -1
                        self.game.players[hitten_player_id].reward -= 0.1
                    else:
                        self.reward += 0.25
                        self.kill_count += 1
                        hitten_player_id = found_ball.owner -1
                        self.game.players[hitten_player_id].reward -= 0.1
                except Exception as e:
                    raise RuntimeError(f"Error processing collision at index {newPos}: {str(e)}")

        # Update the position of the moving ball
        ball.pos = newPos % board_length
        self.game.Board[ball.pos] = ball.owner  # Reflect the ball's new position on the board
        # self.game.check_duplicate_ball_positions() temprarly till getting some AI running
        return True
    
    # def print_board_with_indices(self,board):
    #     """
    #     Prints the board as 4 rows of 19 items with their indices for debugging.
    #     Args:
    #         board (numpy.ndarray): 1D numpy array representing the board.
    #     """
    #     reshaped_board = board.reshape(4, 19)  # Reshape to 4x19
    #     print("Current Board State (with indices):")
    #     print("     " + " ".join(f"{i:2}" for i in range(19)))  # Header for column indices
    #     print("    " + "-" * 57)  # Divider

    #     for row_idx, row in enumerate(reshaped_board):
    #         row_start_index = row_idx * 19
    #         row_indices = range(row_start_index, row_start_index + 19)
    #         print(f"{row_start_index:2}-{row_start_index + 18:2} | " + " ".join(f"{cell:2}" for cell in row))
    #     print()  # Extra newline for spacing


    def get_ball_indices(self, balls, target_pos, player_id=None):
        """
        Find ball indices (row, col) given a position and optionally player_id.
        Raises appropriate errors if conditions are not met or the ball isn't found.

        Handles both positive and negative target positions:
        - Positive: Search the entire 2D structure (balls).
        - Negative: Restrict the search to the row associated with player_id if provided,
        otherwise raise an error.
        """

        try:
            # Validate input types
            if not isinstance(target_pos, int):
                raise TypeError(f"'target_pos' must be an integer, got {type(target_pos)} instead.")
            if player_id is not None and not isinstance(player_id, int):
                raise TypeError(f"'player_id' must be an integer or None, got {type(player_id)} instead.")

            # Handle cases where balls is None or invalid
            if not isinstance(balls, list) or not all(isinstance(row, list) for row in balls):
                raise InvalidBallsStructureError("The 'balls' structure must be a 2D list.")
            if not balls:
                raise InvalidBallsStructureError("The 'balls' structure is empty.")

            if target_pos >= 0:
                # Search the entire 2D matrix for positive target_pos
                for row_idx, row in enumerate(balls):
                    for col_idx, ball in enumerate(row):
                        # Ensure ball has 'pos' attribute
                        if not hasattr(ball, "pos"):
                            raise AttributeError(f"Ball object at ({row_idx}, {col_idx}) has no 'pos' attribute.")
                        if ball.pos == target_pos:
                            return (row_idx, col_idx)
                raise BallNotFoundError(f"Ball with position {target_pos} not found in the entire structure.")

            else:
                # For negative target_pos
                if self.hasWon or self.game.players[self.teamMate_id].hasWon:
                    pass
                if player_id is not None:
                    # Restrict search to the specified player's row
                    if 0 <= player_id < len(balls):
                        row = balls[player_id]
                        for col_idx, ball in enumerate(row):
                            # Ensure ball has 'pos' attribute
                            if not hasattr(ball, "pos"):
                                raise AttributeError(f"Ball object at ({player_id}, {col_idx}) has no 'pos' attribute.")
                            if ball.pos == target_pos:
                                return (player_id, col_idx)
                        raise BallNotFoundError(
                            f"Ball with position {target_pos} not found in player {player_id}'s row."
                        )
                    else:
                        raise MissingPlayerRowError(
                            f"Player ID {player_id} is out of bounds for the 'balls' structure."
                        )
                else:
                    raise ValueError("Asking for negative ball position without providing player_id.")

        except (TypeError, ValueError, AttributeError, 
                InvalidBallsStructureError, MissingPlayerRowError, BallNotFoundError) as e:
            # Allow you to handle the error explicitly in your calling code
            raise e

    