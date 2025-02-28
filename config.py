import itertools

# Define grid dimensions
GRID_SIZE = 4

ACTION_MAP = [
{'card_value': 1, 'action_id': 1, 'verb': 'MOVE', 'offset': 1, 'ball_pos': None},
{'card_value': 1, 'action_id': 2, 'verb': 'MOVE', 'offset': 11, 'ball_pos': None},
{'card_value': 1, 'action_id': 3, 'verb': 'JAILBREAK', 'ball_idx': None},
{'card_value': 2, 'action_id': 4, 'verb': 'MOVE', 'offset': 2, 'ball_pos': None},
{'card_value': 3, 'action_id': 5, 'verb': 'MOVE', 'offset': 3, 'ball_pos': None},
{'card_value': 4, 'action_id': 6, 'verb': 'MOVE', 'offset': -4, 'ball_pos': None},
{'card_value': 5, 'action_id': 7, 'verb': 'MOVEANY', 'offset': 5, 'ball_pos': None},
{'card_value': 6, 'action_id': 8, 'verb': 'MOVE', 'offset': 6, 'ball_pos': None},
{'card_value': 7, 'action_id': 9, 'verb': 'MOVE', 'offset': 7, 'ball_pos': None},
{'card_value': 7, 'action_id': 10, 'verb': 'FLEXMOVE', 'offset1': 6, 'offset2': 1, 'ball_pos1': None, 'ball_pos2': None},
{'card_value': 7, 'action_id': 11, 'verb': 'FLEXMOVE', 'offset1': 5, 'offset2': 2, 'ball_pos1': None, 'ball_pos2': None},
{'card_value': 7, 'action_id': 12, 'verb': 'FLEXMOVE', 'offset1': 4, 'offset2': 3, 'ball_pos1': None, 'ball_pos2': None},
{'card_value': 7, 'action_id': 13, 'verb': 'FLEXMOVE', 'offset1': 3, 'offset2': 4, 'ball_pos1': None, 'ball_pos2': None},
{'card_value': 7, 'action_id': 14, 'verb': 'FLEXMOVE', 'offset1': 2, 'offset2': 5, 'ball_pos1': None, 'ball_pos2': None},
{'card_value': 7, 'action_id': 15, 'verb': 'FLEXMOVE', 'offset1': 1, 'offset2': 6, 'ball_pos1': None, 'ball_pos2': None},
{'card_value': 8, 'action_id': 16, 'verb': 'MOVE', 'offset': 8, 'ball_pos': None},
{'card_value': 9, 'action_id': 17, 'verb': 'MOVE', 'offset': 9, 'ball_pos': None},
{'card_value': 10, 'action_id': 18, 'verb': 'MOVE', 'offset': 10, 'ball_pos': None},
{'card_value': 10, 'action_id': 19, 'verb': 'BURN'},
{'card_value': 11, 'action_id': 20, 'verb': 'MOVE', 'ball_pos': None, 'offset': 11},
{'card_value': 12, 'action_id': 21, 'verb': 'MOVE', 'offset': 12, 'ball_pos': None},
{'card_value': 13, 'action_id': 22, 'verb': 'SUPER_MOVE', 'offset': 13, 'ball_pos': None},
{'card_value': 13, 'action_id': 23, 'verb': 'JAILBREAK', 'ball_idx': None},  # Fixed: Add JAILBREAK for King
{'card_value': 14, 'action_id': 777, 'verb': 'SWAP', 'ball_pos1': None,'ball_pos2': None},

{'card_value': 1, 'action_id': 24, 'verb': 'BURN_CARD'},
{'card_value': 2, 'action_id': 25, 'verb': 'BURN_CARD'},
{'card_value': 3, 'action_id': 26, 'verb': 'BURN_CARD'},
{'card_value': 4, 'action_id': 27, 'verb': 'BURN_CARD'},
{'card_value': 5, 'action_id': 28, 'verb': 'BURN_CARD'},
{'card_value': 6, 'action_id': 29, 'verb': 'BURN_CARD'},
{'card_value': 7, 'action_id': 30, 'verb': 'BURN_CARD'},
{'card_value': 8, 'action_id': 31, 'verb': 'BURN_CARD'},
{'card_value': 9, 'action_id': 32, 'verb': 'BURN_CARD'},
{'card_value': 10, 'action_id': 33, 'verb': 'BURN_CARD'},
{'card_value': 11, 'action_id': 34, 'verb': 'BURN_CARD'},
{'card_value': 12, 'action_id': 35, 'verb': 'BURN_CARD'},
{'card_value': 13, 'action_id': 36, 'verb': 'BURN_CARD'},
{'card_value': 14, 'action_id': 37, 'verb': 'BURN_CARD'}
]



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
                    actions_with_ball_pos +=1
                    new_action = action.copy()
                    new_action['ball_pos'] = pos
                    all_actions.append(new_action)
            else:    
                for pos in player_1_balls_positions:
                    actions_with_ball_pos +=1
                    new_action = action.copy()
                    new_action['ball_pos'] = pos
                    all_actions.append(new_action)
        elif 'ball_idx' in action and action['ball_idx'] is None:

            for idx in player_1_balls_positions:
                actions_with_ball_idx +=1
                new_action = action.copy()
                new_action['ball_idx'] = idx
                all_actions.append(new_action)
        elif 'ball_pos1' in action and action['ball_pos1'] is None and 'ball_pos2' in action and action['ball_pos2'] is None:
            if 'verb'in action and action['verb'] == 'FLEXMOVE':
                for pos1, pos2 in itertools.product(player_1_balls_positions, repeat=2):
                    actions_with_two_pos_flex +=1
                    new_action = action.copy()
                    new_action['ball_pos1'] = pos1
                    new_action['ball_pos2'] = pos2
                    all_actions.append(new_action)
            else:
                for pos1, pos2 in itertools.product(all_positions, repeat=2):
                    actions_with_two_pos_swap +=1

                    actions_with_two_pos +=1
                    new_action = action.copy()
                    new_action['ball_pos1'] = pos1
                    new_action['ball_pos2'] = pos2
                    all_actions.append(new_action)
        else:
            # Actions without ball_pos or similar fields
            other_actions +=1
            all_actions.append(action)
    total_summed_action =    actions_with_ball_pos + actions_with_ball_idx + actions_with_two_pos_flex +actions_with_two_pos_swap + other_actions 
    # print(f"ball_pos action\t\t:{actions_with_ball_pos}\naction of ball_idx\t\t:{actions_with_ball_idx}\nactions with two balls flex:swap:\t\t:{actions_with_two_pos_flex} : {actions_with_two_pos_swap}\nother actions\t\t:{other_actions}\nTotal\t\t;{total_summed_action} ")
    print("Env. Action size: ",len(all_actions))
    return all_actions

ACTION_SPACE = generate_all_potential_action_map(ACTION_MAP, GRID_SIZE)

def encode_action_map(all_actions):
    return {idx: action for idx, action in enumerate(all_actions)}

def decode_action(action_index, action_map):
    return action_map[action_index]

ENCODED_ACTION_MAP = encode_action_map(ACTION_SPACE)

# Save generated actions to a file
def save_action_map_to_file(filename, action_map):
    with open(filename, 'w') as f:
        f.write("ACTION_SPACE = [\n")
        for action in action_map.values():
            f.write(f"    {action},\n")
        f.write("]\n")

save_action_map_to_file("generated_actions.py", ENCODED_ACTION_MAP)
