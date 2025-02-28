GOOD_REWARD_BALL_ENTERED_GOAL1 = 1
GOOD_REWARD_BALL_ENTERED_GOAL2 = 1
GOOD_REWARD_BALL_ENTERED_GOAL3 = 1
GOOD_REWARD_BALL_ENTERED_GOAL_FINAL = 10

BAD_REWARD_OPPONENT_BALL_ENTERED_GOAL1 = 0.25
BAD_REWARD_OPPONENT_BALL_ENTERED_GOAL2 = 0.25
BAD_REWARD_OPPONENT_BALL_ENTERED_GOAL3 = 0.25
BAD_REWARD_OPPONENT_BALL_ENTERED_GOAL_FINAL = 2.5
class Ball:
    def __init__(self, game, pos, owner, distance_to_obstacle=-1, distance_to_win_gate=-1):
        self.game = game
        self.owner = owner
        self.owner_2 = -1 # no second owner
        self.pos = pos
        self.distance_to_obstacle = distance_to_obstacle
        self.distance_to_win_gate = distance_to_win_gate
        self.distance_to_playerWinGateOb = -1
        self.done = False
        self._last_state = None  # To store the last known state of the ball


    def update_state(self):
        """Update the current state of the ball (position, distance, etc.)."""
        self._last_state = (self.pos, self.distance_to_obstacle, self.distance_to_win_gate)
    
    def has_state_changed(self):
        """Check if the ball's state has changed since the last update."""
        if self._last_state is None:
            # If no previous state is stored, assume the ball has changed
            return True
        
        # Compare current state to the last saved state
        current_state = (self.pos, self.distance_to_obstacle, self.distance_to_win_gate)
        return current_state != self._last_state

    def reset_state(self):
        """Reset the ball's state, useful for reinitialization or after a move."""
        self._last_state = (self.pos, self.distance_to_obstacle, self.distance_to_win_gate)

    def isDone(self):
        """Returns True if the ball has completed its goal (reached win gate), otherwise False."""
        # player_number = self.game.players[self.owner - 1].number
        # safe_cells = self.game.safeCells[self.owner - 1]
        ball_owner_player = self.game.players[self.owner - 1]
        # Get the teammates and opponents correctly based on player.teamMate_id
        teammate = self.game.players[ball_owner_player.teamMate_id]  # Teammate is the one on the same team
        player_ToBe_rewarded = ball_owner_player
        opponents = [
            self.game.players[(self.owner - 1 + 1) % 4],  # First opponent
            self.game.players[(self.owner - 1 - 1) % 4]   # Second opponent
        ]
        position_to_check = {
            -4: True,
            -3: 3,
            -2: 2,
            -1: 1
        }
        if not self.pos in position_to_check:
            return False
            """  if self.done:
            return True
        if teammate.id == ball_owner_player.id and self.owner  != self.game.players[self.game.current_player].number: #(if a teammate has won and this is not his ball) the player has won and playing for his team mate, fix the rewards to be recived by the teamMate not the ball_owner
            # actualTeammate = ball_owner_player.teamMate_id
            # if the ball owner has already won, return True, we dont need to check His balls ,we only check if hes player connect AND then check only the balls thats isn't his
            # if self.game.players[self.owner -1].hasWon :
            #     return True
            # now if we checking, when player connected and the the balls arent't his becuase they the balls left to win, how to be rewarded?
            if  self.game.current_player == self.owner -1:
                ball_owner_player = self.game.players[self.game.current_player]
            elif self.game.current_player == self.game.players[self.owner-1 ].teamMate_id:
                ball_owner_player = self.game.players[self.owner-1 ].teamMate_id
                ball_owner_player = self.game.players[ball_owner_player]
            else:
                return 
            # ball_owner_player = self.game.players[ball_owner_player.teamMate_id]
            actual_teammate = self.game.players[ball_owner_player.teamMate_id]
            opponents = [
            self.game.players[(self.owner - 1 + 1) % 4],  # First opponent
            self.game.players[(self.owner - 1 - 1) % 4]   # Second opponent
        ]
            # player_tower = ball_owner_player.tower
            # Map positions to corresponding indices in safeCells
            position_to_check = {
                -4: True,
                -3: 3,
                -2: 2,
                -1: 1
            }
            # if ball_owner_player.hasWon:
            #     return True
            if self.pos in position_to_check:

                if self.pos == -4 or ball_owner_player.rewards_given[3] == 2:
                    self.done = True
                    if ball_owner_player.rewards_given[3] == 1:
                        if self.owner == 1:
                            print(f"granting reward to player {self.owner} for reaching goal 1, player current reward: {ball_owner_player.reward}")
                    # Rewarding first goal achieved
                        ball_owner_player.reward += GOOD_REWARD_BALL_ENTERED_GOAL1
                        for opponent in opponents:
                            opponent.reward -= BAD_REWARD_OPPONENT_BALL_ENTERED_GOAL1
                        ball_owner_player.rewards_given[3] = 2 
                        return True


                    if ball_owner_player.rewards_given[3] == 2:
                        if self.pos == -3: # give reward for reaching 3rd goal
                            self.done = True
                            if ball_owner_player.rewards_given[2] == 1:
                                if self.owner == 1:
                                    print(f"granting reward to player {self.owner} for reaching goal 2, player current reward: {ball_owner_player.reward}")
                                
                            # Rewarding: second goal achieved
                                ball_owner_player.reward += GOOD_REWARD_BALL_ENTERED_GOAL2
                                for opponent in opponents:
                                    opponent.reward -= BAD_REWARD_OPPONENT_BALL_ENTERED_GOAL2
                                ball_owner_player.rewards_given[2] = 2 
                                return True

                        if ball_owner_player.rewards_given[2] == 2:# check if 3rd goal has been achived
                            if self.pos == -2:
                                self.done = True
                                if ball_owner_player.rewards_given[1] == 1:
                                    if self.owner == 1:
                                        print(f"granting reward to player {self.owner} for reaching goal 3, player current reward: {ball_owner_player.reward}")
                                # Rewarding third goal achieved
                                    ball_owner_player.reward += GOOD_REWARD_BALL_ENTERED_GOAL3
                                    for opponent in opponents:
                                        opponent.reward -= BAD_REWARD_OPPONENT_BALL_ENTERED_GOAL3
                                    ball_owner_player.rewards_given[1] = 2 
                                    return True
                            if ball_owner_player.rewards_given[1] == 2:
                                if self.pos == -1:
                                    self.done = True
                                    # Rewarding: final goal achieved !
                                    if ball_owner_player.rewards_given[0] == 1:
                                        if self.owner == 1:
                                            print(f"granting reward to player {self.owner} for reaching final goal, player current reward: {ball_owner_player.reward}")
                                        ball_owner_player.reward += GOOD_REWARD_BALL_ENTERED_GOAL_FINAL
                                        for opponent in opponents:
                                            opponent.reward -= BAD_REWARD_OPPONENT_BALL_ENTERED_GOAL_FINAL
                                        ball_owner_player.rewards_given[0] = 2
                                        actual_teammate.hasWon = True 
                                        return True
        opponents = [
        self.game.players[(self.owner - 1 + 1) % 4],  # First opponent
        self.game.players[(self.owner - 1 - 1) % 4]   # Second opponent
    ]
        # player_tower = ball_owner_player.tower
        # Map positions to corresponding indices in safeCells"""

        # if ball_owner_player.hasWon:
        #     return True

        if ball_owner_player.hasWon or self.game.players[ball_owner_player.teamMate_id].hasWon:
            # the player and his teamMate will be connecected
            if self.owner_2 != -1:
                ball_owner_player_2 = self.game.players[self.owner_2 -1] 
                if self.game.current_player == ball_owner_player_2 or self.game.current_player ==ball_owner_player:
                    if self.game.current_player == ball_owner_player_2:
                        player_ToBe_rewarded =  ball_owner_player_2
                    else:
                        player_ToBe_rewarded =  ball_owner_player

        if self.pos in position_to_check:

            if self.pos == -4 or ball_owner_player.rewards_given[3] == 1:
                self.done = True
                if ball_owner_player.rewards_given[3] == 0:
                    # if self.owner == 1:
                        # print(f"granting reward to player {self.owner} for reaching goal 1, player current reward: {ball_owner_player.reward}")
                # Rewarding first goal achieved
                    player_ToBe_rewarded.reward += GOOD_REWARD_BALL_ENTERED_GOAL1
                    for opponent in opponents:
                        opponent.reward -= BAD_REWARD_OPPONENT_BALL_ENTERED_GOAL1
                    ball_owner_player.rewards_given[3] = 1 
                    return True


                if ball_owner_player.rewards_given[3] == 1:
                    if self.pos == -3: # give reward for reaching 3rd goal
                        self.done = True
                        if ball_owner_player.rewards_given[2] == 0:
                            # if self.owner == 1:
                                # print(f"granting reward to player {self.owner} for reaching goal 2, player current reward: {ball_owner_player.reward}")
                            
                        # Rewarding: second goal achieved
                            player_ToBe_rewarded.reward += GOOD_REWARD_BALL_ENTERED_GOAL2
                            for opponent in opponents:
                                opponent.reward -= BAD_REWARD_OPPONENT_BALL_ENTERED_GOAL2
                            ball_owner_player.rewards_given[2] = 1 
                            return True

                    if ball_owner_player.rewards_given[2] == 1:# check if 3rd goal has been achived
                        if self.pos == -2:
                            self.done = True
                            if ball_owner_player.rewards_given[1] == 0:
                                # if self.owner == 1:
                                    # print(f"granting reward to player {self.owner} for reaching goal 3, player current reward: {ball_owner_player.reward}")
                            # Rewarding third goal achieved
                                player_ToBe_rewarded.reward += GOOD_REWARD_BALL_ENTERED_GOAL3
                                for opponent in opponents:
                                    opponent.reward -= BAD_REWARD_OPPONENT_BALL_ENTERED_GOAL3
                                ball_owner_player.rewards_given[1] = 1 
                                return True
                        if ball_owner_player.rewards_given[1] == 1:
                            if self.pos == -1:
                                self.done = True
                                # Rewarding: final goal achieved !
                                if ball_owner_player.rewards_given[0] == 0:
                                    # if self.owner == 1:
                                        # print(f"granting reward to player {self.owner} for reaching final goal, player current reward: {ball_owner_player.reward}")
                                    player_ToBe_rewarded.reward += GOOD_REWARD_BALL_ENTERED_GOAL_FINAL
                                    for opponent in opponents:
                                        opponent.reward -= BAD_REWARD_OPPONENT_BALL_ENTERED_GOAL_FINAL
                                    ball_owner_player.rewards_given[0] = 1
                                    ball_owner_player.hasWon = True 
                                    return True
                
        
                    



































        # # Check if the position is in the mapping and the player number matches
        # if self.pos in position_to_check:
        #     if self.pos == -4 or self.done:
        #         self.done = True
        #         if self.rewards_given[3] == 0:
        #                 # Rewarding: Final goal achieved
        #                 ball_owner_player.reward += 1
        #                 for opponent in opponents:
        #                     opponent.reward -= 1
        #                 self.rewards_given[3] = 1
        #                 return True
        #         elif safe_cells[position_to_check[self.pos]] == player_number:
        #             self.done = True
                    
        #             if self.pos == -2 and self.rewards_given[1] == 0 or self.pos == -3 and self.rewards_given[2] == 0:
        #             # Rewarding: Final goal achieved
        #                 ball_owner_player.reward += 1
        #                 for opponent in opponents:
        #                     opponent.reward -= 1
        #                 if self.pos == -2:
        #                     self.rewards_given[1] = 1
        #                 else:
        #                     self.rewards_given[2] = 1
                
                
        #         if self.pos == -1:
        #             self.game.players[self.owner - 1].hasWon = True
        #             if self.rewards_given[0] == 0:
        #                 # Rewarding: Final goal achieved
        #                 ball_owner_player.reward += 10
        #                 for opponent in opponents:
        #                     opponent.reward -= 10
        #                 self.rewards_given[0] = 1
        #         return True
        
        # return False

    def update_till_obstacle(self):
        """Update the distance to the next obstacle."""
        self.distance_to_playerWinGateOb = self.distance_to_win_gate + 1
        if self.pos < 0:
            if self.pos != -10:
                if self.isDone():
                    self.distance_to_obstacle = 0
                    return True

                tiny_ob_map = self.game.tinyObMap
                pos = self.pos
                if pos == -4: 
                    self.distance_to_obstacle = 0
                elif pos == -3:
                    if tiny_ob_map[self.owner - 1][3] == 1:
                        self.distance_to_obstacle = 0
                    else:
                        self.distance_to_obstacle = 1
                elif pos == -2:
                    if tiny_ob_map[self.owner - 1][2] == 1:
                        self.distance_to_obstacle = 0
                    elif tiny_ob_map[self.owner - 1][3] == 1:
                        self.distance_to_obstacle = 1
                    else:
                        self.distance_to_obstacle = 2
                elif pos == -1:
                    if tiny_ob_map[self.owner - 1][1] == 1:
                        self.distance_to_obstacle = 0
                    elif tiny_ob_map[self.owner - 1][2] == 1:
                        self.distance_to_obstacle = 1
                    elif tiny_ob_map[self.owner - 1][3] == 1:
                        self.distance_to_obstacle = 2
                    else:
                        self.distance_to_obstacle = 3

                if self.distance_to_obstacle not in range(4):
                    raise ValueError(f'ERROR: Distance to obstacle for position {self.pos} is {self.distance_to_obstacle}, not set properly!')
                return True

        # Check obstacles along the board
        board = self.game.Board
        for dist in range(1, 14):
            check_pos = (self.pos + dist) % len(board)
            if self.game.obsticles_map[check_pos] == 1 :
                self.distance_to_obstacle = dist - 1
                return
            elif self.game.obsticles_map[check_pos] == -1:
                self.distance_to_obstacle = dist - 1
                return
            
        self.distance_to_obstacle = 14  # No obstacle within the distance range

    def update_till_winGate(self):
        """Update the distance to the win gate."""
        if self.pos < 0:
            self.distance_to_win_gate = -1
            return

        board = self.game.Board
        win_gate = self.game.win_gates[self.owner - 1]
        if self.pos <= win_gate:
            self.distance_to_win_gate = win_gate - self.pos
        else:
            self.distance_to_win_gate = (len(board) - self.pos) + win_gate
