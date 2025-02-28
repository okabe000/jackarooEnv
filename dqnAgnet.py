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
import threading
import keyboard
import json 


import json
import glob
import os
import torch
import time
from datetime import datetime, timedelta
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from colorama import Fore, Style
import threading
import keyboard
# ================================
# PPO Network: Shared Base with Two Heads
# ================================
class PPOModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512, rnn_layers=1):
        super(PPOModel, self).__init__()
        # Base network layers (BatchNorm removed)
        self.fc1 = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        # RNN layer remains unchanged
        self.rnn = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True
        )
        # Remove BatchNorm from subsequent layers
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, action_size)
        self.critic = nn.Linear(256, 1)
    
    def forward(self, x, hidden_state=None):
        # Remainder of the forward pass unchanged
        x = self.fc1(x)
        x = self.fc2(x)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x, hidden_state = self.rnn(x, hidden_state)
        x = x[:, -1, :]
        x = self.fc3(x)
        x = self.fc4(x)
        logits = self.actor(x)
        value = self.critic(x).squeeze(1)
        return logits, value, hidden_state
# ================================
# PPO Agent
# ================================
class PPOAgent:
    def __init__(self, state_size, action_size, device='cuda', lr=3e-4, gamma=0.99,
                 clip_epsilon=0.2, update_epochs=4, batch_size=64, gae_lambda=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.choosen_action = -1
        self.hidden_state = None
        self.model = PPOModel(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Memory for a rollout
        self.memory = []

    def reset_hidden_state(self):
        self.hidden_state = None
        
    def act(self, state, valid_actions=None, hidden_state=None):
        if state is None:
            print("DEBUG: In act(), received state with type:", type(state), "and shape:", np.shape(state) if hasattr(state, 'shape') else "N/A")
            raise ValueError("Received a None state in act() – please check your environment's get_STATE() method!")
        # For debugging: print a summary of the state
        
        self.model.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, value, new_hidden_state = self.model(state_tensor, self.hidden_state)
        logits = logits.squeeze(0)
        if valid_actions is not None:
            mask = torch.ones(self.action_size, device=self.device) * float('-inf')
            mask[valid_actions] = 0.0
            logits = logits + mask
        probs = F.softmax(logits, dim=-1)
        self.hidden_state = new_hidden_state.detach()  # Update for next step
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if 0 > action.item() > self.action_size:
            raise ValueError('choosing out of the action size')
        return action.item(), log_prob.detach(), value.detach()


    def store_transition(self, transition):
        """
        Store a transition in memory.
        Each transition is a tuple:
        (state, action, log_prob, reward, done, value)
        """
        self.memory.append(transition)

    def clear_memory(self):
        self.memory = []

    def compute_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation (GAE)
        Args:
            rewards: list of rewards for the rollout
            values: list of value estimates from the critic
            dones: list of done flags (True/False) for the rollout
        Returns:
            advantages: computed advantages for each time step
            returns: discounted returns (advantages + values)
        """
        advantages = []
        gae = 0
        # Convert lists to tensors if needed
        values = values + [0]  # Append a zero for the last value
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

        
    def update(self):
        """
        Update the policy and value network using PPO's clipped objective.
        Returns:
            tuple: (total_loss, entropy) for tracking training progress
        """
        if len(self.memory) == 0:
            print("ERROR: Memory is empty. Cannot update model.")
            return 0.0, 0.0  # Return default values instead of raising error
            
        # Convert memory to separate lists
        states = torch.FloatTensor([t[0] for t in self.memory]).to(self.device)
        actions = torch.LongTensor([t[1] for t in self.memory]).to(self.device)
        old_log_probs = torch.stack([t[2] for t in self.memory]).to(self.device)
        rewards = [t[3] for t in self.memory]
        dones = [t[4] for t in self.memory]
        values = [t[5].item() for t in self.memory]
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Track losses and entropy for reporting
        total_loss_avg = 0
        entropy_avg = 0
        
        # PPO update: run several epochs on the rollout batch
        dataset_size = states.size(0)
        for epoch in range(self.update_epochs):
            # Shuffle the indices for mini-batches
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                mini_batch_indices = indices[start:end]
                
                batch_states = states[mini_batch_indices]
                batch_actions = actions[mini_batch_indices]
                batch_old_log_probs = old_log_probs[mini_batch_indices]
                batch_returns = returns[mini_batch_indices]
                batch_advantages = advantages[mini_batch_indices]
                
                logits, values_pred, _ = self.model(batch_states)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                # New log probabilities
                new_log_probs = dist.log_prob(batch_actions)
                # Ratio for PPO
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                # PPO surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                # Critic loss (MSE)
                critic_loss = F.mse_loss(values_pred, batch_returns)
                # Calculate entropy
                entropy = dist.entropy().mean()
                # Total loss with entropy bonus
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Update averages
                total_loss_avg += total_loss.item()
                entropy_avg += entropy.item()
        
        # Calculate averages over all updates
        num_updates = self.update_epochs * ((dataset_size + self.batch_size - 1) // self.batch_size)
        total_loss_avg /= num_updates
        entropy_avg /= num_updates
        
        self.clear_memory()
        return total_loss_avg, entropy_avg


    @staticmethod
    def train_ppo_agent(Game, num_episodes=1000, rollout_length=2048, log_to_tensorboard=False, resume_training=False):
        """
        Enhanced PPO training loop with save/load capability and comprehensive analytics.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        unique_id = f"instance_{os.getpid()}_{timestamp}"
        log_dir = f'runs/ppo_agent_{unique_id}' if log_to_tensorboard else None
        writer = SummaryWriter(log_dir) if log_to_tensorboard else None

        # Initialize game and agent
        game = Game()
        state_size = len(game.get_STATE())
        action_size = len(game.action_size)
        print(f'Action size\t{action_size}:\nState size:\t{state_size}\ntraining started:')
        agent = game.players[0].agent
        agent.__class__ = PPOAgent
        agent.__init__(state_size, action_size, device=device)

        # Start interactive adjustment thread
        adjustments_thread = threading.Thread(target=interactive_adjustments, daemon=True)
        adjustments_thread.start()

        # Load previous training state if resuming
        if resume_training:
            print('resuming training...')
            start_episode, team_a_wins, team_b_wins, saved_elapsed_time = load_training_state(agent)
            load_latest_model(agent)
            training_start_time = time.time() - saved_elapsed_time
        else:
            start_episode = 1
            team_a_wins = 0
            team_b_wins = 0
            training_start_time = time.time()

        # Analytics tracking
        moving_avg_rewards = deque(maxlen=100)
        moving_avg_losses = deque(maxlen=100)
        recent_rewards = deque(maxlen=100)
        recent_losses = deque(maxlen=100)

        for episode in range(start_episode, start_episode + num_episodes):
            # Apply interactive parameter updates
            if interactive_params["learning_rate"] is not None:
                agent.optimizer.param_groups[0]['lr'] = interactive_params["learning_rate"]
                interactive_params["learning_rate"] = None
            if interactive_params["clip_range"] is not None:
                agent.clip_range = interactive_params["clip_range"]
                interactive_params["clip_range"] = None
            if interactive_params["batch_size"] is not None:
                agent.batch_size = interactive_params["batch_size"]
                interactive_params["batch_size"] = None

            total_steps = 0
            state = game.reset()
            agent.reset_hidden_state()
            episode_reward = 0
            episode_loss = 0
            entropy = 0

            game.deal_cards(game.players)
            done = False

            while not done and len(agent.memory) < rollout_length:
                cards_this_round = 5 if game.round % 3 == 1 else 4
                for _ in range(cards_this_round):
                    if game.done:
                        break
                    next_state, reward, done, agent_outputs = game.step(show_game=False)
                    # print(f"Faliare in Step function:contining to next epsiode")
                    episode_reward += reward
                    total_steps += 1

                    if done:
                        if game.team_has_won == 1:
                            team_a_wins += 1
                        elif game.team_has_won == 2:
                            team_b_wins += 1

                    action, log_prob, value = agent_outputs[0]
                    agent.store_transition((state, action, log_prob, reward, done, value))
                    state = next_state

                if not done:
                    game.deal_cards(game.players)

            # Update policy and track loss
            if len(agent.memory) >= agent.batch_size:
                print('agent mem:',len(agent.memory))
                episode_loss, entropy = agent.update()
                # print(f"Error during update: {e}")
                episode_loss, entropy = 0.0, 0.0
            else:
                episode_loss, entropy = 0.0, 0.0

            recent_losses.append(episode_loss)

            # Save state every 100 episodes
            if episode % 100 == 0:
                elapsed_time = time.time() - training_start_time
                save_training_state(agent, episode, team_a_wins, team_b_wins, elapsed_time)

            # Update analytics
            recent_rewards.append(episode_reward)
            recent_losses.append(episode_loss)

            # Calculate moving averages after each episode
            moving_avg_reward = sum(recent_rewards) / len(recent_rewards)
            moving_avg_loss = sum(recent_losses) / len(recent_losses)

            # Append the moving averages to their respective deques
            moving_avg_rewards.append(moving_avg_reward)
            moving_avg_losses.append(moving_avg_loss)


            # Calculate statistics
            total_games = episode - start_episode + 1
            win_rate_a = (team_a_wins / total_games) * 100 if total_games > 0 else 0
            win_rate_b = (team_b_wins / total_games) * 100 if total_games > 0 else 0

            # Log to tensorboard and game history
            if log_to_tensorboard:
                log_episode_data_to_console(episode, num_episodes, episode_loss, win_rate_a, win_rate_b,
                            team_a_wins, team_b_wins, episode_reward, entropy, total_steps)
                writer.add_scalar("Loss", episode_loss, episode)
                writer.add_scalar("Moving avg. Loss", episode_loss, episode)
                writer.add_scalar("Reward", episode_reward, episode)
                writer.add_scalar("Moving avg. Rewards", moving_avg_reward, episode)
                writer.add_scalar("Win Rate Team A", win_rate_a, episode)
                writer.add_scalar("Steps", game.round, episode)
                writer.add_scalar("Entropy", entropy, episode)

            # Calculate elapsed time
            elapsed_time = time.time() - training_start_time
            elapsed_time_str = str(timedelta(seconds=int(elapsed_time)))

            # Console output
            print(
                f"{Fore.CYAN}Episode {episode:4d}/{start_episode + num_episodes  - 1}{Style.RESET_ALL} "
                f"{Fore.YELLOW}Loss: {episode_loss:8.4f}{Style.RESET_ALL} "
                f"{Fore.GREEN}Reward: {episode_reward:8.2f}{Style.RESET_ALL} "
                f"{Fore.MAGENTA}Entropy: {entropy:6.4f}{Style.RESET_ALL} "
                f"{Fore.LIGHTGREEN_EX}Win Rate A: {win_rate_a:.6f}%{Style.RESET_ALL} "
                f"{Fore.LIGHTRED_EX}Win Rate B: {win_rate_b:.6f}%{Style.RESET_ALL} "
                f"{Fore.BLUE}Elapsed: {elapsed_time_str}{Style.RESET_ALL}"
            )

        # Save final model
        torch.save(agent.model.state_dict(), 'ppo_agent_model_final.pth')
        if log_to_tensorboard:
            writer.close()

        return agent

# ================================
# Example Game Environment Stub
# ================================
# The following stub is only provided as an example. You should replace it
# with your own game/environment code.
# class DummyGame:
#     def __init__(self):
#         # Example: state is a vector of size 10; there are 4 possible actions.
#         self.state_size = 10
#         self.action_size = [0, 1, 2, 3]
#         self.current_step = 0
#         self.max_steps = 50
#         self.players = [type("DummyPlayer", (), {})()]  # Create a dummy player

#         # Create an agent instance for the dummy player.
#         self.players[0].agent = PPOAgent(state_size=self.state_size, action_size=len(self.action_size))

#     def reset(self):
#         self.current_step = 0
#         # Return an initial state
#         return np.random.randn(self.state_size)

#     def get_STATE(self):
#         # Return current state representation
#         return np.random.randn(self.state_size)

#     def get_valid_actions(self):
#         # In this dummy example, all actions are always valid.
#         return self.action_size

#     def step(self, action):
#         # For this dummy game, we randomly generate next state, reward, and done flag.
#         self.current_step += 1
#         next_state = np.random.randn(self.state_size)
#         reward = random.random()
#         done = self.current_step >= self.max_steps
#         return next_state, reward, done

# ================================
# To Run the PPO Training:
# ================================
# if __name__ == '__main__':
#     # Replace DummyGame with your actual game/environment.
#     PPOAgent.train_ppo_agent(DummyGame, num_episodes=1000, rollout_length=2048, log_to_tensorboard=True)



# Shared dictionary for interactive parameter updates
interactive_params = {
    "learning_rate": None,
    "clip_range": None,
    "batch_size": None
}

def save_training_state(agent, episode, team_a_wins, team_b_wins, elapsed_time):
    state = {
        "last_episode": episode,
        "elapsed_time": elapsed_time,
        "team_a_wins": team_a_wins,
        "team_b_wins": team_b_wins
    }
    with open('training_state.json', 'w') as f:
        json.dump(state, f)
    print(f"Training state saved at episode {episode}.")
    save_model(agent, episode)

def save_model(agent, episode, directory="models"):
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, f"ppo_model_episode_{episode}.pth")
    torch.save(agent.model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

def load_training_state(agent):
    try:
        with open('training_state.json', 'r') as f:
            state = json.load(f)
        episode = state.get('last_episode', 1)
        elapsed_time = state.get('elapsed_time', 0)
        team_a_wins = state.get('team_a_wins', 0)
        team_b_wins = state.get('team_b_wins', 0)
        return episode, team_a_wins, team_b_wins, elapsed_time
    except FileNotFoundError:
        return 1, 0, 0, 0

def load_latest_model(agent):
    models = glob.glob('models/ppo_model_episode_*.pth')
    if models:
        latest_model = max(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        agent.model.load_state_dict(torch.load(latest_model))
        print(f"Loaded model from {latest_model}")
    else:
        print("No model found to load.")


def log_episode_data_to_console( episode, total_episodes, loss, win_rate_a, win_rate_b, 
                 team_a_wins, team_b_wins, total_reward, entropy, steps):
    with open('gamesHistory.log', 'a') as f:
        f.write(f"\033[94m[Episode {episode:04}/{total_episodes}]\033[0m")
        f.write(f" -- Loss: {loss:.6f}")
        f.write(f" -- Reward: {total_reward}")
        f.write(f" -- Entropy: {entropy:.2f}")
        f.write(f" -- Teams Winning Rates: {win_rate_a:.2f}% : {win_rate_b:.2f}%")
        f.write(f" -- Steps: {steps}")
        f.write(f" -- Teams Wins: {team_a_wins} : {team_b_wins}")
        f.write("\n")

def interactive_adjustments():
    while True:
        keyboard.wait('ctrl+g')
        print("\n--- Modify PPO Hyperparameters ---")
        print("1: Update Learning Rate")
        print("2: Update Clip Range")
        print("3: Update Batch Size")
        print("4: Exit Interactive Mode")
        try:
            option = int(input("Choose an option: ").strip())
            if option == 1:
                new_value = float(input("Enter new learning rate (e.g., 0.0003): ").strip())
                interactive_params["learning_rate"] = new_value
                print(f"Updating learning rate to {new_value}")
            elif option == 2:
                new_value = float(input("Enter new clip range (e.g., 0.2): ").strip())
                interactive_params["clip_range"] = new_value
                print(f"Updating clip range to {new_value}")
            elif option == 3:
                new_value = int(input("Enter new batch size: ").strip())
                interactive_params["batch_size"] = new_value
                print(f"Updating batch size to {new_value}")
            elif option == 4:
                print("Exiting interactive mode.")
                break
            else:
                print("Invalid option. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")



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
