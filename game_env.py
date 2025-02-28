import gym  # For sb3-contrib compatibility
import gymnasium  # For the environment
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from custom_game_env import CustomGameEnv, get_action_mask
from tqdm import tqdm
import os
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from colorama import init, Fore, Style
import psutil  # For memory monitoring

# Initialize colorama for colored output
init()

# SafeEnv wrapper to catch exceptions and prevent crashes
class SafeEnv(gymnasium.Wrapper):  # Use gymnasium.Wrapper instead of gym.Wrapper
    def __init__(self, env):
        super().__init__(env)
        self._last_obs = None

    def step(self, action):
        try:
            self._last_obs, reward, done, truncated, info = self.env.step(action)
            return self._last_obs, reward, done, truncated, info
        except Exception as e:
            print(f"{Fore.YELLOW}Exception in step (likely invalid action): {e}{Style.RESET_ALL}")
            self._last_obs = self.reset()
            return self._last_obs, -1, True, False, {'invalid_action': True}

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class EnhancedProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, num_envs, verbose=0, log_interval=1000000, window_size=100, eta_window=1000):
        super(EnhancedProgressBarCallback, self).__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.num_envs = num_envs
        self.start_time = None
        self.last_update_time = None
        self.last_step_count = 0
        self.log_interval = log_interval
        self.rewards_buffer = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.team_a_wins = deque(maxlen=window_size)
        self.team_b_wins = deque(maxlen=window_size)
        self.round_counts = deque(maxlen=window_size)
        self.invalid_actions = deque(maxlen=window_size)
        self.global_step_count = 0
        self.episode_count = 0
        self.step_times = deque(maxlen=eta_window)
        self.total_invalid_actions = 0

    def on_training_start(self, locals=None, globals=None):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")
        self.start_time = time.time()
        self.last_update_time = self.start_time

    def _on_step(self):
        try:
            self.global_step_count += self.num_envs
            steps_to_update = self.global_step_count - self.pbar.n
            if steps_to_update > 0:
                self.pbar.update(steps_to_update)
            
            current_time = time.time()
            self.step_times.append(current_time - self.last_update_time)
            self.last_update_time = current_time
            
            infos = self.locals.get('infos', [{}])
            
            for info in infos:
                if 'invalid_action' in info and info['invalid_action']:
                    self.total_invalid_actions += 1
                    self.invalid_actions.append(1)
                else:
                    self.invalid_actions.append(0)
                
                if 'done' in info and info['done']:
                    self.episode_count += 1
                    self.rewards_buffer.append(info.get('reward', 0))
                    self.episode_lengths.append(info.get('step_count', 0))
                    team_won = info.get('team_has_won', 0)
                    if team_won == 1:
                        self.team_a_wins.append(1)
                        self.team_b_wins.append(0)
                    elif team_won == 2:
                        self.team_a_wins.append(0)
                        self.team_b_wins.append(1)
                    else:
                        self.team_a_wins.append(0)
                        self.team_b_wins.append(0)
                    if 'rounds_completed' in info:
                        self.round_counts.append(info['rounds_completed'])

            avg_time_per_step = np.mean(self.step_times) if self.step_times else (current_time - self.start_time) / self.global_step_count
            remaining_steps = self.total_timesteps - self.global_step_count
            eta_seconds = avg_time_per_step * remaining_steps
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            
            team_a_win_rate = np.mean(self.team_a_wins) * 100 if len(self.team_a_wins) > 0 else 0
            team_b_win_rate = np.mean(self.team_b_wins) * 100 if len(self.team_b_wins) > 0 else 0
            steps_per_sec = self.global_step_count / (current_time - self.start_time) if (current_time - self.start_time) > 0 else 0
            
            postfix_dict = {
                "steps/s": f"{steps_per_sec:.2f}",
                "win_rate": f"{team_a_win_rate:.1f}% (A) : {team_b_win_rate:.1f}% (B)",
                "episodes": self.episode_count,
                "invalid": self.total_invalid_actions,
                "ETA": eta_str
            }
            self.pbar.set_postfix(postfix_dict)
            
            if self.num_timesteps % self.log_interval == 0:
                steps_since_last = self.num_timesteps - self.last_step_count
                steps_per_sec_interval = steps_since_last / (current_time - self.last_update_time) if (current_time - self.last_update_time) > 0 else 0
                updates = self.num_timesteps // self.model.n_steps
                
                self.logger.record("performance/steps_per_second", steps_per_sec_interval)
                if len(self.rewards_buffer) > 0:
                    self.logger.record("metrics/mean_reward", np.mean(self.rewards_buffer))
                    self.logger.record("metrics/mean_episode_length", np.mean(self.episode_lengths))
                if len(self.team_a_wins) > 0:
                    self.logger.record("metrics/team_a_win_rate", team_a_win_rate)
                    self.logger.record("metrics/team_b_win_rate", team_b_win_rate)
                if len(self.round_counts) > 0:
                    self.logger.record("metrics/avg_rounds_per_game", np.mean(self.round_counts))
                if len(self.invalid_actions) > 0:
                    self.logger.record("metrics/invalid_action_rate", np.mean(self.invalid_actions))
                
                self.last_step_count = self.num_timesteps
            
            process = psutil.Process()
            mem_usage = process.memory_info().rss / 1024**2  # MB
            if mem_usage > 4000:
                print(f"{Fore.YELLOW}Warning: High memory usage ({mem_usage:.2f} MB). Consider reducing num_envs.{Style.RESET_ALL}")
            
            return True
        except Exception as e:
            print(f"{Fore.RED}Error in training step: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return False

    def _on_rollout_end(self):
        current_time = time.time()
        update_time = current_time - self.last_update_time
        self.logger.record("performance/time_per_update", update_time)
        self.last_update_time = current_time

    def _on_training_end(self):
        total_time = time.time() - self.start_time
        self.logger.record("performance/total_training_time", total_time)
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        avg_steps_per_sec = self.total_timesteps / total_time if total_time > 0 else 0
        self.pbar.set_postfix({"avg_steps/s": f"{avg_steps_per_sec:.2f}"})
        self.pbar.close()
        
        print(f"{Fore.CYAN}===== Training Completed ====={Style.RESET_ALL}")
        print(f"{Fore.GREEN}Total training time:{Style.RESET_ALL} {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"{Fore.GREEN}Average steps per second:{Style.RESET_ALL} {avg_steps_per_sec:.2f}")
        if len(self.team_a_wins) > 0:
            team_a_win_rate = np.mean(self.team_a_wins) * 100
            team_b_win_rate = np.mean(self.team_b_wins) * 100
            print(f"{Fore.YELLOW}Final win rate:{Style.RESET_ALL} {team_a_win_rate:.2f}% (Team A) : {team_b_win_rate:.2f}% (Team B)")
        if len(self.rewards_buffer) > 0:
            print(f"{Fore.YELLOW}Average reward:{Style.RESET_ALL} {np.mean(self.rewards_buffer):.2f}")
            print(f"{Fore.YELLOW}Average episode length:{Style.RESET_ALL} {np.mean(self.episode_lengths):.2f}")
        print(f"{Fore.RED}Total Invalid Actions:{Style.RESET_ALL} {self.total_invalid_actions}")

def make_env():
    def _init():
        env = CustomGameEnv()
        env = ActionMasker(env, get_action_mask)
        env = SafeEnv(env)  # Wrap with SafeEnv to catch crashes
        return env
    return _init

def run_testing_games(model, env, num_games=100, render=False, max_steps=1000):
    game_stats = {
        'wins': 0,
        'losses': 0,
        'rewards': [],
        'steps': [],
        'rounds': [],
        'invalid_actions': 0,
        'timeouts': 0
    }
    
    start_time = time.time()
    pbar = tqdm(total=num_games, desc="Testing Games")
    
    for game_num in range(num_games):
        reset_result = env.reset()
        obs = reset_result if not isinstance(reset_result, tuple) else reset_result[0]
        total_reward = 0
        steps = 0
        done = False
        truncated = False
        game_invalid_actions = 0
        
        while not (done or truncated) and steps < max_steps:
            current_player = env.envs[0].env.current_player
            
            try:
                action_mask = get_action_mask(env.envs[0].env)
            except Exception as e:
                print(f"Error getting action mask: {e}")
                action_mask = np.ones(env.envs[0].env.action_space.n)
            
            valid_actions = np.where(action_mask == 1)[0]
            if len(valid_actions) == 0:
                print(f"Game {game_num + 1}, Step {steps}: No valid actions available for player {current_player}!")
                action_mask = np.ones(env.envs[0].env.action_space.n)
                valid_actions = np.arange(env.envs[0].env.action_space.n)
            
            if current_player == 0:
                try:
                    action, _ = model.predict(obs, action_masks=np.array([action_mask]), deterministic=True)
                    if action_mask[action] != 1:
                        print(f"Game {game_num + 1}, Step {steps}: Agent chose invalid action {action}!")
                        game_invalid_actions += 1
                        action = np.random.choice(valid_actions)
                except Exception as e:
                    print(f"Error predicting action: {e}")
                    action = np.random.choice(valid_actions)
            else:
                action = np.random.choice(valid_actions)
            
            try:
                step_result = env.step([action])
                if len(step_result) == 4:
                    obs, reward, done_array, info = step_result
                    truncated_array = [False]
                else:
                    obs, reward, done_array, truncated_array, info = step_result
                
                info = info[0] if isinstance(info, list) else info
                if 'invalid_action' in info and info['invalid_action']:
                    game_invalid_actions += 1
                
                done = done_array[0]
                truncated = truncated_array[0]
                
                if current_player == 0:
                    total_reward += reward[0]
                
                steps += 1
                
                if render:
                    env.envs[0].env.render()
                    
            except Exception as e:
                print(f"Error during step execution: {e}")
                import traceback
                traceback.print_exc()
                game_invalid_actions += 1
                done = True
                break
        
        game_stats['invalid_actions'] += game_invalid_actions
        
        if steps >= max_steps and not (done or truncated):
            print(f"Game {game_num + 1} reached max steps ({max_steps}) without terminating.")
            game_stats['timeouts'] += 1
        
        game_stats['rewards'].append(total_reward)
        game_stats['steps'].append(steps)
        
        team_won = info.get('team_has_won', 0)
        if team_won == 1:
            game_stats['wins'] += 1
            print(f"Game {game_num + 1}: Team A (model's team) wins!")
        elif team_won == 2:
            game_stats['losses'] += 1
            print(f"Game {game_num + 1}: Team B wins.")
        else:
            if steps >= max_steps:
                game_stats['losses'] += 1
                print(f"Game {game_num + 1}: No winner (timeout) - counted as loss.")
        
        rounds = info.get('rounds_completed', 0)
        game_stats['rounds'].append(rounds)
        
        win_rate = (game_stats['wins'] / (game_num + 1)) * 100
        elapsed_time = time.time() - start_time
        avg_time_per_game = elapsed_time / (game_num + 1)
        remaining_games = num_games - (game_num + 1)
        eta_seconds = avg_time_per_game * remaining_games
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        
        pbar.set_postfix({
            'win_rate': f"{win_rate:.1f}%",
            'avg_reward': f"{np.mean(game_stats['rewards']):.1f}",
            'avg_steps': f"{np.mean(game_stats['steps']):.1f}",
            'invalid': game_stats['invalid_actions'],
            'timeouts': game_stats['timeouts'],
            'ETA': eta_str
        })
        pbar.update(1)
    
    pbar.close()
    
    timestamp = time.strftime("%Y-%m-d %H:%M:%S")
    summary = f"{Fore.CYAN}===== Testing Results ({num_games} games) ====={Style.RESET_ALL}\n"
    summary += f"{Fore.GREEN}Timestamp:{Style.RESET_ALL} {timestamp}\n"
    summary += f"{Fore.GREEN}Model:{Style.RESET_ALL} {str(model.__class__.__name__)} (Player 0)\n"
    summary += f"{Fore.GREEN}Environment:{Style.RESET_ALL} {str(env.envs[0].env.__class__.__name__)}\n"
    summary += f"{Fore.GREEN}Maximum Steps per Game:{Style.RESET_ALL} {max_steps}\n"
    summary += f"{Fore.YELLOW}Win Rate:{Style.RESET_ALL} {win_rate:.2f}%\n"
    summary += f"{Fore.YELLOW}Average Reward:{Style.RESET_ALL} {np.mean(game_stats['rewards']):.2f}\n"
    summary += f"{Fore.YELLOW}Average Steps per Game:{Style.RESET_ALL} {np.mean(game_stats['steps']):.2f}\n"
    summary += f"{Fore.YELLOW}Average Rounds per Game:{Style.RESET_ALL} {np.mean(game_stats['rounds']):.2f}\n"
    summary += f"{Fore.RED}Total Invalid Actions:{Style.RESET_ALL} {game_stats['invalid_actions']}\n"
    summary += f"{Fore.RED}Total Timeouts:{Style.RESET_ALL} {game_stats['timeouts']}\n"
    summary += f"{Fore.CYAN}Testing completed.{Style.RESET_ALL}\n"
    
    print(f"\n{summary}")
    
    results_dir = "./test_results/"
    os.makedirs(results_dir, exist_ok=True)
    
    df = pd.DataFrame({
        'game_num': range(1, num_games + 1),
        'reward': game_stats['rewards'],
        'steps': game_stats['steps'],
        'rounds': game_stats['rounds'],
        'win': [1 if i < game_stats['wins'] else 0 for i in range(num_games)]
    })
    
    timestamp_file = time.strftime("%Y%m%d-%H%M%S")
    df.to_csv(f"{results_dir}test_results_{timestamp_file}.csv", index=False)
    
    plain_summary = f"===== Testing Results ({num_games} games) =====\n"
    plain_summary += f"Timestamp: {timestamp}\n"
    plain_summary += f"Model: {str(model.__class__.__name__)} (Player 0)\n"
    plain_summary += f"Environment: {str(env.envs[0].env.__class__.__name__)}\n"
    plain_summary += f"Maximum Steps per Game: {max_steps}\n"
    plain_summary += f"Win Rate: {win_rate:.2f}%\n"
    plain_summary += f"Average Reward: {np.mean(game_stats['rewards']):.2f}\n"
    plain_summary += f"Average Steps per Game: {np.mean(game_stats['steps']):.2f}\n"
    plain_summary += f"Average Rounds per Game: {np.mean(game_stats['rounds']):.2f}\n"
    plain_summary += f"Total Invalid Actions: {game_stats['invalid_actions']}\n"
    plain_summary += f"Total Timeouts: {game_stats['timeouts']}\n"
    plain_summary += "Testing completed.\n"
    with open(f"{results_dir}test_results_summary_{timestamp_file}.txt", "w") as f:
        f.write(plain_summary)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].plot(df['game_num'], df['reward'])
    axs[0, 0].set_title('Reward per Game')
    axs[0, 0].set_xlabel('Game Number')
    axs[0, 0].set_ylabel('Total Reward')
    
    df['win_rolling'] = df['win'].rolling(window=10, min_periods=1).mean() * 100
    axs[0, 1].plot(df['game_num'], df['win_rolling'])
    axs[0, 1].set_title('Win Rate (10-game rolling average)')
    axs[0, 1].set_xlabel('Game Number')
    axs[0, 1].set_ylabel('Win Rate (%)')
    axs[0, 1].set_ylim(0, 100)
    
    axs[1, 0].plot(df['game_num'], df['steps'])
    axs[1, 0].set_title('Steps per Game')
    axs[1, 0].set_xlabel('Game Number')
    axs[1, 0].set_ylabel('Steps')
    
    axs[1, 1].plot(df['game_num'], df['rounds'])
    axs[1, 1].set_title('Rounds per Game')
    axs[1, 1].set_xlabel('Game Number')
    axs[1, 1].set_ylabel('Rounds')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}test_results_{timestamp_file}.png")
    
    return game_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the Jackaroo game model.")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--timesteps", type=int, default=50000000)
    parser.add_argument("--num_envs", type=int, default=1)  # Default to 1 to avoid memory issues
    parser.add_argument("--checkpoint_freq", type=int, default=5000000)  # Updated from your run
    parser.add_argument("--log_interval", type=int, default=1000000)
    parser.add_argument("--test_games", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("model_path", nargs="?", help="Path to the model.zip file for testing")
    args = parser.parse_args()

    print(f"Using gym version: {gym.__version__}")
    print(f"Using gymnasium version: {gymnasium.__version__}")

    if args.mode == "train":
        num_envs = args.num_envs
        if num_envs > 1:
            print(f"{Fore.YELLOW}Warning: Using DummyVecEnv with {num_envs} environments to prevent memory crashes. For parallelism, optimize CustomGameEnv.{Style.RESET_ALL}")
            vec_env = DummyVecEnv([make_env() for _ in range(num_envs)])
        else:
            vec_env = DummyVecEnv([make_env()])

        log_dir = "./tensorboard_logs/"
        checkpoint_dir = "./checkpoints/"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_config = {
            "learning_rate": lambda progress: 3e-4 * (1 - progress),
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": lambda progress: 0.2,
            "clip_range_vf": lambda progress: 0.2,
            "target_kl": 0.05,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "tensorboard_log": log_dir,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "policy_kwargs": {
                "net_arch": dict(pi=[256, 256], vf=[256, 256]),
                "activation_fn": torch.nn.ReLU,
                "ortho_init": True
            },
            "verbose": 0
        }

        if args.resume:
            print(f"{Fore.GREEN}Resuming training from {checkpoint_dir}{Style.RESET_ALL}")
            import glob
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "ppo_jackaroo_maskable_*.zip"))
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                try:
                    resume_config = {"tensorboard_log": log_dir, "device": model_config["device"], "verbose": 0}
                    model = MaskablePPO.load(latest_checkpoint, env=vec_env, **resume_config)
                    model.learning_rate = model_config["learning_rate"]
                    model.clip_range = model_config["clip_range"]
                    model.clip_range_vf = model_config["clip_range_vf"]
                    model.target_kl = model_config["target_kl"]
                    print(f"{Fore.GREEN}Successfully resumed training from: {latest_checkpoint}{Style.RESET_ALL}")
                    import re
                    step_match = re.search(r'_(\d+)_steps', latest_checkpoint)
                    if step_match:
                        last_step = int(step_match.group(1))
                        print(f"{Fore.GREEN}Continuing from step {last_step}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Error loading checkpoint: {e}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Starting fresh training session instead.{Style.RESET_ALL}")
                    model = MaskablePPO("MultiInputPolicy", vec_env, **model_config)
            else:
                print(f"{Fore.YELLOW}No checkpoint found. Starting training from scratch.{Style.RESET_ALL}")
                model = MaskablePPO("MultiInputPolicy", vec_env, **model_config)
        else:
            model = MaskablePPO("MultiInputPolicy", vec_env, **model_config)

        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_freq // num_envs,
            save_path=checkpoint_dir,
            name_prefix='ppo_jackaroo_maskable',
            verbose=1
        )

        progress_callback = EnhancedProgressBarCallback(
            total_timesteps=args.timesteps,
            num_envs=num_envs,
            log_interval=args.log_interval,
            eta_window=1000
        )

        callback_list = CallbackList([checkpoint_callback, progress_callback])

        print(f"{Fore.CYAN}Starting training with {num_envs} environments...{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Using device: {model.device}{Style.RESET_ALL}")
        
        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=callback_list,
                reset_num_timesteps=not args.resume
            )
        except MemoryError:
            print(f"{Fore.RED}MemoryError: Training crashed due to memory allocation failure. Falling back to num_envs=1.{Style.RESET_ALL}")
            vec_env = DummyVecEnv([make_env()])
            model = MaskablePPO("MultiInputPolicy", vec_env, **model_config)
            progress_callback = EnhancedProgressBarCallback(total_timesteps=args.timesteps, num_envs=1, log_interval=args.log_interval, eta_window=1000)
            callback_list = CallbackList([checkpoint_callback, progress_callback])
            model.learn(total_timesteps=args.timesteps, callback=callback_list, reset_num_timesteps=True)

        final_model_path = f"ppo_jackaroo_maskable_final_{args.timesteps}_steps.zip"
        model.save(final_model_path)
        print(f"{Fore.GREEN}Training completed and final model saved as '{final_model_path}'.{Style.RESET_ALL}")

        print(f"\n{Fore.CYAN}Evaluating final model performance...{Style.RESET_ALL}")
        test_env = DummyVecEnv([make_env()])
        run_testing_games(model, test_env, num_games=10, render=False)

    elif args.mode == "test":
        if args.model_path is None:
            print(f"{Fore.RED}Error: model_path is required for test mode.{Style.RESET_ALL}")
            exit(1)
        
        env = DummyVecEnv([make_env()])
        try:
            model = MaskablePPO.load(args.model_path, env=env)
            print(f"{Fore.GREEN}Loaded model from {args.model_path}{Style.RESET_ALL}")
            run_testing_games(model, env, num_games=args.test_games, render=args.render)
            print(f"{Fore.CYAN}Testing completed.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error loading model: {e}{Style.RESET_ALL}")
            exit(1)