from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import time
import numpy as np
from collections import deque

class EnhancedProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0, log_interval=1000, window_size=100, num_envs=1):
        super(EnhancedProgressBarCallback, self).__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.last_update_time = None
        self.last_step_count = 0
        self.log_interval = log_interval
        self.num_envs = num_envs
        
        # Metrics tracking
        self.rewards_buffer = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.win_buffer = deque(maxlen=window_size)
        self.round_counts = deque(maxlen=window_size)
        self.invalid_actions = deque(maxlen=window_size)
        self.global_step_count = 0

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")
        self.start_time = time.time()
        self.last_update_time = self.start_time

    def _on_step(self):
        self.global_step_count += self.num_envs
        steps_to_update = self.global_step_count - self.pbar.n
        if steps_to_update > 0:
            self.pbar.update(steps_to_update)
        
        current_time = time.time()
        
        infos = self.locals.get('infos', [{}])
        for info in infos:
            if info.get('episode') is not None:
                episode_info = info.get('episode')
                self.rewards_buffer.append(episode_info.get('r', 0))
                self.episode_lengths.append(episode_info.get('l', 0))
                
                if 'team_has_won' in info:
                    won = 1 if info['team_has_won'] == 1 else 0
                    self.win_buffer.append(won)
                
                if 'rounds_completed' in info:
                    self.round_counts.append(info['rounds_completed'])
                
                if 'invalid_actions' in info:
                    self.invalid_actions.append(info['invalid_actions'])

        if self.num_timesteps % self.log_interval == 0:
            elapsed = current_time - self.last_update_time
            steps_since_last = self.num_timesteps - self.last_step_count
            steps_per_sec = steps_since_last / elapsed if elapsed > 0 else 0
            
            self.logger.record("performance/steps_per_second", steps_per_sec)
            
            if len(self.rewards_buffer) > 0:
                self.logger.record("metrics/mean_reward", np.mean(self.rewards_buffer))
                self.logger.record("metrics/mean_episode_length", np.mean(self.episode_lengths))
            
            if len(self.win_buffer) > 0:
                win_rate = np.mean(self.win_buffer) * 100
                self.logger.record("metrics/win_rate", win_rate)
                
            if len(self.round_counts) > 0:
                self.logger.record("metrics/avg_rounds_per_game", np.mean(self.round_counts))
                
            if len(self.invalid_actions) > 0:
                self.logger.record("metrics/invalid_action_rate", np.mean(self.invalid_actions))
            
            postfix_dict = {"steps/s": f"{steps_per_sec:.2f}"}
            if len(self.win_buffer) > 0:
                postfix_dict["win_rate"] = f"{win_rate:.1f}%"
            if len(self.rewards_buffer) > 0:
                postfix_dict["reward"] = f"{np.mean(self.rewards_buffer):.1f}"
                
            self.pbar.set_postfix(postfix_dict)
            
            self.last_update_time = current_time
            self.last_step_count = self.num_timesteps

        return True

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
        
        avg_steps_per_sec = self.total_timesteps / total_time
        self.pbar.set_postfix({"avg_steps/s": f"{avg_steps_per_sec:.2f}"})
        self.pbar.close()
        
        print(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"Average steps per second: {avg_steps_per_sec:.2f}")
        
        if len(self.win_buffer) > 0:
            win_rate = np.mean(self.win_buffer) * 100
            print(f"Final win rate: {win_rate:.2f}%")
        
        if len(self.rewards_buffer) > 0:
            print(f"Average reward: {np.mean(self.rewards_buffer):.2f}")
            print(f"Average episode length: {np.mean(self.episode_lengths):.2f}")
