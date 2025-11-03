# JackarooEnv

Lightweight OpenAI Gym-compatible reinforcement learning environment for the Jackaroo board game.

One-line: A Gym-compatible environment to experiment with RL agents playing the Jackaroo board game.

## Table of contents
- About
- Features
- Requirements
- Installation
- Quick start
- API / interface
- Observation & action spaces
- Rewards & termination
- Self-play & multi-agent notes
- Example: training with Stable Baselines3
- Tips & debugging
- Results (optional)
- Contributing
- License
- Contact

## About
JackarooEnv provides a Gym/Gymnasium-compatible environment that simulates the Jackaroo board game. It is intended for research and learning: to prototype RL agents for board-game strategies, run self-play experiments, and benchmark algorithms on discrete turn-based play.

## Features
- Gym/Gymnasium API (env.reset(), env.step(), env.render())
- Turn-based, discrete-action gameplay matching Jackaroo rules (configurable)
- Single-agent and self-play modes (if implemented)
- Example training scripts for Stable Baselines3
- Lightweight and easy to extend to alternate rule variants

## Requirements
- Python 3.8+
- pip
- Core packages:
  - gym or gymnasium
  - numpy
- Optional (recommended for examples):
  - stable-baselines3
  - torch
  - matplotlib

## Install
Clone the repo and install:

```bash
git clone https://github.com/okabe000/jackarooEnv.git
cd jackarooEnv
pip install -e .
```

Or install dependencies only:

```bash
pip install gym numpy
# Optional for examples:
pip install stable-baselines3[extra] torch matplotlib
```

## Quick start
Basic usage with a Gym-compatible interface:

```python
import gym
import jackarooenv  # package name if packaged; otherwise import path

env = gym.make("Jackaroo-v0")
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # random legal move
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
```

## API / Interface
The environment implements the standard Gym API:
- env.reset(seed=None) -> observation
- env.step(action) -> (observation, reward, done, info)
- env.render(mode="human") -> optional board visualization
- env.close()

## Observation & action spaces
Below are reasonable defaults — replace with exact values from the implementation if different.

- Observation:
  - Type: Dict or Box representing board state and current player
  - Example: Dict({"board": Box(0, 1, shape=(N_CELLS,), dtype=np.int8), "current_player": Discrete(num_players)})
  - Description: representation of the board (piece positions), current player turn, and any game-specific state (pieces in hand, scores).

- Action:
  - Type: Discrete
  - Size: Discrete(M) where each action maps to a legal move index for the current player
  - Description: a discrete index representing a legal board move (move a piece, place token, pass, etc.). The environment may provide legal-action masks in info['legal_actions'] or filter invalid actions.

## Rewards & termination
- Reward: By default the environment uses a sparse win/loss reward: +1 for a win, 0 for a draw, -1 for a loss.
- Optional shaping: per-move bonuses (e.g., capturing or advancing) or small step penalties can be enabled via config.
- Episode termination:
  - Win: a player meets the game-specific win condition (e.g., all pieces home).
  - Loss: the opponent meets the win condition.
  - Draw / Max steps: fallback if max steps reached.

## Self-play & multi-agent notes
- If the environment supports self-play, you can run matches between policies by providing opponent policies or using built-in heuristic/random opponents.
- For single-agent training, commonly you train one agent while the environment runs opponent moves automatically. Consider using wrappers that expose only the controlled player observation/action space.

## Example: training with Stable Baselines3
Minimal training example using PPO. For turn-based discrete games consider wrappers that convert the environment into a standard MDP (e.g., exposing only the controlled player and letting the env advance through opponent turns).

```python
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env_id = "Jackaroo-v0"
env = make_vec_env(lambda: gym.make(env_id), n_envs=8)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("ppo_jackaroo")
```

## Tips & debugging
- Legal-action masking: returning info['legal_actions'] (boolean mask or list) helps training and prevents illegal moves.
- Self-play: use population-based training or league self-play to get stronger opponents.
- Rendering: env.render() should show an ASCII or simple matplotlib board for quick debugging.
- Reproducibility: set seeds for numpy, torch, and the environment (env.reset(seed=...)).

## Results (optional)
Add training curves, final rewards, win rates, and a short summary of experiments here. Attach images or link to training logs.

## Contributing
Contributions welcome. Suggested process:
1. Open an issue to discuss significant changes or new features.
2. Fork the repo and create a feature branch.
3. Create a PR with tests and a clear description.
4. Keep commits small and focused.

Suggested labels: enhancement, bug, question, docs

## License
If you don't have a preference, MIT is a good default. Add a LICENSE file to the repo if you choose MIT.

## Contact
Repo: https://github.com/okabe000/jackarooEnv
Author: okabe000
Email: TODO (optional)

## Acknowledgements
- Inspired by OpenAI Gym and standard RL benchmarks.

## Appendix: useful commands
- Run lint/tests:
  - lint: flake8 / black
  - tests: pytest
- Run example training: python examples/train_sb3_ppo.py
- Run visualization: python examples/render_demo.py
