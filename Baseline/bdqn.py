import bsuite
from bsuite.baselines.tf.boot_dqn.agent import BootstrappedDqn, make_ensemble
from bsuite.baselines import experiment
from bsuite.environments.deep_sea import DeepSea
from bsuite.logging import csv_logging
import os
import sys
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.append('/scratch/Rabbit/work/ABC_SMC_RL/')
sys.path.append('/Users/guojiaqi/work/ABC_SMC_RL/')
from parameter import *

import numpy as np
import pandas as pd
import termcolor
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import sonnet as snt
import argparse
import dm_env
from typing import Callable, Optional, Sequence
parser = argparse.ArgumentParser()
parser.add_argument('-R', '--random_prior', default=False, action='store_true', help='Bool type')
parser.add_argument('--seed', default=SEED, type=int)
args = parser.parse_args()
time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
RANDOM_PRIOR = args.random_prior
seed = args.seed

def load_and_record_to_csv_(env,
                           results_dir: str,
                           overwrite: bool = False) -> dm_env.Environment:
    """Returns a bsuite environment that saves results to CSV.

    To load the results, specify the file path in the provided notebook, or to
    manually inspect the results use:
    ```python
    from bsuite.logging import csv_load

    results_df, sweep_vars = csv_load.load_bsuite(results_dir)
    ```
    Args:
        bsuite_id: The bsuite id identifying the environment to return. For example,
        "catch/0" or "deep_sea/3".
        results_dir: Path to the directory to store the resultant CSV files. Note
        that this logger will generate a separate CSV file for each bsuite_id.
        overwrite: Whether to overwrite existing CSV files if found.

    Returns:
        A bsuite environment determined by the bsuite_id.
    """
    raw_env = env
    termcolor.cprint(
        f'Logging results to CSV file for each bsuite_id in {results_dir}.',
        color='yellow',
        attrs=['bold'])
    return csv_logging.wrap_environment(
        env=raw_env,
        bsuite_id=f'deep_sea/{seed}_{time}',
        results_dir=results_dir,
        overwrite=overwrite,
    )
# Define the custom DeepSea environment
class CustomDeepSea(DeepSea):
    def __init__(self, goal_reward=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_reward = goal_reward
        
    def _step(self, action: int) -> dm_env.TimeStep:
        reward = 0.
        action_right = action == self._action_mapping[self._row, self._column]

        # Reward calculation
        if self._column == self._size - 1 and action_right:
            reward += 1.
            self._denoised_return += 1.
        if not self._deterministic:  # Noisy rewards on the 'end' of chain.
            if self._row == self._size - 1 and self._column in [0, self._size - 1]:
                reward += self._rng.randn()

        # Transition dynamics
        if action_right:
            if self._rng.rand() > 1 / self._size or self._deterministic:
                self._column = np.clip(self._column + 1, 0, self._size - 1)
            reward -= self._unscaled_move_cost / self._size
        else:
            reward += self._unscaled_move_cost / self._size
            if self._row == self._column:  # You were on the right path and went wrong
                self._bad_episode = True
            self._column = np.clip(self._column - 1, 0, self._size - 1)
        self._row += 1

        observation = self._get_observation()
        if self._row == self._size:
            if self._bad_episode:
                self._total_bad_episodes += 1
            return dm_env.termination(reward=reward, observation=observation)
        return dm_env.transition(reward=reward, observation=observation)

# Define the ensemble of networks without randomized priors
def create_ensemble(num_networks, obs_spec, action_spec):
    return [snt.nets.MLP([50, 50], activation=tf.nn.relu) for _ in range(num_networks)]

# Define the optimizer
def create_optimizer(learning_rate):
    return snt.optimizers.Adam(learning_rate)

# Initialize the CustomDeepSea environment
results_dir = f'/tmp/bsuite_results_Random{RANDOM_PRIOR}'
custom_env = CustomDeepSea(size=5, randomize_actions=False, seed=seed)

#Modified env by adding positive reward for going left
env = load_and_record_to_csv_(custom_env, results_dir=results_dir, overwrite=True)

#original env
# env = bsuite.load_and_record(bsuite_id='deep_sea/0', save_path=results_dir, logging_mode='csv', overwrite=True)

# # Initialize the BootstrappedDqn agent
prior_scale = 3. if RANDOM_PRIOR else 0.

agent = BootstrappedDqn(
    obs_spec=env.observation_spec(),
    action_spec=env.action_spec(),
    ensemble = make_ensemble(num_actions=env.action_spec().num_values, num_ensemble=20, prior_scale=prior_scale), #if RANDOM_PRIOR else create_ensemble(20, env.observation_spec(), env.action_spec()),
    batch_size=32,
    discount=0.99,
    replay_capacity=100000,
    min_replay_size=128,
    sgd_period=1,
    target_update_period=4,
    optimizer=create_optimizer(1e-3),
    mask_prob=0.5,
    noise_scale=0.1,
    epsilon_fn=lambda step: 0.1,  # Example epsilon function
    seed=seed
)
# Run the BootstrappedDqn agent on the Deep Sea environment
# for episode in range(1000):
#     timestep = env.reset()  # Reset environment to start a new episode
#     done = False

#     while not done:
#         # Select action using the agent
#         action = agent.select_action(timestep)

#         # Step in the environment
#         next_timestep = env.step(action)  # Call the custom _step method

#         # Update agent with the transition
#         agent.update(
#             timestep=timestep,
#             action=action,
#             new_timestep=next_timestep
#         )

#         # Move to the next timestep
#         timestep = next_timestep
#         done = timestep.last()  # Check if episode has ended

#         # Optionally: Log or print some information for monitoring
#         # print(f"Episode: {episode}, Reward: {next_timestep.reward}, Done: {done}")

# env.close()

experiment.run(
      agent=agent,
      environment=env,
      num_episodes=1000,
      verbose=True)

# Load the results for plotting
def load_multiple_csv(results_dir):
    """Loads multiple CSV files from a directory."""
    data_frames = []
    for file in os.listdir(results_dir):
        if file.endswith('.csv'):
            path = os.path.join(results_dir, file)
            df = pd.read_csv(path)
            data_frames.append(df)
    return data_frames

def aggregate_data(data_frames):
    """Aggregates data from multiple data frames."""
    # Ensure all data frames have the same number of episodes
    min_length = min(len(df) for df in data_frames)
    truncated_dfs = [df.iloc[:min_length] for df in data_frames]

    # Stack all data frames
    combined_df = pd.concat(truncated_dfs, axis=0)

    # Group by episode and calculate mean and standard deviation
    mean_df = combined_df.groupby('episode').mean().reset_index()
    std_df = combined_df.groupby('episode').std().reset_index()

    return mean_df, std_df

def plot_results(mean_df, std_df):
    """Plots the results with mean and standard deviation."""
    plt.figure(figsize=(10, 6))

    plt.plot(mean_df['episode'], mean_df['episode_return'], label='Mean Return')
    plt.fill_between(
        mean_df['episode'],
        mean_df['episode_return'] - std_df['episode_return'],
        mean_df['episode_return'] + std_df['episode_return'],
        color='blue',
        alpha=0.2,
        label='±1 Standard Deviation'
    )

    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Return over Episodes (with different seeds)')
    plt.legend()
    plt.grid()
    plt.show()

# Load and aggregate the data
df = load_multiple_csv(results_dir)
mean_df, std_df = aggregate_data(df)

# Plot the results
plot_results(mean_df, std_df)
