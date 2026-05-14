import numpy as np
# from gym import spaces
# import random
from matplotlib import colors, colormaps
import matplotlib.pyplot as plt
from gym import spaces
from scipy.sparse import diags
from scipy.stats import multivariate_normal
# import sys
# import os
# os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
# sys.path.append('/scratch/Rabbit/work/ABC_SMC_RL/')
# sys.path.append('/Users/guojiaqi/work/ABC_SMC_RL/')
'''module import'''
from model import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.stats import multivariate_normal

class IrregularGridWorld:
    def __init__(self, n_cell, obstacle_mask=None, starting_position=(0, 0), horizon=20):
        """
        shape: (rows, cols) of the grid
        obstacle_mask: boolean np.array where True = obstacle
        reward_distributions: dict mapping (i,j) -> function returning a reward sample
        """
        self.rows, self.cols = n_cell
        self.n_cell = n_cell
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(n_cell[0] * n_cell[1])
        self.observation_space_high = (n_cell[0], n_cell[1])
        self.observation_space_low = (0, 0)
        self.action_size = self.action_space.n
        self.obstacle_mask = obstacle_mask if obstacle_mask is not None else np.zeros(n_cell, dtype=bool)
        self.grid = np.zeros(n_cell)
        self.start_pos = starting_position
        print("Starting position:", self.start_pos)
        self.agent_pos = self.start_pos
        self.horizon = horizon
        self.reset(reset_features=True)
        learnable_idx = []
        for i in range(self.n_cell[0]):
            for j in range(self.n_cell[1]):
                learnable_idx.append(np.array((i, j)))
        self.learnable_idx = tuple(torch.tensor(np.array(learnable_idx)).T)
        self.names = learnable_idx
        self.learnable_no = range((n_cell[0]) * n_cell[1] * self.action_space.n)
        self.learnable_shape = ((n_cell[0]),  n_cell[1])
        # self.goal_idx = None
        # self.env.not_learnable_idx = None
        
    def reset(self, starting_position=None, reset_features=False):
        if starting_position is not None:
            self.start_pos = starting_position
        else:
            self.start_pos = self.start_pos
        if self.start_pos[0] < 0 or self.start_pos[0] >= self.rows or \
           self.start_pos[1] < 0 or self.start_pos[1] >= self.cols:
            raise ValueError("Starting position out of bounds.")
        if self.obstacle_mask[self.start_pos]:
            raise ValueError("Starting position is an obstacle.")
        self.agent_pos = self.start_pos
        self.t = 0   
        if reset_features:
            self.features = self.sample_gaussian_mrf()
            feature_values = np.where(
                self.obstacle_mask,
                np.nan,
                self.features  # or use np.exp(self.features), etc.
            )
            self.rewards = self.reward_function(feature_values)
        return self.agent_pos, self.rewards
    
    def reward_function(self, feature_value):
        return feature_value**2  # or np.exp(feature_value), etc.

    def sample_gaussian_mrf(self, sigma=1.0, beta=1.0):
        rows, cols = self.rows, self.cols
        N = rows * cols
        idx = lambda i, j: i * cols + j

        precision = lil_matrix((N, N))

        for i in range(rows):
            for j in range(cols):
                center = idx(i, j)
                if self.obstacle_mask[i, j]:
                    precision[center, center] = 1e6  # Large precision for obstacle (near-zero variance)
                    continue

                precision[center, center] = 4  # diagonal term
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and not self.obstacle_mask[ni, nj]:
                        neighbor = idx(ni, nj)
                        precision[center, neighbor] = -beta

        cov = np.linalg.inv(precision.toarray()) * sigma
        feature_vector = multivariate_normal.rvs(mean=np.zeros(N), cov=cov)
        feature_map = feature_vector.reshape((rows, cols))
        feature_map[self.obstacle_mask] = np.nan
        return feature_map

    
    def step(self, action):
        """
        action: 0=up, 1=right, 2=down, 3=left
        Returns (new_pos, reward, done)
        """
        
        direction_map = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1),  # left
        }

        if action not in direction_map:
            raise ValueError("Invalid action. Must be 0, 1, 2, or 3.")

        di, dj = direction_map[action]
        ni, nj = self.agent_pos[0] + di, self.agent_pos[1] + dj

        # Check bounds and obstacle
        if 0 <= ni < self.rows and 0 <= nj < self.cols and not self.obstacle_mask[ni, nj]:
            self.agent_pos = (ni, nj)
            reward = self.rewards[ni, nj]
        else:
            # Stay in place, maybe penalize
            reward = -1.0  # small penalty for invalid move

          # Could customize if you add a goal/horizon
        self.t += 1
        done = self.t >= self.horizon
        return self.agent_pos, reward, done
    
    def rollout(self, actions):
        self.reset()
        trajectory = [self.start_pos]
        for action in actions:
            new_pos, _, _ = self.step(action)
            trajectory.append(new_pos)
        return trajectory
    
    @staticmethod
    def get_neighbors(i, j, max_rows=None, max_cols=None):
        """
        Returns list of valid neighboring cells (up, down, left, right)
        that are not obstacles and are within grid bounds.
        """
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < max_rows and 0 <= nj < max_cols:
                neighbors.append((ni, nj))
        
        return neighbors
    
    def build_flat_neighbor_dict(self):
        """
        Builds a dictionary mapping each flattened index I in the grid
        to a list of its valid 4-neighbor indices (also flattened).
        """
        neighbor_dict = {}
        for i in range(self.rows):
            for j in range(self.cols):
                current_index = i * self.cols + j
                neighbors = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.rows and 0 <= nj < self.cols:
                        neighbor_index = ni * self.cols + nj
                        neighbors.append(neighbor_index)
                neighbor_dict[current_index] = neighbors
        return neighbor_dict


    def visualize(self):
        plt.figure(figsize=(6, 6))
        cmap = plt.cm.viridis
        cmap.set_bad(color='black')  # obstacle cells
        plt.imshow(self.rewards, cmap=cmap, origin='upper')
        plt.title("Irregular GridWorld (Stochastic Rewards Sample)")
        plt.colorbar(label="Reward")
        plt.grid(True)
        plt.show()
                   
    
if __name__ == '__main__':
    from Planning.MCTS_old import MCTSPlanner
    # Example usage

    rows, cols = 3, 8
    obstacles = np.zeros((rows, cols), dtype=bool)
    # Create irregular shape
    # obstacles[3:5, 2:7] = True
    # obstacles[7, :] = True
    # obstacles[:, 0] = True
    # obstacles[0, :] = True

    # Define stochastic reward distributions

    def normal_reward(mu, sigma):
        return lambda: np.random.normal(mu, sigma)
    
    # Create environment
    env = IrregularGridWorld((rows, cols), obstacles)
    # env.visualize()
    
    plt.subplot(1, 2, 1)
    plt.imshow(env.features, cmap="coolwarm")
    plt.title("Feature Map")

    plt.subplot(1, 2, 2)
    plt.imshow(env.rewards, cmap="viridis")
    plt.title("Reward Map")
    plt.show()
        
    planner = MCTSPlanner(env.rewards, T=20, start=(2, 2))
    result = planner.run(simulations=5000)
    if result:
        end_pos, path = result
        print("Best path:", path)
        print("Total reward:", sum(env.rewards[i, j] for i, j in path))
    else:
        print("No valid closed loop found")