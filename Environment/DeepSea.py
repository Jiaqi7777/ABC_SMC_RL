import numpy as np
from gym import spaces
import random
from matplotlib import colors, colormaps
import matplotlib.pyplot as plt
import sys
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.append('/scratch/Rabbit/work/ABC_SMC_RL/')
sys.path.append('/Users/guojiaqi/work/ABC_SMC_RL/')
'''module import'''
from model import *

class DeepSea:
    def __init__(self, depth, starting_position=(0,0), goal_position=(-1,-1)):
        self.env_name = 'GridWorld'
        self.n_cell = n_cell = (depth, depth)
        self.starting_position = starting_position
        if goal_position == (-1, -1):
            goal_position = ( n_cell[0] - 1, n_cell[1] - 1 )
        self.goal_position = goal_position
        self.treasure = 1 if np.random.uniform(0, 1) > 0 else - 1
        self.done = False
        self.observation_space = spaces.Discrete(n_cell[0] * n_cell[1])
        self.observation_space_high = (n_cell[0], n_cell[1])
        self.observation_space_low = (0, 0)
        self.action_space = spaces.Discrete(2)
        self.action_size = self.action_space.n
        self.R =  np.round(np.array([-1, 1]) / ( 2 * depth ), 2)
        print('goal: ', goal_position)
        print('start: ', starting_position)
        print('World Scale: ', f'{n_cell[0]}x{n_cell[1]}')
        gridworld = np.arange(
        self.observation_space.n
        ).reshape((n_cell[0], n_cell[1]))
        self.P = np.zeros((n_cell[0], n_cell[1], self.action_space.n, 2), dtype='int32')
        for s in gridworld.flat:
            row, col = np.argwhere(gridworld == s)[0]
            for a, d in zip(
                    range(self.action_space.n), [(1, 1), (1, -1)]
                    ):
                next_row = max(0, min(row + d[0], n_cell[0] - 1))
                next_col = max(0, min(col + d[1], n_cell[1] - 1))
                s_prime = [next_row, next_col] #gridworld[next_row, next_col]
                self.P[row, col, a] = s_prime
                self.P[goal_position + (a, )] = goal_position
                
        # self.names = [f'{i,j,a}' for i in range(n_cell[1]) for j in range(n_cell[0]) for a in range(self.action_space.n)]
        # self.learnable_idx = (slice(None, -1), slice(None))
        learnable_idx = []
        for i in range(self.n_cell[0] - 1):
            for j in range(i+1):
                learnable_idx.append(np.array((i, j, 0,)))
                learnable_idx.append(np.array((i, j, 1,)))
        self.learnable_idx = tuple(torch.tensor(learnable_idx).T)
        self.names = learnable_idx
        self.learnable_no = range((n_cell[0] - 1) * n_cell[1] * self.action_space.n)
        self.learnable_shape = ((n_cell[0] - 1),  n_cell[1])
        self.goal_idx = (-1, Ellipsis)
        not_learnable_idx = []
        for i in range(n_cell[0]):
            for j in range(i+1, n_cell[0]):
                not_learnable_idx.append(np.array((i, j, 0,)))
                not_learnable_idx.append(np.array((i, j, 1,)))
        self.not_learnable_idx = tuple(torch.tensor(not_learnable_idx).T)

    def reset(self):
        self.state = self.starting_position
        #self.state = random.choice(range(self.n_cell))
        self.done = False
        self.expert_obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
        return self.state, None
    
    def step(self, action, state=None, multiple=False):
        done = False
        if state is None:
            real_step = True
            state = self.state
            self.state = new_state = tuple(self.P[self.state + (action, )])
        else:
            real_step = False
            if np.array(state).shape > np.array(self.starting_position).shape:
                new_state = [tuple(self.P[s + (a, )]) for s, a in zip(state, action)]
            else:
                new_state = tuple(self.P[state + (action, )])
        if not multiple:
            reward = self.R[action]
            if new_state[0] == self.goal_position[0]:
                done = True
                if new_state == self.goal_position:
                    reward = self.treasure

        else:
            done = torch.where(torch.any(torch.tensor(new_state) == torch.tensor(self.goal_position), dim=1), True, False)
            nongoal_reward = torch.where(action, torch.ones(len(new_state)), - torch.ones(len(new_state)))
            reward = torch.where(torch.all(torch.tensor(new_state) == torch.tensor(self.goal_position), dim=1), torch.ones(len(new_state)) * self.treasure, nongoal_reward)
        if real_step:
            self.done = done
        return new_state, reward, done, None
    
    def uniform_policy(self, unique_verbose=False, data_percentage=1):
        self.uniform_obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
        for r in range(self.n_cell[0] - 1):
            for c in range(self.n_cell[1]):
                for a in range(self.action_space.n):
                    done = False
                    s0 = (r, c)
                    s1 = tuple(self.P[s0 + (a, )])
                    reward = 1 if a else -1
                    if s1[0] == self.goal_position[0]:
                        done = True
                        if s1 == self.goal_position:
                            reward = self.treasure                 
                    self.uniform_obs.insert({'state0': s0, 'state1': s1, 'action': a, 'rewards': reward, 'done': done}, unique_verbose=unique_verbose)
        print('Buffer data numbers', len(self.uniform_obs._buffers['state0']))
                    
    
if __name__ == '__main__':
    T = 10
    env = DeepSea(depth=3)
    A = np.random.binomial(1, 0.5, size=T)
    s0 = env.reset()
    for t in range(T):
        a = A[t]
        s1, r, done, _ = env.step(a)
        print(s1)
        if done:
            print(f'Done in {t+1} steps')
            break
    env.uniform_policy()
    print(env.uniform_obs._buffers)
        