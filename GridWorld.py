import numpy as np
from gym import spaces
import random
from matplotlib import colors, colormaps
import matplotlib.pyplot as plt
'''module import'''
from model import *

class GridWorld:
    def __init__(self, n_cell, starting_position=(0,0), goal_position=(-1,-1), obstacles=False):
        self.env_name = 'GridWorld'
        self.n_cell = n_cell
        self.starting_position = starting_position
        if goal_position == (-1, -1):
            goal_position = ( n_cell[0] - 1, n_cell[1] - 1 )
        self.goal_position = goal_position
        self.done = False
        self.observation_space = spaces.Discrete(n_cell[0] * n_cell[1])
        self.observation_space_high = (n_cell[0], n_cell[1])
        self.observation_space_low = (0, 0)
        self.action_space = spaces.Discrete(4)
        print('goal: ', goal_position)
        print('start: ', starting_position)
        print('World Scale: ', f'{n_cell[0]}x{n_cell[1]}')
        self.P = np.zeros((n_cell[0], n_cell[1], self.action_space.n, 2), dtype='int32')
        # any action taken in terminal state has no effect
        gridworld = np.arange(
                self.observation_space.n
                ).reshape((n_cell[0], n_cell[1]))
        # gridworld[goal_position] = 0
        for s in gridworld.flat:
            row, col = np.argwhere(gridworld == s)[0]
            for a, d in zip(
                    range(self.action_space.n),
                    [(-1, 0), (0, 1), (1, 0), (0, -1)]
                    ):
                next_row = max(0, min(row + d[0], n_cell[0]-1))
                next_col = max(0, min(col + d[1], n_cell[1]-1))
                s_prime = [next_row, next_col] #gridworld[next_row, next_col]
                self.P[row, col, a] = s_prime
                self.P[goal_position + (a, )] = goal_position
                
        self.R = np.full((n_cell[0], n_cell[1]), -1)
        if obstacles:
            n_obs = np.prod(n_cell) // 2
            self.R[random.choices(range(0, n_cell[0]), k=n_obs), random.choices(range(0, n_cell[1]), k=n_obs)] = -20
            # self.R[:, range(1, n_cell[1], 4)] = -2
            # self.R[range(1, n_cell[0], 5), :]  = -2
            # self.R[n_cell[0]//2, n_cell[1]//2] = -2
        self.R[starting_position] = -1
        self.R[goal_position] = 0
        self.names = [f'{i,j,a}' for i in range(n_cell[1]) for j in range(n_cell[0]) for a in range(4)]

    def reset(self):
        self.state = self.starting_position
        #self.state = random.choice(range(self.n_cell))
        self.done = False
        self.expert_obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
        return self.state, None
    
    def step(self, action, state=None):
        done = False
        if state == None:
            real_step = True
            state = self.state
            self.state = new_state = tuple(self.P[self.state + (action, )])
        else:
            real_step = False
            new_state = tuple(self.P[state + (action, )])
        if new_state == self.goal_position:
            done = True
            if real_step:
                self.done = done
        return new_state, self.R[state], done, None
        

    def oneD_step(self, action, state=None):
        done = False
        reward = -1
        if state == None:
            new_state = self.state + action - 1
        else:
            new_state = state + action - 1
        new_state = max(0, new_state)
        new_state = min(new_state, self.n_cell-1)
        if new_state == self.goal_position:
            done = True
            reward = 0
        if state == None:
            self.state = new_state
            self.done = done
        return (new_state, ), reward, done, None
    
    def plot_env(self, value=[]):
        # print(self.R)
        # norm = colors.BoundaryNorm(bounds, cmap.N)
        value = self.R if value == [] else value
        fig, ax = plt.subplots()
        ax.imshow(value, cmap=colormaps['pink'])

        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-.', color='k', linewidth=0.2)
        ax.set_xticks(np.arange(0, self.n_cell[1], 1))
        ax.set_yticks(np.arange(0, self.n_cell[0], 1))
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        plt.show()

    def expert(self, reset=True):
        if reset:
            self.expert_obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
        self.expert_traj = np.zeros(shape=self.R.shape)
        s0 = self.starting_position
        self.expert_traj[s0] = 10
        done  = False
        epsilon = 0.2
        while done is False:
            distance = list(map(lambda i, j: i - j, self.goal_position, s0))
            if np.random.uniform(0, 1)< 0.5:
                action = 1 if distance[1] > 0 else 3                       
            else:
                action = 2 if distance[0] > 0 else 0
            if np.random.uniform(0, 1) < epsilon:
                action = (action + 2) % 4
            s1, r, done, _ = self.step(action, s0)
            if self.R[s1] < -1:
                if action % 2 == 0:
                    action = random.choice([1, 3])
                else:
                    action = random.choice([0, 2])
                s1, r, done, _ = self.step(action, s0)
            self.expert_traj[s1] = 5
            # print(distance, s1, action)
            self.expert_obs.insert({'state0': s0, 'state1': s1, 'action': action, 'rewards': r, 'done': done})
            s0 = s1
        self.expert_traj[s1] = 15
        
    def plot_env_with_R(self):
        self.plot_env(self.expert_traj + self.R)         
      
    def uniform_policy(self):
        self.uniform_obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
        for r in range(self.n_cell[0]):
            for c in range(self.n_cell[1]):
                for a in range(self.action_space.n):
                    done = False
                    s0 = (r, c)
                    s1 = tuple(self.P[s0 + (a, )])
                    if s1 == self.goal_position:
                        done = True
                    self.uniform_obs.insert({'state0': s0, 'state1': s1, 'action': a, 'rewards': self.R[s1], 'done': done})
                    

if __name__ == '__main__':
    for seed in range(555, 557):
        random.seed(seed)
        env = GridWorld((3,4), (0,0), obstacles=True)
        print(env.reset(), seed)
        print(env.observation_space.n)
        env.plot_env()
    # env.expert()
    # env.plot_env(env.expert_traj + env.R)
    # print(env.expert_obs._buffers['state1'])
    # env = GridWorld((3,4), (1,2), (2,3), obstacles=True)
    # assert(env.step(1, (2,2)) == ((2,3), 0, True, None))
