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

class GridWorld:
    def __init__(self, n_cell, starting_position=(0,0), goal_position=(-1,-1), obstacles=False, stochastic=False):
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
        self.stochastic = stochastic
        if self.stochastic:
            self.move_prob = [CORRECT_MOVE_PROB] + [(1 - CORRECT_MOVE_PROB)/5] * 5
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
                    range(self.action_space.n),
                    [(-1, 0), (0, 1), (1, 0), (0, -1)]
                    ):
                next_row = max(0, min(row + d[0], n_cell[0]-1))
                next_col = max(0, min(col + d[1], n_cell[1]-1))
                s_prime = [next_row, next_col] #gridworld[next_row, next_col]
                self.P[row, col, a] = s_prime
                self.P[goal_position + (a, )] = goal_position
                    
        if self.stochastic:
            random_move = np.array([0, -1, 1, 2]) #moving forwards, left, right, back when facing the correct direction
            self.stochasticP = np.zeros((n_cell[0], n_cell[1], self.action_space.n, len(self.move_prob), 2), dtype='int32')
            for s in gridworld.flat:
                row, col = np.argwhere(gridworld == s)[0]
                for a, d in zip(
                        range(self.action_space.n),
                        [(-1, 0), (0, 1), (1, 0), (0, -1)]
                        ):
                    next_next_row = max(0, min(row + 2 * d[0], n_cell[0]-1))
                    next_next_col = max(0, min(col + 2 * d[1], n_cell[1]-1))
                    self.stochasticP[row, col, a, :4] = self.P[row, col, (a + random_move) % 4] 
                    self.stochasticP[row, col, a, -2] = [row, col] #stay still
                    self.stochasticP[row, col, a, -1] = [next_next_row, next_next_col]
                    self.stochasticP[(*goal_position, a, slice(None))] = [goal_position] * len(self.move_prob)
                
        self.R = np.full((n_cell[0], n_cell[1]), - 0.1)
        if obstacles:
            n_obs = np.prod(n_cell) // 2
            self.R[random.choices(range(0, n_cell[0]), k=n_obs), random.choices(range(0, n_cell[1]), k=n_obs)] = - 1
            # self.R[:, range(1, n_cell[1], 4)] = -2
            # self.R[range(1, n_cell[0], 5), :]  = -2
            # self.R[n_cell[0]//2, n_cell[1]//2] = -2
        self.R[starting_position] = - 0.1
        self.R[goal_position] = 0
        self.names = [f'{i,j,a}' for i in range(n_cell[1]) for j in range(n_cell[0]) for a in range(4)]

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
            if self.stochastic:
                self.state = new_state = tuple(random.choices(self.stochasticP[self.state + (action, )], self.move_prob)[0])
            else:
                self.state = new_state = tuple(self.P[self.state + (action, )])
        else:
            real_step = False
            if self.stochastic:
                if multiple:
                    if np.array(state).shape > np.array(self.starting_position).shape:
                        new_state = [random.choices(self.stochasticP[tuple(s) + (a, )], self.move_prob, k=multiple) for s, a in zip(state, action)]
                    else:
                        new_state = random.choices(self.stochasticP[state + (action, )], self.move_prob, k=multiple)
                    return np.array(new_state), self.R[tuple(map(tuple, np.array(state).T))]
                else:
                    new_state = tuple(random.choices(self.stochasticP[state + (action, )], self.move_prob)[0])
            else:
                if np.array(state).shape > np.array(self.starting_position).shape:
                    new_state = [tuple(self.P[s + (a, )]) for s, a in zip(state, action)]
                else:
                    new_state = tuple(self.P[state + (action, )])
        if not multiple:
            if new_state == self.goal_position:
                done = True
        else:
            done = torch.where(torch.all(torch.tensor(new_state) == torch.tensor(self.goal_position), dim=1), True, False)
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
      
    def uniform_policy(self, unique_verbose=False, data_percentage=1):
        self.uniform_obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
        if False:#self.stochastic:
            for r in range(self.n_cell[0]):
                for c in range(self.n_cell[1]):
                    for a in range(self.action_space.n):
                        for i in range(len(self.move_prob)):
                            done = False
                            s0 = (r, c)
                            s1 = tuple(self.stochasticP[s0 + (a, )][i])
                            if s0 == self.goal_position:
                                done = True
                            self.uniform_obs.insert({'state0': s0, 'state1': s1, 'action': a, 'rewards': self.R[s0], 'done': done}, unique=True, unique_verbose=unique_verbose)
            print('Unique data numbers', len(self.uniform_obs._buffers['state0']))
        else:
            for r in range(self.n_cell[0]):
                for c in range(self.n_cell[1]):
                    for a in range(self.action_space.n):
                        if data_percentage < 1 and np.random.uniform(0, 1) > data_percentage:
                            continue
                        done = False
                        s0 = (r, c)
                        s1 = tuple(self.P[s0 + (a, )])
                        if s0 == self.goal_position:
                            done = True
                        self.uniform_obs.insert({'state0': s0, 'state1': s1, 'action': a, 'rewards': self.R[s0], 'done': done}, unique_verbose=unique_verbose)
            print('Unique data numbers', len(self.uniform_obs._buffers['state0']))
                    

if __name__ == '__main__':
    for seed in range(555, 557):
        random.seed(seed)
        env = GridWorld((3,4), (0,0), obstacles=True, stochastic=False)
        print(env.reset(), seed)
        print(env.observation_space.n)
        # env.plot_env()
        for t in range(10):
            print(t)
            env.step(random.randint(0,3))
    env.uniform_policy(unique_verbose=True)
    print(env.step(action=0, state = (1,2), multiple=3))
    # env.plot_env(env.expert_traj + env.R)
    # print(env.expert_obs._buffers['state1'])
    # env = GridWorld((3,4), (1,2), (2,3), obstacles=True)
    # assert(env.step(1, (2,2)) == ((2,3), 0, True, None))
