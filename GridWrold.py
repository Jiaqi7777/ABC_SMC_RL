import numpy as np
from gym import spaces
import random

class GridWorld:
    def __init__(self, n_cell, starting_position=(0,0), goal_position=(-1,-1)):
        '''
        actions: [1,-1]
        '''
        self.n_cell = n_cell
        self.starting_position = starting_position
        if goal_position == (-1, -1):
            goal_position = ( n_cell[0] - 1, n_cell[1] - 1 )
        self.goal_position = goal_position
        self.done = False
        self.observation_space = spaces.Discrete(n_cell[0] * n_cell[1])
        self.observation_space_high = (n_cell, 0)
        self.observation_space_low = (0, 0)
        self.action_space = spaces.Discrete(4)
        print('goal: ', goal_position)
        print('start: ', starting_position)
        print('length: ', n_cell)
        self.P = np.zeros((n_cell[0], n_cell[1], self.action_space.n))
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
                s_prime = (next_row, next_col) #gridworld[next_row, next_col]
                self.P[row, col, a] = s_prime

        self.R = np.full((n_cell[0], n_cell[1]), -1)
        self.R[goal_position] = 0

    def reset(self):
        self.state = self.starting_position
        #self.state = random.choice(range(self.n_cell))
        self.done = False
        return (self.state, ), None
    
    def step(self, action, state=None):
        done = False
        if state == None:
            new_state = self.P[state, action]
        else:
            self.state = new_state = self.P[self.state, action]
        if new_state == self.goal_position:
            done = True
            if state == None:
                self.done = done
        return new_state, self.R[new_state], done, None
        

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
