import numpy as np
from gym import spaces
import random

class GridWorld:
    def __init__(self, n_cell, starting_position, goal_position):
        '''
        actions: [1,-1]
        '''
        self.n_cell = n_cell
        self.starting_position = starting_position
        self.goal_position = goal_position
        self.done = False
        self.observation_shape = (1, )
        self.observation_space = spaces.Discrete(n_cell,)
        self.observation_space_high = (n_cell, 0)
        self.observation_space_low = (0, 0)
        self.action_space = spaces.Discrete(3,)
        print('goal: ', goal_position)
        print('start: ', starting_position)
        print('length: ', n_cell)

    def reset(self):
        self.state = self.starting_position
        #self.state = random.choice(range(self.n_cell))
        self.done = False
        return (self.state, ), None

    def step(self, action, state=None):
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
