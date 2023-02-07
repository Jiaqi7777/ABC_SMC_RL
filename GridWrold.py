import numpy as np
from gym import spaces

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
        return (self.state, ), None

    def step(self, action):
        
        reward = 0
        self.state += action - 1
        self.state = max(0, self.state)
        self.state = min(self.state, self.n_cell-1)
        if self.state == self.goal_position:
            self.done = True
            reward = 1
       
        return (self.state, ), reward, self.done, None
