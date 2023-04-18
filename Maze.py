import numpy as np
from gym import spaces
class Maze:
    def __init__(self, action_size=2, depth=5):
        self.env_name = 'Maze'
        self.action_size = action_size
        self.starting_position = (0, )
        self.goal_position = (depth - 1, )
        self.n_cell = (depth, )
        if action_size == 2:
            self.R = np.array([[-3, -1] for i in range(depth)])
        else:
            return NotImplementedError
        self.done = False
        self.observation_space = spaces.Discrete(depth)
        self.observation_space_high = (depth,)
        self.observation_space_low = (0,)
        self.action_space = spaces.Discrete(action_size)
        self.names = [f'{i,a}' for i in range(self.n_cell[0]) for a in range(self.action_size)]

        
    def reset(self):
        self.state  = self.starting_position
        self.done = False
        return self.state, self.done
        
    def step(self, action, state=None):
        done = False
        real_step = False
        if state == None:
            state = self.state
            real_step = True
        new_state = (min(state[0] + 1, self.goal_position[0]), )
        if new_state == self.goal_position:
            done = True
        if real_step:
            self.done = done
            self.state = new_state
        return  new_state, self.R[new_state + (action, )], done, None
        
if __name__ == '__main__':
    import random
    env = Maze()
    s0, _ = env.reset()
    for i in range(11):
        s1, r, done, _ = env.step(random.randint(0, env.action_size - 1))
        print(s1, r, done)
        if done:
            print('done with steps:', i)
            break
    