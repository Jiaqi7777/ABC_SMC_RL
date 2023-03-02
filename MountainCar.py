from copy import deepcopy
from model import Tabular
from gym.envs.classic_control import MountainCarEnv
import numpy as np

def uniform_grid(low, high, bins=(10,10), verbose=False):
    """Define a uniformly-spaced grid that can be used to discretize a space.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins along each corresponding dimension.
    
    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    if verbose:
        print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
        for l, h, b, splits in zip(low, high, bins, grid):
            print("    [{}, {}] / {} => {}".format(l, h, b, splits))
    return grid

class MountainCar:
    def __init__(self):
        self.env = MountainCarEnv()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        obs, rew, done, *info = self.env.step(action)
        return obs, rew, done, info 

    def set_state(self, target_state):
        self.env.state = target_state

    def get_state(self):
        return self.env.state

    def virtual_step(self, state, action):
        env_copy = deepcopy(self.env)
        env_copy.set_state(state)
        return env_copy.step(action)

    def render(self):
        self.env.render(render_mode="rgb_array")

    #Tabular method
    def apply_policy(self, state, para):
        r_max = -np.inf
        for a in self.env.action_space.n:
            s1, _, _, _ = self.virtual_step(state, a)
            if para[s1] > r_max:
                r_max = para[s1]
                a_hat = a
        return a_hat, r_max

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()