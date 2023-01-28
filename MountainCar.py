from copy import deepcopy
from model import Tabular
from gym.envs.classic_control import MountainCarEnv
import numpy as np

def uniform_grid(low, high, bins=(10,10)):
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
        return obs, rew, done, *info

    def set_state(self, target_state):
        self.env.state = target_state

    def get_state(self):
        return deepcopy(self.env.state)

    def virtual_step(self, state, action):
        env_copy = deepcopy(self.env)
        env_copy.set_state(state)
        return env_copy.step(action)


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


class Agent:
    def __init__(self, env, discrete=True, alpha=0.02, gamma=0.99, n_particle=10, seed=505):
        # Environment info
        self.env = env
        self.discrete = discrete
        self.state_grid = uniform_grid(self.env.observation_space.high, self.env.observation_space.low, bins=(10, 10))
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)  # n-dimensional state space
        self.action_size = self.env.action_space.n  # 1-dimensional discrete action space
        self.seed = np.random.seed(seed)
        print("Environment:", self.env)
        print("State space size:", self.state_size)
        print("Action space size:", self.action_size)
        
        # Learning parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        
        self.q_table = Tabular(self.state_size, self.action_size, n_particle, gamma=gamma, prior='normal')
        #np.zeros(shape=(self.state_size + (self.action_size,)))
        print("Q table size:", self.q_table.shape)

    def act(self, state):
        if self.discrete:
            state = self.discrete_state(state)
        return np.argmax(self.q_table[state])

    def discrete_state(self, sample_state, bins=(10, 10)):
        """Discretize a sample as per given grid.
    
        Parameters
        ----------
        sample : array_like
            A single sample from the (original) continuous space.
        grid : list of array_like
            A list of arrays containing split points for each dimension.
        
        Returns
        -------
        discretized_sample : array_like
            A sequence of integers with the same number of dimensions as sample.
        """
        return list(int(np.digitize(s, g)) for s, g in zip(sample_state, self.state_grid))  