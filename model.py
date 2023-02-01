import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from utils import plot_3d

def uniform_grid(low, high, bins=(10,10), include_low=1, verbose=False):
    """Define a uniformly-spaced grid that can be used to discretize a space.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins along each corresponding dimension.
    include_low: 0, 1
        1 not including the lowest boundary of the grid
        0 for including 
    
    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[include_low:-1] for dim in range(len(bins))]
    if verbose:
        print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
        for l, h, b, splits in zip(low, high, bins, grid):
            print("    [{}, {}] / {} => {}".format(l, h, b, splits))
    return grid

class Buffer:
    def __init__(self, entry_keys, seed=555):
        self._buffers = {key: [] for key in entry_keys}
        self.seed = random.seed(seed)

    def insert(self, items):
        if set(items.keys()) != set(self._buffers.keys()):
            raise IndexError

        for k, v in items.items():
            self._buffers[k].append(v)

    def get_minibatch(self, key, batch_size):
        if batch_size == 1:
            minibatch_indices = np.reshape(np.array(-1),(1,))
        else:
            minibatch_indices = random.choice(key, np.array(range(len(self))),
                                            shape=(batch_size,))
        return self.get(minibatch_indices)

    def get(self, indices):
        return {k: np.array([v[i] for i in indices])
                for k, v in self._buffers.items()}

    def __len__(self):
        return len(list(self._buffers.values())[0])

class Tabular:
    def __init__(self, env, n_particle, prior='normal', discrete=True, gamma=0.95, std=0.1, verbose=True):
        self.env = env
        self.discrete = discrete
        self.obs_size = env.observation_space.shape
        self.bins=(10,)*self.obs_size[0]
        self.state_grid = uniform_grid(high=self.env.observation_space.high, low=self.env.observation_space.low, bins=self.bins, verbose=verbose)
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)  # n-dimensional state space
        self.action_size = self.env.action_space.n  # 1-dimensional discrete action space
        if verbose:
            print("Environment:", self.env)
            print("State space size:", self.state_size)
            print("Action space size:", self.action_size)
        self.gamma = gamma
        self.std = std
        self.n_particle = n_particle
        self._weights = np.ones(n_particle) / n_particle
        if prior == 'normal':
            self.tables = np.random.normal(size=((n_particle,) + self.bins + (self.action_size,)))
            if verbose:
                print("Q table size:", self.tables[-1].shape)        
        else:
            raise NotImplementedError(f'The prior method corresponds to {prior} has not been implemented')

    def act(self, state, table):
        return np.argmax(table[state])

    def discrete_state(self, sample_state):
        """Discretize a sample as per given grid."""
        return tuple(int(np.digitize(s, g)) for s, g in zip(sample_state, self.state_grid))  

    def sample_para(self, weights):
        return random.choices(self.tables, weights)

    def q_value(self, table, s, a):            
        if len(table.shape) > len(s+ (a,)):
            return table[(slice(None), *(s+ (a,)))]
        return table[s][a]

    def v_value(self, table, s):
        if len(table.shape) > len(s) + 1:
            return np.max(table[(slice(None), *s)], axis = 1)
        return max(table[s])

    def r_hat(self, s0, s1, a):
        '''
        samples = Buffer(['a_hat', 'r_hat'])
        for p in table:
            a_hat, r_hat = self.act(state, p, env)
            samples.insert({'a_hat':a_hat, 'r_hat': r_hat})
        return samples
        '''
        return self.q_value(self.tables, s0, a) - self.gamma * self.v_value(self.tables, s1) 
        
    def lld(self, obs, sample):
        reward = obs['rewards'][-1]
        lld = stats.norm.pdf(reward, loc=sample, scale=self.std)
        return np.prod(lld)

    def get_parameter(self):
        return self.tables

    def set_parameter(self, new_para):
        self.tables = new_para

    def set_weights(self, new_weights):
        self._weights = new_weights

    def optimal_parameter(self):
        return self.get_parameter()[np.argmax(self._weights)]

    def plot_value(self, title='Value for each state', xlabel=None, ylabel=None, zlabel='Value'):
        para = np.max(self.optimal_parameter(), axis=-1)
        s0, s1 = uniform_grid(high=self.env.observation_space.high, low=self.env.observation_space.low, bins=self.bins, include_low=0)
        S0, S1 = np.meshgrid(s0, s1)
        plot_3d(S0, S1, para, title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

    def plot_policy(self, title='Policy for each state', xlabel=None, ylabel=None, zlabel='Policy'):
        para = np.argmax(self.optimal_parameter(), axis=-1)
        s0, s1 = uniform_grid(high=self.env.observation_space.high, low=self.env.observation_space.low, bins=self.bins, include_low=0)
        S0, S1 = np.meshgrid(s0, s1)
        plot_3d(S0, S1, para, title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

    def save(self, episode, file_path=''):
        if file_path == '':
            file_path=f'Models/MountainCar_P{self.n_particle}_B{self.bins[0]}_E{episode}'
        with open(f'{file_path}_tables.npy', 'wb') as f:
            np.save(f, self.tables)
        with open(f'{file_path}_weights.npy', 'wb') as f:
            np.save(f, self._weights)
        


