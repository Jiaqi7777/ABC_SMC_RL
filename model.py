import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
import torch
from utils import *
from scipy.sparse import csr_matrix, coo_array
from parameter import *

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
    if len(bins) == 1:
        return grid, [0]
    return grid

class Buffer:
    def __init__(self, entry_keys, seed=SEED):
        self.entry_keys = entry_keys
        self._buffers = {key: [] for key in entry_keys}
        self.unique_set = []
        self._unique_buffers = {key: [] for key in entry_keys}
        self._new_data_buffers = {key: [] for key in entry_keys}
        
    def init_new_data_buffer(self):
        for k, v in self._new_data_buffers.items():
            for vi in v:
                self._unique_buffers[k].append(vi)
                self._buffers[k].append(vi)
        self._new_data_buffers = {key: [] for key in self.entry_keys}

    def insert(self, items, unique=False, unique_verbose=True, update_new_data=False):
        if set(items.keys()) != set(self._buffers.keys()):
            raise IndexError
        unique_check = list(items.values())
        unique_check.remove(items['state1'])
        unique_check = str(unique_check)
        if unique:
            if unique_check in self.unique_set:
                return False
            self.unique_set.append(unique_check)
            for k, v in items.items():
                if update_new_data:
                    self._new_data_buffers[k].append(v) 
                else:   
                    self._unique_buffers[k].append(v)
                    self._buffers[k].append(v)
            if unique_verbose:
                print('New items added', unique_check)
            return True
        else:
            if unique_check not in self.unique_set:
                for k, v in items.items():
                    self._unique_buffers[k].append(v)
                if unique_verbose:
                    print('New items explored', unique_check)
            if update_new_data:
                for k, v in items.items():
                    self._new_data_buffers[k].append(v) 
            else:   
                for k, v in items.items():
                    self._buffers[k].append(v)
            return True
        # if unique_check in self.unique_set:
        #     if unique:
        #         return False
        # else:
        #     self.unique_set.append(unique_check)
        #     for k, v in items.items():
        #         if update_new_data:
        #             self._new_data_buffers[k].append(v)  
        #         else:   
        #             self._unique_buffers[k].append(v)
        #     if unique_verbose:
        #         print('New items added', unique_check)
        #     # if len(self._buffers[k]) > BUFFER_SIZE:
        #     #     self._buffers[k].pop(0)
        # for k, v in items.items():
        #     self._buffers[k].append(v)
        # return True
        
    def get_minibatch(self, batch_size):
        if batch_size == 1:
            minibatch_indices = np.reshape(np.array(-1),(1,))
        else:
            minibatch_indices = random.choice(np.array(range(len(self))),
                                            shape=(batch_size,))
        return self.get(minibatch_indices)

    def get(self, indices):
        return {k: np.array([v[i] for i in indices])
                for k, v in self._buffers.items()}

    def __len__(self):
        return len(list(self._buffers.values())[0])

class Tabular:
    def __init__(self, env, n_particle, prior='normal', discrete=False, gamma=0.95, std=0.1, mean=0, verbose=True, bins=(10,), initial_tables=None, initial_weights=None, idx=None):
        self.env = env
        self.discrete = discrete
        self.obs_size = env.observation_space.shape
        self.idx = idx
        self.initial_tables = initial_tables
        if discrete:
            self.bins = bins*self.obs_size[0]
            self.state_grid = uniform_grid(high=self.env.observation_space.high, low=self.env.observation_space.low, bins=self.bins, verbose=verbose)
            self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)  # n-dimensional state space
        else:
            self.state_size = self.bins = env.n_cell
        self.action_size = self.env.action_space.n  # 1-dimensional discrete action space 
        if verbose:
                print("Environment:", self.env)
                print("State space size:", self.bins)
                print("Action space size:", self.action_size)
        
        self.gamma = gamma
        self.mean = mean
        self.std = std
        self.n_particle = n_particle if (initial_tables is None) else len(initial_tables)
        self._weights = initial_weights
        if prior == 'normal':
            self.random_tables = torch.normal(mean=mean, std=std, size=((n_particle,) + self.bins + (self.action_size,)))
            self.tables = initial_tables if (initial_tables is not None) else self.random_tables
            # self.tables = torch.tensor(np.repeat(initial_tables[np.newaxis, ...], n_particle, axis=0), dtype=torch.float32) if (initial_tables is not None) else self.random_tables
            #diagnosis
            # self.tables = torch.load('../Samples/MCMC/094980/T200_Repeat1_StoFalse_M1200_GdyFalse_Sigma4.pt')[-1][27]
            if idx is not None:
                # last_samples = torch.load('2DT1050000HMC.pt')[-1]
                for i in idx:
                    self.tables[(slice(None), *i)] = torch.minimum(torch.normal(mean=mean, std=std, size=(n_particle, )), torch.zeros(n_particle))
            if env.goal_idx:
                for n in range(n_particle):
                    self.tables[n][env.goal_idx] = 0
            if env.not_learnable_idx:
                for n in range(n_particle):
                    self.tables[n][env.not_learnable_idx] = 0
            if verbose:
                print("Q table size:", self.tables[-1].shape)     
        else:
            raise NotImplementedError(f'The prior method corresponds to {prior} has not been implemented')

    def act(self, state, table=[], weights=[], greedy=False, greedy_epsilon=GREEDY_EPSILON):
        if table == []:
            return self.act(state, self.tables, weights=self._weights)
        elif table[0].shape == self.tables[0].shape:
            tables = table
            if weights == []:
                weights = torch.ones(len(tables)) / len(tables)
            '''Thompson Sampling'''
            # action_idx = np.argmax(tables[(slice(None), *state)], axis=-1)
            # thompson_matrix = csr_matrix((np.ones(len(tables)), (action_idx, np.array(range(len(tables))))), shape=(self.action_size, len(tables)))
            # thompson_weights = thompson_matrix @ weights
            # return np.argmax(thompson_weights)
            # return random.choices(range(self.action_size), thompson_weights)[0]
            if greedy:
                if np.random.uniform(0, 1) > greedy_epsilon:
                    table = torch.mean(tables, 0)
                else: 
                    return torch.randint(0, self.action_size, (1,))[0]
            else:
                table = random.choice(tables)
            return torch.argmax(table[state])
        '''Greedy Action'''
        return torch.argmax(table[state]) if np.random.uniform(0, 1) > greedy_epsilon else torch.randint(0, self.action_size, (1,))[0]

    def discrete_state(self, sample_state):
        """Discretize a sample as per given grid."""
        return tuple(int(np.digitize(s, g)) for s, g in zip(sample_state, self.state_grid))  

    def sample_para(self, weights=None, burn_in=0):
        # i = random.choices(range(len(self.tables)), weights)
        # print('Best table', i)
        if weights is None:
            return random.choice( self.tables[ burn_in : ] )
        return random.choices (self.tables[ burn_in : ], weights )
    
    # def q_value_with_linear_index(self, para, s, a):
        

    def q_value(self, table, s, a, full=True): 
        if len(s) == 0:
            return torch.tensor([])
        if not full:
            table = self.fill_learnable_table(table)
        if hasattr(a, "__len__"):# multiple s, mutiple a 
            if len(table.shape) > len(s) + len(a):
                return table[(slice(None), *s, a)]
        elif len(table.shape) > len(s+ (a,)):# multiple s, single a 
            return table[(slice(None), *(s+ (a,)))]
        if hasattr(s[0], "__len__"):
            return table[tuple(s) + (a, )]
        return table[s][a]
    
    def fill_learnable_table(self, table):
        # table = table.reshape(self.env.learnable_shape + (self.action_size, ))
        extend_table = torch.tensor(self.tables[0], dtype=torch.float32).clone().detach()
        try:
            extend_table[self.env.learnable_idx] = table
        except:
            table = table.reshape(extend_table[self.env.learnable_idx].shape)
            extend_table[self.env.learnable_idx] = table
        return extend_table
    
    def v_value(self, table, s, full=True):
        if len(s) == 0:
            return torch.tensor([])
        if not full:
            table = self.fill_learnable_table(table)
        if len(table.shape) > len(s) + 1:
            return torch.max(table[(slice(None), *s)], 1).values
        if hasattr(s[0], "__len__"):
            return torch.max(table[tuple(s)], -1).values
        return torch.max(table[s])

    def r_hat(self, s0, s1, a):
        a = int(a)
        '''
        samples = Buffer(['a_hat', 'r_hat'])
        for p in table:
            a_hat, r_hat = self.act(state, p, env)
            samples.insert({'a_hat':a_hat, 'r_hat': r_hat})
        return samples
        '''
        # print(self.v_value(self.tables, s1))
        return self.q_value(self.tables, s0, a) - self.gamma * self.v_value(self.tables, s1).values 
        
    def lld(self, obs, sample):
        reward = obs['rewards'][-1]
        lld = stats.norm.pdf(reward, loc=sample, scale=self.std)
        return np.prod(lld)
    
    def sample_random_tables(self, set=False):
        self.random_tables = torch.normal(mean=self.mean, std=self.std, size=((self.n_particle,) + self.bins + (self.action_size,)))
        if set:
            self.tables = self.random_tables        

    def set_learnable_idx(self, obs):
        index_arrays = []
        all_obs = np.array(list(obs._unique_buffers.values())).T
        for s0, _, a, _, _ in list(all_obs):
            index_arrays.append(np.array(s0 + (a,)))
        self.env.learnable_idx = tuple(torch.tensor(index_arrays).T)
        self.env.names = index_arrays

    def get_parameter(self):
        return self.tables
    
    def get_learnable_parameter(self):
        return self.tables[(slice(None),  *self.env.learnable_idx)]

    def set_parameter(self, new_para, idx=None):
        if idx is None:
            self.tables = new_para
        else:
            for i in idx:
                self.tables[(slice(None), *i)] = new_para[(slice(None), *i)]

    def set_learnable_parameter(self, new_para):
        # for i in self.env.learnable_idx:
        #     self.tables[(slice(None), *i)] = new_para[(slice(None), *i)]
        try:
            self.tables[(slice(None),  *self.env.learnable_idx) ] = new_para
        except:
            shape = self.tables[(slice(None),  *self.env.learnable_idx)].shape
            new_para = new_para.reshape(shape)
            self.tables[(slice(None),  *self.env.learnable_idx) ] = new_para
    
    def set_weights(self, new_weights, log=False):
        if log:
            self._weights = torch.exp(new_weights)
        else:
            self._weights = new_weights

    def optimal_parameter(self):
        return self.get_parameter()[np.argmax(self._weights)]
    
    def policy(self, paras=[]):
        if paras==[]:
            paras = self.tables
            n = self.n_particle
            weights = self._weights
        else:
            n = len(paras)
            weights = np.ones(n) / n
        idx = np.argmax(paras, axis=-1).T.reshape(-1)
        thp_matrix = coo_array((np.ones(n * np.prod(self.state_size)), (idx, np.array(range(n * np.prod(self.state_size))))), shape=(self.action_size, n * np.prod(self.state_size)))
        thp_matrix = thp_matrix.toarray().reshape(thp_matrix.shape[0],-1, n).swapaxes(0,1).reshape(-1, n)
        thp_vec = thp_matrix @ weights
        thp_weights = np.moveaxis(thp_vec.reshape(self.state_size[::-1]+ (self.action_size,)),range(len(self.state_size)),range(len(self.state_size))[::-1])
        return thp_weights#np.argmax(thp_weights, axis=-1)
        

    def plot_value(self, title='Value for each state', xlabel=None, ylabel=None, zlabel='Value', show=False, paras=[]):
        weights = np.ones(len(paras))
        if paras == []:
            paras = self.tables
            weights=self._weights
        para = np.average(np.max(paras, axis=-1), weights=weights, axis=0)
        if self.discrete:
            s0, s1 = uniform_grid(high=self.env.observation_space.high, low=self.env.observation_space.low, bins=self.bins, include_low=0)
        else:
            s0, s1 = uniform_grid(high=self.env.observation_space_high, low=self.env.observation_space_low, bins=self.bins, include_low=0)
            s0 = np.array(s0).astype('int32')
            s1 = np.array(s1).astype('int32')
        S0, S1 = np.meshgrid(s0, s1)
        # print('Value', '\n', np.round(para, 1))
        # plot_3d(S0, S1, para, title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

    def plot_policy(self, paras=[], title='Policy for each state', xlabel=None, ylabel=None, zlabel='Policy', show=False, additional_info=[], save=False):
        # for s in range(self.state_size):
        para = self.policy(paras=paras)
        if self.discrete:
            s0, s1 = uniform_grid(high=self.env.observation_space.high, low=self.env.observation_space.low, bins=self.bins, include_low=0)
        else:
            s0, s1 = uniform_grid(high=self.env.observation_space_high, low=self.env.observation_space_low, bins=self.bins, include_low=0)
        
        S0, S1 = np.meshgrid(s0, s1)
        # print('policy:', '\n',  para)
        para = para
        plot_2d(s0, s1, para, env_name=self.env.env_name, action_dim=self.action_size, title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, show=show, additional_info=additional_info, save=save)
        # plot_3d(S0, S1, para.T, title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, show=show)

    def save(self, episode, file_path='', HORIZON=200, ENV_NAME=''):
        if file_path == '':
            file_path=f'Models/{ENV_NAME}_H{HORIZON}_P{self.n_particle}_B{self.bins[0]}_E{episode}'
        with open(f'{file_path}_tables.npy', 'wb') as f:
            np.save(f, self.tables)
        with open(f'{file_path}_weights.npy', 'wb') as f:
            np.save(f, self._weights)
        


