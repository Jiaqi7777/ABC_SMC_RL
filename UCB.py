import numpy as np
import math
from parameter import *

class UCB_EpisodicRL:
    def __init__(self, env, horizon=HORIZON, c=1.0, alpha=0.2, gamma=1):
        self.horizon = horizon
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.state_n = env.n_cell
        self.action_n = env.action_space.n
        self.observation_n = env.observation_space.n
        self.reset()
        
    def reset(self):
        self.Q = np.zeros(shape=(self.state_n + (self.action_n, ))) / self.observation_n / self.action_n  # Value estimates
        self.N = np.zeros(shape=(self.state_n + (self.action_n, ))) / self.observation_n / self.action_n  # Action counts
        
    def select_action(self, state):
        # UCB action selection
        total_counts = np.sum(self.N[state])
        ucb_values = self.Q[state] + self.c * np.sqrt(np.log(total_counts + 1) / (self.N[state] + 1e-5))
        return np.random.choice(np.where(ucb_values == ucb_values.max())[0])
    
    def update(self, s0, s1, action, reward):
        # Update value estimates and counts
        self.N[s0 + (action,)] += 1
        td_error = reward + self.gamma * np.max(self.Q[s1]) - self.Q[s0 + (action,)]
        self.Q[s0 + (action,)] += self.alpha * td_error
        
class UCRL2Agent:
    def __init__(self, env, horizon=HORIZON, c=1.0, alpha=0.2, gamma=1, delta=0.1, tol=1e-5):
        self.horizon = horizon
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.state_n = env.n_cell
        self.action_n = env.action_space.n
        self.observation_n = env.observation_space.n
        self.delta = delta / 10  # Use 10 times smaller confidence interval
        self.tol = tol
        self.reset()
        
    def reset(self):
        self.Q = np.zeros(shape=(self.state_n + (self.action_n, ))) / self.observation_n / self.action_n  # Value estimates
        self.N = np.zeros(shape=(self.state_n + (self.action_n, ))) / self.observation_n / self.action_n  # Action counts
        self.R_hat = env.R  # Estimated rewards
        self.P_hat = np.zeros(shape=(self.state_n + (self.action_n, )))  # Estimated transition probabilities
        self.policy = np.zeros(shape=(self.state_n), dtype=int)  # Current policy
        self.V = np.zeros(shape=(self.state_n))  # Value function
        self.obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])

    def confidence_interval(self, n):
        return np.sqrt(2 * np.log(1 / self.delta) / n) if n > 0 else float('inf')
    
    def evi(self):
        V = np.zeros(shape=(self.state_n))
        while True:
            V_new = np.zeros_like(V)
            for s in S[::-1]:
                max_value = -float('inf')
                for a in range(self.action_n):
                    if self.N[s + (a,)] > 0:
                        ci = self.confidence_interval(self.N[s + (a,)])
                        optimistic_reward = min(1, self.R_hat[s + (a,)] + ci)
                        optimistic_transitions = np.clip(self.P_hat[s + (a,)] + ci, 0, 1)
                        optimistic_transitions /= np.sum(optimistic_transitions)  # Ensure probabilities sum to 1
                        expected_value = optimistic_reward + self.gamma * np.sum(optimistic_transitions * V)
                        if expected_value > max_value:
                            max_value = expected_value
                V_new[s] = max_value
            
            if np.max(np.abs(V_new - V)) < self.tol:
                break
            V = V_new
        
        # Extract policy
        for s in range(self.num_states):
            max_value = -float('inf')
            best_action = 0
            for a in range(self.num_actions):
                if self.N[s + (a,)] > 0:
                    ci = self.confidence_interval(self.N[s + (a,)])
                    optimistic_reward = min(1, self.R_hat[s + (a,)] + ci)
                    optimistic_transitions = np.clip(self.P_hat[s + (a,)] + ci, 0, 1)
                    optimistic_transitions /= np.sum(optimistic_transitions)
                    expected_value = optimistic_reward + self.gamma * np.sum(optimistic_transitions * V)
                    if expected_value > max_value:
                        max_value = expected_value
                        best_action = a
            self.policy[s] = best_action
    
    def update_estimates(self, state, action, reward, next_state):
        self.N[state + (action,)] += 1
        n = self.N[state + (action,)]
        self.R_hat[state + (action,)] = (self.R_hat[state + (action,)] * (n - 1) + reward) / n
        self.P_hat[state + (action,), next_state] += 1
        self.P_hat[state + (action,)] /= np.sum(self.P_hat[state + (action,)])
    
    def run_episode(self, env, initial_state):
        s0 = env.starting_position
        while True:
            action = self.policy[s0]
            print('action', action)
            s1, r, done, *info = env.step(action)
            self.obs.insert({'state0': s0, 'state1': s1, 'action': int(action), 'rewards': r, 'done': done}, unique=UNIQUE_OBS, update_new_data=True)
            self.update_estimates(s0, action, r, s1)
            if done:
                break
            s0 = s1
    
    def train(self, env, num_episodes, S):
        for episode in range(num_episodes):
            self.evi(S)
            for _ in range(10):  # Repeat each episode's trajectory 10 times
                initial_state = env.reset()
                self.run_episode(env, initial_state)

        
                
if __name__ == '__main__':
    # from tqdm import tqdm
    from tqdm import tqdm
    import datetime
    import argparse
    import os
    import sys
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.path.append('/scratch/Rabbit/work/ABC_SMC_RL/')
    sys.path.append('/Users/guojiaqi/work/ABC_SMC_RL/')
    '''module import'''
    from Environment.GridWorld import *
    from Environment.Maze import *
    from Environment.DeepSea import *
    from QLearning import *
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--training_step', default=EG_T, type=int)
    parser.add_argument('-t', '--time', default=datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument('-s', '--save', default=SAVE)
    parser.add_argument('-p', '--show', default=SHOW)
    parser.add_argument('--seed', default=SEED, type=int)
    parser.add_argument('-c', '--c_ucb', default=1, type=float)
    parser.add_argument('--Env', default=ENV_NAME)
    args = parser.parse_args()
    time = args.time
    save = args.save
    show = args.show
    seed = args.seed
    c = args.c_ucb
    training_steps = args.training_step
    env_name = args.Env
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dircty=''

    if env_name == 'GridWorld':
        env = GridWorld((1, 2), obstacles=False, stochastic=STOCHASTIC)
        # env.plot_env()
    if env_name == 'Maze':
        env = Maze()
    if env_name == 'DeepSea':
        env = DeepSea(depth=20)
        EPISODES = 3000000#env.n_cell[0] * 100
    if save:
        dircty_top = f'../UCB/{env.n_cell[0]}/c{c}/{time}/'
        if not os.path.exists(dircty_top):
            os.makedirs(dircty_top)
    
    S = []
    if len(env.n_cell) == 1:
        S = [(i, ) for i in range(env.n_cell[0])]
    else:
        if env_name == 'DeepSea':
            for i in range(env.n_cell[0] - 1):
                for j in range(env.n_cell[1]):
                    S.append((i, j)) 
        else:
            for i in range(env.n_cell[0]):
                for j in range(env.n_cell[1]):
                    S.append((i, j)) 
    Q_dp = np.zeros(shape=(env.n_cell + (env.action_space.n, ))) / env.observation_space.n / env.action_space.n
    A = range(env.action_space.n)
    # pi_star, Q_star, V_star = OfflineQLearning(Q, A, S, env, gamma=GAMMA, show=show, thresh=1e-3, alpha=1)
    pi_star, Q_star, V_star = DynamicProgramming(Q_dp, A, S, env, gamma=GAMMA, show=show)
    dim = env.observation_space.n * env.action_space.n
    # Q = np.random.normal(loc=PRIOR_MEAN, scale=PRIOR_SIGMA, size=(env.n_cell + (env.action_space.n, )))
    # if env.not_learnable_idx:
    #     Q[env.not_learnable_idx] = 0
    
    env.reset()
    results = []
    r_all_repeat = []
    Q_all_repeat = []
    N_all_repeat = []
    samples_all_repeat = []
    smc_all_repeat = []
    dircty = ''
    for repeat in range(REPEAT_EXPERIMENT_NAIVE):
        agent = UCB_EpisodicRL(env=env, c=c)
        if save:
            dircty = f'{dircty_top}R{repeat}/'
            if not os.path.exists(dircty):
                os.makedirs(dircty)

        STEPSIZE = INITIAL_STEPSIZE
        samples_all_ep = []
        obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
        s0, _ = env.reset()
        r_all_epi=[]
        for e in range(EPISODES):
            s0, _ = env.reset()
            # plot_qtable(para, title=f'Sampled Q table for episode {e} repeat {repeat}', save=save, figure_path=dircty)
            # print(f'Episode {e} in repeat {repeat}')
            R = 0
            # h = 0
            # while True: #Turn on h += 1
            new_data_flag = False
            for h in range(HORIZON):
                action = agent.select_action(s0)
                s1, r, done, *info = env.step(action)
                agent.update(s0, s1, action, r)
                R += r
                new_data_flag = (obs.insert({'state0': s0, 'state1': s1, 'action': int(action), 'rewards': r, 'done': done}, unique=UNIQUE_OBS, update_new_data=True) or new_data_flag)
                s0 = s1
                if done:
                    if new_data_flag:
                        print(f'Episode {e} in repeat {repeat}')
                        plot_obs(obs, env, env_name=ENV_NAME, title=f'ExplorationE{e}', additional_info=V_star, figure_path=dircty, save=save, show=show)
                        # print('Return', R)
                    obs.init_new_data_buffer()
                    # print("done with", h + 1, 'steps')
                    break
            r_all_epi.append(R)

            if len(obs._buffers['state0']) == len(env.learnable_no):
                print('============================', '\n', f'Finished exploration with {e} Episodes')
                # break
        r_all_repeat.append(r_all_epi)
        Q_all_repeat.append(agent.Q)
        N_all_repeat.append(agent.N)
        if save:
            torch.save(r_all_repeat, f'{dircty_top}Return.pt')
            torch.save(Q_all_repeat, f'{dircty_top}Q.pt')
            torch.save(N_all_repeat, f'{dircty_top}N.pt')

        plot_obs(obs, env, env_name=ENV_NAME, title=f'ExplorationE{e}', additional_info=V_star, figure_path=dircty, save=save, show=show)
        plot_return_vs_episodes(r_all_epi, repeat=repeat, save=save, figure_path=dircty, show=show)

    plot_return_vs_episodes_repeat(r_all_repeat, save=save, figure_path=dircty, show=show, smooth=10)
    if show:
        plt.show()
