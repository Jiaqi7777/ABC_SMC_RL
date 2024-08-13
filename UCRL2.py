from parameter import *

class UCRL2Agent:
    def __init__(self, env, horizon=HORIZON, alpha=0.2, gamma=1, delta=0.8, tol=1e-2):
        self.horizon = horizon
        self.alpha = alpha
        self.gamma = gamma
        self.state_n = env.n_cell
        self.action_n = env.action_space.n
        self.observation_n = env.observation_space.n
        self.delta = delta  # Use 10 times smaller confidence interval
        self.tol = tol
        self.reset()
        
    def reset(self):
        self.Q = np.zeros(shape=(self.state_n + (self.action_n, ))) / self.observation_n / self.action_n  # Value estimates
        self.N = np.zeros(shape=(self.state_n + (self.action_n, ))) / self.observation_n / self.action_n  # Action counts
        self.R_hat = env.Reward  # Estimated rewards
        # print('R_hat', self.R_hat)
        self.P_hat = np.zeros(shape=(self.state_n + (self.action_n, ) + self.state_n))  # Estimated transition probabilities
        self.policy = np.zeros(shape=(self.state_n), dtype=int)  # Current policy
        self.V = np.zeros(shape=(self.state_n))  # Value function
        self.obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
        self.return_l = []
        self.episode = 0

    def confidence_interval(self, n):
        return np.sqrt(2 * np.log(1 / self.delta) / n) if n > 0 else float('inf')
    
    def evi(self, S):
        V = np.zeros(shape=(self.state_n))
        V_new = self.V
        loop = 0
        while True:
            for s in S[::-1]:
                max_value = -float('inf')
                for a in range(self.action_n):
                    if self.N[s + (a,)] > 0:
                        # print(s, a)
                        ci = max(0, self.confidence_interval(self.N[s + (a,)]))
                        optimistic_reward = min(1, self.R_hat[s + (a,)] + ci)
                        optimistic_transitions = np.clip(self.P_hat[s + (a,)], 0, 1)
                        optimistic_transitions /= np.sum(optimistic_transitions)  # Ensure probabilities sum to 1
                        expected_value = optimistic_reward + self.gamma * V[np.unravel_index(np.argmax(optimistic_transitions), V.shape)]
                        # print(expected_value, max_value)
                        if expected_value > max_value:
                            # print(ci, optimistic_reward, optimistic_transitions, V, expected_value, max_value)
                            max_value = expected_value
                V_new[s] = max_value
                # print(s, V_new)
            # break
            finite_mask = np.isfinite(V) | np.isfinite(V_new)
            # print(V, V_new)
            if np.max(np.abs(V_new[finite_mask] - V[finite_mask])) < self.tol:
                print('Converged with loop ', loop, V_new)
                self.V = V_new
                break
            V = V_new
            loop += 1
        # Extract policy
        for s in S[::-1]:
            max_value = -float('inf')
            best_action = 0
            for a in range(self.action_n):
                if self.N[s + (a,)] > 0:
                    ci = max(0, self.confidence_interval(self.N[s + (a,)]))
                    optimistic_reward = min(1, self.R_hat[s + (a,)] + ci)
                    optimistic_transitions = np.clip(self.P_hat[s + (a,)], 0, 1)
                    optimistic_transitions /= np.sum(optimistic_transitions)
                    expected_value = optimistic_reward + self.gamma * V[np.unravel_index(np.argmax(optimistic_transitions), V.shape)]
                    print(optimistic_reward, expected_value, max_value)
                    if expected_value > max_value:
                        max_value = expected_value
                        best_action = a
                    elif expected_value == max_value:
                        best_action = np.random.choice([best_action, a])
            self.policy[s] = best_action
        print('Policy:', self.policy)
    
    def update_estimates(self, state, action, reward, next_state):
        self.N[state + (action,)] += 10
        n = self.N[state + (action,)]
        self.R_hat[state + (action,)] = 10 * (self.R_hat[state + (action,)] * (n - 1) + reward) / n
        self.P_hat[state + (action,) + next_state] += 1
        self.P_hat[state + (action,)] /= np.sum(self.P_hat[state + (action,)])
    
    def run_episode(self, env):
        self.episode += 1
        s0, _ = env.reset()
        R_ = 0
        while True:
            action = self.policy[s0]
            # print('action', action)
            s1, r, done, *info = env.step(action)
            # print(s0, action, s1, r)
            R_ += r
            self.obs.insert({'state0': s0, 'state1': s1, 'action': int(action), 'rewards': r, 'done': done}, unique=UNIQUE_OBS, update_new_data=True)
            self.update_estimates(s0, action, r, s1)
            if done:
                break
            s0 = s1
        self.return_l.append(R_)
    
    def train(self, env, num_episodes, S):
        for e in range(num_episodes):
            self.run_episode(env)
            self.evi(S)
            # plot_obs(agent.obs, env, env_name=ENV_NAME, title=f'ExplorationE{e}', additional_info=V_star, figure_path=dircty, save=save, show=show)

        
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
        env = DeepSea(depth=3)
        EPISODES = 1#env.n_cell[0] * 100
    if save:
        dircty_top = f'../UCRL2/{env.n_cell[0]}/c{c}/{time}/'
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

    dircty = ''
    for repeat in range(REPEAT_EXPERIMENT_NAIVE):
        agent = UCRL2Agent(env=env)
        if save:
            dircty = f'{dircty_top}R{repeat}/'
            if not os.path.exists(dircty):
                os.makedirs(dircty)
        
        agent.train(env, EPISODES, S)

        r_all_repeat.append(agent.return_l)
        Q_all_repeat.append(agent.Q)
        N_all_repeat.append(agent.N)
        if save:
            torch.save(r_all_repeat, f'{dircty_top}Return.pt')
            torch.save(Q_all_repeat, f'{dircty_top}Q.pt')
            torch.save(N_all_repeat, f'{dircty_top}N.pt')
        print(r_all_repeat)
        plot_return_vs_episodes(agent.return_l, repeat=repeat, save=save, figure_path=dircty, show=show)

    plot_return_vs_episodes_repeat(r_all_repeat, save=save, figure_path=dircty, show=show, smooth=1)
    if show:
        plt.show()
