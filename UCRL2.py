from parameter import *
import sys
import numpy as np

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

# PSRL agent for 2D Grid
class PSRL:
    def __init__(self, env, gamma=1):
        self.n = env.n_cell[0]
        self.n_actions = env.action_space.n
        self.return_l = []
        self.Q = np.zeros((self.n, self.n, self.n_actions))
        self.gamma = gamma
        # self.horizon = horizon
        
        # Initialize transition counts and rewards as if observed 10 times
        self.transition_counts = np.ones((self.n, self.n, self.n_actions, self.n, self.n)) 
        self.rewards = np.zeros((self.n, self.n, self.n_actions))
        self.reward_counts = np.ones((self.n, self.n, self.n_actions))

    def update_model(self, s, action, r, s_prime):
        """ Update the transition and reward model. """
        self.transition_counts[s[0], s[1], action, s_prime[0], s_prime[1]] += 10
        self.reward_counts[s[0], s[1], action] += 10
        self.rewards[s[0], s[1], action] += 10 * r

    def sample_model(self):
        """ Sample the transition and reward models for planning. """
        sampled_transitions = np.zeros((self.n, self.n, self.n_actions, self.n, self.n))
        sampled_rewards = np.zeros((self.n, self.n, self.n_actions))
        
        for i in range(self.n - 1):
            for j in range(i+1):
                for a in range(self.n_actions):
                    # Sample next state based on transition counts
                    sampled_transitions[i, j, a] = np.random.dirichlet(self.transition_counts[i, j, a].flatten()).reshape(self.n, self.n)
                    # print(np.random.dirichlet(self.transition_counts[i, j, a].flatten()).reshape(5,5)[0], sampled_transitions[i, j, a][0])
                    # Sample reward based on observed rewards
                    reward_mean = self.rewards[i, j, a] / self.reward_counts[i, j, a]
                    sampled_rewards[i, j, a] = np.random.normal(reward_mean, 0.1 / np.sqrt(self.reward_counts[i, j, a]))
        return sampled_transitions, sampled_rewards
    
    def plan_old(self, transitions, rewards):
        """ Value iteration to find the optimal policy. """
        V = np.zeros((self.n, self.n))
        policy = np.zeros((self.n, self.n), dtype=int)
        
        for _ in range(self.n):
            V_next = np.zeros((self.n, self.n))
            for i in range(self.n):
                for j in range(i+1):
                    Q_values = [rewards[i, j, a] + self.gamma * V[transitions[i, j, a][0], transitions[i, j, a][1]]
                                for a in range(self.n_actions)]
                    V_next[i, j] = max(Q_values)
                    max_indices = np.flatnonzero(Q_values == np.max(Q_values))
                    policy[i, j] = np.random.choice(max_indices)
                    self.Q[i, j] = torch.tensor(Q_values)
            V = V_next
        return policy
    def plan(self, transition_model, reward_model, max_iterations=1000, tol=1e-6):
        # Initialize value function for all states
        V = np.zeros((self.n, self.n))
        
        # Run value iteration
        for _ in range(max_iterations):
            V_prev = V.copy()
            for i in range(self.n - 1):
                for j in range(i+1):
                # Bellman update for each state
                    V[i, j] = max(
                        reward_model[i, j, a] + self.gamma * np.sum(transition_model[i, j, a] * V_prev)
                        for a in range(self.n_actions)
                    )
            # Check for convergence
            if np.max(np.abs(V - V_prev)) < tol:
                break
        
        # Extract the optimal policy
        policy = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n - 1):
                for j in range(i+1):
                    self.Q[i, j] = np.array([
                        reward_model[i, j, a] + self.gamma * np.sum(transition_model[i, j, a] * V)
                        for a in range(self.n_actions)
                    ])
                    policy[i, j] = np.argmax([self.Q[i, j]])
        
        return policy

# UCRL2 agent for 2D Grid
class UCRL2:
    def __init__(self, env):
        self.n = env.n_cell[0]
        self.n_actions = env.action_space.n
        # self.horizon = horizon
        
        # Initialize transition counts and rewards as if observed 10 times
        self.transition_counts = np.ones((self.n, self.n, self.n_actions, self.n, self.n)) * 10
        self.reward_sums = np.zeros((self.n, self.n, self.n_actions))
        self.reward_counts = np.ones((self.n, self.n, self.n_actions)) * 10

    def update_model(self, s, action, r, s_prime):
        """ Update the transition and reward models. """
        self.transition_counts[s[0], s[1], action, s_prime[0], s_prime[1]] += 10
        self.reward_sums[s[0], s[1], action] += 10 * r
        self.reward_counts[s[0], s[1], action] += 10

    def compute_confidence_bounds(self, s, a, total_transitions):
        """ Compute reduced confidence bounds (scaled by factor of 10). """
        return 1 / np.sqrt(10 * total_transitions[s[0], s[1], a])

    def plan(self, gamma=1):
        """ Value iteration with confidence bounds. """
        V = np.zeros((self.n, self.n))
        policy = np.zeros((self.n, self.n), dtype=int)
        
        for _ in range(self.horizon):
            V_next = np.zeros((self.n, self.n))
            for i in range(self.n):
                for j in range(self.n):
                    Q_values = []
                    for a in range(self.n_actions):
                        total_transitions = np.sum(self.transition_counts[i, j, a])
                        expected_reward = self.reward_sums[i, j, a] / self.reward_counts[i, j, a]
                        confidence_bound = self.compute_confidence_bounds((i, j), a, total_transitions)
                        Q_value = expected_reward + confidence_bound + gamma * np.max(V)
                        Q_values.append(Q_value)
                    V_next[i, j] = max(Q_values)
                    policy[i, j] = np.argmax(Q_values)
            V = V_next
        self.Q = Q_values
        return policy


        
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
    parser.add_argument('--algorithm', default='PSRL', type=str)
    parser.add_argument('--Env_d', default=10, type=int)
    parser.add_argument('--episode', default=1e6, type=int)
    parser.add_argument('--load', default=False, action='store_true', help='Bool type')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--sample_path', type=str, default='')
    args = parser.parse_args()
    time = args.time
    save = args.save
    show = args.show
    seed = args.seed
    c = args.c_ucb
    training_steps = args.training_step
    env_name = args.Env
    algorithm = args.algorithm
    Env_d = args.Env_d
    EPISODES = args.episode
    load = args.load
    load_path = args.load_path
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(args)
    if load:
        load_path_info = load_path.split('/')
        env_d, time, initial_repeat = load_path_info
        initial_repeat = int(initial_repeat[1:])
    else:
        time = args.time
        initial_repeat = 0
    
    dircty=''
    dircty_top = ''
    env_d_l = [22, 23]
    skip = env_d_l.index(int(env_d)) if load else 0
    ep_l = [1400000, 2000000, 150000, 2500000, 2600000, 2800000, 3000000, 4000000][skip:]
    for idx, env_d in enumerate(env_d_l[skip:]):
        print(f'Running {env_d} depth with {ep_l[idx]} episodes')
        if env_name == 'GridWorld':
            env = GridWorld((1, 2), obstacles=False, stochastic=STOCHASTIC)
            # env.plot_env()
        if env_name == 'Maze':
            env = Maze()
        if env_name == 'DeepSea':
            if load:
                env_d = int(load_path_info[0])
            env = DeepSea(depth=env_d)
            EPISODES = ep_l[idx]#env.n_cell[0] * 100
        if save:
            dircty_top = f'../{algorithm}/{env.n_cell[0]}/{time}/'
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
        BestValue = np.max(Q_star)
        dim = env.observation_space.n * env.action_space.n
        # Q = np.random.normal(loc=PRIOR_MEAN, scale=PRIOR_SIGMA, size=(env.n_cell + (env.action_space.n, )))
        # if env.not_learnable_idx:
        #     Q[env.not_learnable_idx] = 0
        
        env.reset()
        results = []
        r_all_repeat = []
        Q_all_repeat = []
        N_all_repeat = []
        R_count_all_repeat = []
        Rewards_all_repeat = []

        dircty = ''
        for repeat in range(initial_repeat, REPEAT_EXPERIMENT_NAIVE):
            print(f'Running {algorithm} for {repeat}th repeat')
            sys.stdout.flush()
            # agent = UCRL2Agent(env=env)
            if algorithm == 'PSRL':
                agent = PSRL(env)
                if load:
                    folder_path = f'../{algorithm}/{env_d}/{time}/R{repeat}/'
                    if not os.path.exists(folder_path): 
                        print('use collection of data')
                        folder_path = f'../{algorithm}/{env_d}/{time}/'
                        for file_name in os.listdir(folder_path):
                            if file_name.endswith('.pt'):
                                record_path = os.path.join(folder_path, file_name)
                                record = torch.load(record_path)
                                if len(record) - 1 < repeat:
                                    print('No data for this repeat')
                                    load = False
                                    break
                            if file_name.startswith('Return') and file_name.endswith('.pt'):
                                return_path = os.path.join(folder_path, file_name)
                                agent.return_l = torch.load(return_path)[repeat]
                            if file_name.startswith('Q') and file_name.endswith('.pt'):
                                Q_path = os.path.join(folder_path, file_name)
                                agent.Q = torch.load(Q_path)[repeat]
                            if file_name.startswith('N') and file_name.endswith('.pt'):
                                N_path = os.path.join(folder_path, file_name)
                                agent.transition_counts = torch.load(N_path)[repeat]
                            if file_name.startswith('R_count') and file_name.endswith('.pt'):
                                R_count_path = os.path.join(folder_path, file_name)
                                agent.reward_counts = torch.load(R_count_path)[repeat]
                            if file_name.startswith('Rewards') and file_name.endswith('.pt'):
                                Rewards_path = os.path.join(folder_path, file_name)
                                agent.rewards = torch.load(Rewards_path)[repeat]
                    else:
                        for file_name in os.listdir(folder_path):
                            if file_name.startswith('Return') and file_name.endswith('.pt'):
                                return_path = os.path.join(folder_path, file_name)
                                agent.return_l = torch.load(return_path)
                            if file_name.startswith('Q') and file_name.endswith('.pt'):
                                Q_path = os.path.join(folder_path, file_name)
                                agent.Q = torch.load(Q_path)
                            if file_name.startswith('N') and file_name.endswith('.pt'):
                                N_path = os.path.join(folder_path, file_name)
                                agent.transition_counts = torch.load(N_path)
                            if file_name.startswith('R_count') and file_name.endswith('.pt'):
                                R_count_path = os.path.join(folder_path, file_name)
                                agent.reward_counts = torch.load(R_count_path)
                            if file_name.startswith('Rewards') and file_name.endswith('.pt'):
                                Rewards_path = os.path.join(folder_path, file_name)
                                agent.rewards = torch.load(Rewards_path)

                psrl_transitions, psrl_rewards = agent.sample_model()
                psrl_policy = agent.plan(psrl_transitions, psrl_rewards)
                ep_ = len(agent.return_l)
                for ep in range(ep_, EPISODES):
                # while True:
                    # PSRL planning
                    psrl_transitions, psrl_rewards = agent.sample_model()
                    psrl_policy = agent.plan(psrl_transitions, psrl_rewards)
                    s0, _ = env.reset()
                    done = False
                    R = 0
                    while not done:
                        action = psrl_policy[s0]
                        s1, r, done, *info = env.step(action)
                        agent.update_model(s0, action, r, s1)
                        s0 = s1
                        R += r
                    agent.return_l.append(R)
                    # ep += 1
                    if ep % 1000 == 0:
                        print(f'episodes {ep} with average return {sum(agent.return_l)/ep}')
                        if save:
                            repeat_path = f'{dircty_top}R{repeat}/'
                            if not os.path.exists(repeat_path):
                                os.makedirs(repeat_path)
                            torch.save(agent.return_l, f'{repeat_path}Return.pt')
                            torch.save(agent.Q, f'{repeat_path}Q.pt')
                            torch.save(agent.transition_counts, f'{repeat_path}N.pt')
                            torch.save(agent.reward_counts, f'{repeat_path}R_count.pt')
                            torch.save(agent.rewards, f'{repeat_path}Rewards.pt')
                    sys.stdout.flush()
                    if  ep > 0 and sum(agent.return_l)/ep > 0.12:
                        print('===================================')
                        print('\n', '\n')
                        #0.12L + X = 0.49 (L + X) -> 0.37L = 0.51X -> X = 0.37/0.51 L = 0.72 L
                        break
            # UCRL2 planning
            elif algorithm == 'UCRL2':
                agent = UCRL2(env)
                ucrl2_policy = agent.plan()
                # for episode in range(EPISODES):
                ep = 0
                while True:
                    ep += 1
                    # PSRL planning
                    ucrl2_policy = agent.plan()
                    s0, _ = env.reset()
                    done = False
                    R = 0
                    while not done:
                        action = ucrl2_policy[s0]
                        s1, r, done, *info = env.step(action)
                        agent.update_model(s0, action, r, s1)
                        s0 = s1
                        R += r
                    agent.return_l.append(R)
                    if sum(agent.return_l)/ep > 0.12:
                        break

            r_all_repeat.append(agent.return_l)
            Q_all_repeat.append(agent.Q)
            N_all_repeat.append(agent.transition_counts)
            R_count_all_repeat.append(agent.reward_counts)  
            Rewards_all_repeat.append(agent.rewards)
            if save:
                torch.save(r_all_repeat, f'{dircty_top}Return.pt')
                torch.save(Q_all_repeat, f'{dircty_top}Q.pt')
                torch.save(N_all_repeat, f'{dircty_top}N.pt')
                torch.save(R_count_all_repeat, f'{dircty_top}R_count.pt')
                torch.save(Rewards_all_repeat, f'{dircty_top}Rewards.pt')   

            # plot_return_vs_episodes(agent.return_l, repeat=repeat, save=save, figure_path=dircty, show=show)

            plot_return_vs_episodes_repeat(append_rewards(r_all_repeat), save=save, figure_path=dircty_top, show=show, smooth=1)
        if show:
            plt.show()
        load = False