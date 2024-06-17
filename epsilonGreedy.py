def epsilon_greedy(Q, s, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.choice(len(Q[s]))
    else:
        return np.random.choice(np.where(Q[s] == Q[s].max())[0])

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
    from model import *
    from QLearning import *
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--training_step', default=EG_T, type=int)
    parser.add_argument('-t', '--time', default=datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument('-s', '--save', default=SAVE)
    parser.add_argument('-p', '--show', default=SHOW)
    parser.add_argument('--seed', default=SEED, type=int)
    parser.add_argument('-eg', '--epsilon_greedy', default=EG_EPSILON, type=float)
    parser.add_argument('--Env', default=ENV_NAME)
    args = parser.parse_args()
    time = args.time
    save = args.save
    show = args.show
    seed = args.seed
    epsilon = args.epsilon_greedy
    training_steps = args.training_step
    env_name = args.Env
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dircty=''
    if save:
        dircty_top = f'EG/{time}/'
        if not os.path.exists(dircty_top):
            os.makedirs(dircty_top)

    if env_name == 'GridWorld':
        env = GridWorld((1, 2), obstacles=False, stochastic=STOCHASTIC)
        # env.plot_env()
    if env_name == 'Maze':
        env = Maze()
    if env_name == 'DeepSea':
        env = DeepSea(depth=10)
        EPISODES = 1000000#env.n_cell[0] * 100
    
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
    Q = np.zeros(shape=(env.n_cell + (env.action_space.n, ))) / env.observation_space.n / env.action_space.n
    A = range(env.action_space.n)
    # pi_star, Q_star, V_star = OfflineQLearning(Q, A, S, env, gamma=GAMMA, show=show, thresh=1e-3, alpha=1)
    pi_star, Q_star, V_star = DynamicProgramming(Q, A, S, env, gamma=GAMMA, show=show)
    dim = env.observation_space.n * env.action_space.n

    #SMC initialization
    alpha = ESS_ALPHA
    compromise_alpha = COMPROMISE_ALPHA
    
    env.reset()
    results = []
    r_all_repeat = []
    samples_all_repeat = []
    smc_all_repeat = []
    dircty = ''
    for repeat in range(REPEAT_EXPERIMENT):
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
                action = epsilon_greedy(Q, s0, epsilon=epsilon)
                s1, r, done, *info = env.step(action)
                R += r
                new_data_flag = (obs.insert({'state0': s0, 'state1': s1, 'action': int(action), 'rewards': r, 'done': done}, unique=UNIQUE_OBS, update_new_data=True) or new_data_flag)
                s0 = s1
                if done:
                    if new_data_flag:
                        print(f'Episode {e} in repeat {repeat}')
                        plot_obs(obs, env, env_name=ENV_NAME, title=f'ExplorationE{e}', additional_info=V_star, figure_path=dircty, save=save, show=show)
                    obs.init_new_data_buffer()
                    Q = QLearningWithData(Q, obs._buffers, training_steps=training_steps, gamma=1, alpha=1)
                    # print("done with", h + 1, 'steps')
                    # print('Return', R)
                    break
            r_all_epi.append(R)

            if len(obs._buffers['state0']) == len(env.learnable_no):
                print('============================', '\n', f'Finished exploration with {e} Episodes')
                # break
        r_all_repeat.append(r_all_epi)
        # smc_all_repeat.append(smc)
        # samples_all_repeat.append(samples_all_ep)
        # if save:
            # save_results(results=r_all_repeat, folder='Returns', dir=dircty, stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat)        
            # save_results(results=samples_all_repeat, folder='Samples', dir=dircty, stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat)
            # save_results(results=obs, folder='Obs', dir=dircty, stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat)
        plot_obs(obs, env, env_name=ENV_NAME, title=f'ExplorationE{e}', additional_info=V_star, figure_path=dircty, save=save, show=show)
        plot_return_vs_episodes(r_all_epi, repeat=repeat, save=save, figure_path=dircty, show=show)
        # display_smc_results(smc, save=save, figure_path=dircty, show=show)
    plot_return_vs_episodes_repeat(r_all_repeat, save=save, figure_path=dircty, show=show, smooth=10)
    torch.save(r_all_repeat, f'../EpsilonGreedy/Return_{time}_depth{env.n_cell[0]}_epsilon{epsilon}.pt')
    if show:
        plt.show()
