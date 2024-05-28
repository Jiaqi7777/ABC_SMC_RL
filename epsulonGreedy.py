if __name__ == '__main__':
    # from tqdm import tqdm
    import json
    from mcmcplot import mcmcplot as mcp
    from arviz import ess, plot_autocorr, plot_trace
    from sklearn.metrics import mean_squared_error
    import datetime
    import argparse
    '''module import'''
    from Environment.GridWorld import *
    from Environment.Maze import *
    from Environment.DeepSea import *
    from model import *
    from QLearning import *
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time', default=datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument('-s', '--save', default=SAVE)
    parser.add_argument('-p', '--show', default=SHOW)
    parser.add_argument('--seed', default=SEED, type=int)
    parser.add_argument('--Env', default=ENV_NAME)
    args = parser.parse_args()
    save = args.save
    show = args.show
    env_name = args.Env
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dircty=''
    if save:
        dircty_top = f'../SMC/{time}/'
        if not os.path.exists(dircty_top):
            os.makedirs(dircty_top)

    if env_name == 'GridWorld':
        env = GridWorld((1,2), obstacles=False, stochastic=STOCHASTIC)
        # env.plot_env()
    if env_name == 'Maze':
        env = Maze()
    if env_name == 'DeepSea':
        env = DeepSea(depth=15)
        EPISODES = 500#env.n_cell[0] * 100
    
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
    # model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal', mean=PRIOR_MEAN, std=PRIOR_SIGMA, gamma=GAMMA, initial_tables=Q_star, idx=FROZEN_IDX)
    # model.plot_policy(paras=np.array([Q_star]), title=f'True Values/Policy by Dynamic Programming', additional_info = V_star, show=show)
    
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
        epsilon = abc_epsilon
        STEPSIZE = INITIAL_STEPSIZE
        # model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal', gamma=GAMMA)
        model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal', mean=PRIOR_MEAN, std=PRIOR_SIGMA, gamma=GAMMA, initial_tables=Q_star, idx=FROZEN_IDX)#For frozen all but one dimensions
        smc = SMC(model=model, initial_params=model.get_learnable_parameter())
        smc.update_history(model.get_learnable_parameter(), torch.log(model._weights))
        r_all_epi = [] 
        samples_all_ep = []
        obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
        s0, _ = env.reset()
        for e in range(EPISODES):
            # objgraph.show_most_common_types()
            s0, _ = env.reset()
            para = model.sample_para()
            # plot_qtable(para, title=f'Sampled Q table for episode {e} repeat {repeat}', save=save, figure_path=dircty)
            print(f'Episode {e} in repeat {repeat} with epsilon={epsilon}')
            R = 0
            # h = 0
            # while True: #Turn on h += 1
            new_data_flag = False
            for h in tqdm(range(HORIZON)):
                
                action = model.act(s0, para, greedy=GREEDY)
                s1, r, done, *info = env.step(action)
                # if STOCHASTIC:
                #     s1_augmented = []
                #     for _ in range(M_Z):
                #         s1_augmented.append(env.step(action, state=s0)[0])
                    # print(s1_augmented)
                R += r
                new_data_flag = (obs.insert({'state0': s0, 'state1': s1, 'action': int(action), 'rewards': r, 'done': done}, unique=UNIQUE_OBS, update_new_data=True) or new_data_flag)
                s0 = s1
                if done or ( h + 1)  % FROZEN_T == 0:
                    if new_data_flag:
                        plot_obs(obs, env, env_name=ENV_NAME, title=f'ExplorationE{e}', additional_info=V_star, figure_path=dircty, save=save, show=show)
                    # model.set_learnable_idx(obs)
                    smc_samples = torch.tensor(model.get_learnable_parameter())
                    #MCMC
                    # posterior_samples, accept_probs = MCMC_update(posterior_samples=smc_samples, obs=obs, model=model, env=env)[:2]
                    # mode_idx = torch.argmax(mcmc.logdensities) if 'mcmc' in vars() else None
                    pre_sample_size = len(smc.samples)
                    if TRANSFORM:
                        smc_samples = TruncatedGaussianABCLikelihood.log_neg_transform(smc_samples)
                    if new_data_flag and e > 0:
                        Model = get_SMC_Model(obs=obs, model=model, env=env, epsilon=epsilon)
                        smc.SMC_Model = Model
                        _, smc_samples = smc.update(alpha, smc_samples, episode=e, repeat=repeat)
                        # plt.plot(smc.bellman_err_l[-1])
                        # if save:
                        #     plt.savefig(f'{dircty}bellmanErrE{e}NewData.png', bbox_inches='tight')
                        # plt.show()
                    # model.sample_random_tables(set=True)
                    obs.init_new_data_buffer()
                    Model = get_SMC_Model(obs=obs, model=model, env=env, epsilon=epsilon)
                    smc.SMC_Model = Model
                    
                    #fixed decreasing
                    if STOPPING_CRITERIA == 'fixed_reduce':
                        epsilon *= min(1, (0.35 + e * 0.01)**0.1)
                        _, smc_samples = smc.update(alpha, smc_samples, epsilon=epsilon, episode=e, repeat=repeat)
                    
                    #Natural decreasing
                    elif STOPPING_CRITERIA == 'natural_reduce':
                        epsilon, smc_samples = smc.update(alpha, smc_samples, epsilon=epsilon, episode=e, repeat=repeat, error_lag=ERROR_LAG, error_perc=ERROR_PERCENTAGE)
                        plot_save(smc.bellman_err_l, figure_path=dircty, repeat=repeat, save=save, title='bellmanErr', show=show)
                    
                    if TRANSFORM:
                        smc_samples = TruncatedGaussianABCLikelihood.neg_exp_transform(smc_samples)
                    if len(smc.samples) - pre_sample_size <=2:
                        pre_sample_size -= min(pre_sample_size, 10)
                    # display_smc_results(smc, n=pre_sample_size, episode=e, repeat=repeat, save=save, figure_path=dircty, show=show)
                if done:
                    print("done with", h + 1, 'steps')
                    print('Return', R)
                    break
                # h += 1
            r_all_epi.append(R)
            # samples_all_ep.append(smc_samples)
            explore_pct = [model.get_parameter().numpy()[:, i, i, 0] > model.get_parameter().numpy()[:, i, i, 1] for i in range(env.n_cell[0] - 1)]
            explore_pct_all = np.sum([np.logical_and.reduce(explore_pct[:i + 1], axis=0) for i in range(len(explore_pct))], axis=1) / len(smc_samples)
            # plot_save(smc.epsilon_epi, figure_path=dircty, episode=e, repeat=repeat, save=save, title=f'Epsilon{smc.epsilon_all_history[-1]}', show=show)
            plot_save(smc.epsilon_all_history[:], figure_path=dircty, repeat=repeat, save=save, title=f'Epsilon', show=show)
            # plot_save(smc.mcmc_steps_epi[:], figure_path=dircty, episode=e, repeat=repeat, save=save, title=f'MCMCSteps', show=show)
            # plot_save(smc.ess_history[:], figure_path=dircty, repeat=repeat, save=save, title='ESS')
            print('explore percentage', explore_pct_all)
            if save:
                save_results(results=r_all_epi, folder='Returns', dir=dircty, stochastic=STOCHASTIC, episode=e, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat, episodic=False)
                save_results(results=smc.samples, folder='Samples', dir=dircty, stochastic=STOCHASTIC, episode=e, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat, episodic=False)
                save_results(results=smc._weights_history, folder='Weights', dir=dircty, stochastic=STOCHASTIC, episode=e, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat, episodic=False)
                save_results(results=obs, folder='Obs', dir=dircty, stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat)
            if len(obs._buffers['state0']) == smc.params_dim:
                print('============================', '\n', f'Finished exploration with {e} Episodes')
                # break
        r_all_repeat.append(r_all_epi)
        # smc_all_repeat.append(smc)
        # samples_all_repeat.append(samples_all_ep)
        # if save:
            # save_results(results=r_all_repeat, folder='Returns', dir=dircty, stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat)        
            # save_results(results=samples_all_repeat, folder='Samples', dir=dircty, stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat)
            # save_results(results=obs, folder='Obs', dir=dircty, stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat)
        plot_return_vs_episodes(r_all_epi, repeat=repeat, save=save, figure_path=dircty, show=show)
        # display_smc_results(smc, save=save, figure_path=dircty, show=show)
    plot_return_vs_episodes_repeat(r_all_repeat, save=save, figure_path=dircty)
    if show:
        plt.show()
