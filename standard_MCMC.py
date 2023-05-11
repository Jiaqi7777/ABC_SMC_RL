import pyro
import pyro.distributions as dist
import torch
import matplotlib.pyplot as plt
from parameter import *
from MCMC.MCMC import *
from functools import partial
from QLearning import *


def likelihood(data):
    obs, model, dim, batch_indices, abc_epsilon = data
    if not BATCH_TRAINING:
        batch_indices = range(min(len(obs['state0']), BUFFER_SIZE))
    r_hat = torch.tensor(obs['rewards'])[-BUFFER_SIZE:][batch_indices]
    
    prior_parameter = pyro.sample("prior_parameter", dist.MultivariateNormal(torch.zeros(dim), PRIOR_SIGMA **2 * torch.eye(dim)))
    mean = generate_samples(prior_parameter, model, obs, batch_indices=batch_indices).reshape(-1)
    with pyro.plate("data_plate"):
        pyro.sample("obs", dist.MultivariateNormal(mean, abc_epsilon ** 2 * torch.eye(len(r_hat))), obs=r_hat)


def mcmc(data, prior_parameter, num_samples=MCMC_SAMPLE, warmup_steps=MCMC_T//10, MCMC_SHOW_DISABLE=MCMC_SHOW_DISABLE, stepsize=STEPSIZE):
    pyro.clear_param_store()
    kernel = pyro.infer.mcmc.HMC(likelihood, full_mass=FULL_MASS, step_size=stepsize, adapt_step_size=ADAPT_STEP_SIZE, adapt_mass_matrix=ADAPT_MASS_MATRIX, target_accept_prob=TARGET_ACCEPT_PROB, num_steps=NUM_STEPS)
    # kernel = pyro.infer.mcmc.HMC(likelihood, full_mass=True, step_size=stepsize, adapt_step_size=ADAPT_STEP_SIZE, adapt_mass_matrix=ADAPT_MASS_MATRIX, target_accept_prob=TARGET_ACCEPT_PROB)
    mcmc_run = pyro.infer.mcmc.MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, initial_params={'prior_parameter': prior_parameter}, disable_progbar=MCMC_SHOW_DISABLE)
    mcmc_run.run(data)

    return mcmc_run


if __name__ == '__main__':
    from tqdm import tqdm
    from mcmcplot import mcmcplot as mcp
    from arviz import ess, plot_autocorr, plot_trace
    import datetime
    import argparse
    '''module import'''
    from Environment.GridWorld import *
    from Environment.Maze import *
    from model import *
    from MCMC.MCMC import *
    from QLearning import *
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--training_step', default=MCMC_T, type=int)
    parser.add_argument('-t', '--time', default=datetime.datetime.now().strftime("%f"))
    parser.add_argument('-s', '--save', default=SAVE)
    parser.add_argument('-p', '--show', default=SHOW)
    parser.add_argument('-e', '--epsilon', default=EPSILON, type=float)
    parser.add_argument('-n', '--stepsize', default=STEPSIZE, type=float)
    parser.add_argument('--seed', default=SEED, type=int)
    parser.add_argument('--MCMC', default=True, action='store_false', help='Bool type')
    parser.add_argument('-g', '--Greedy', default=GREEDY, action='store_true', help='Bool type')
    parser.add_argument('--Env', default=ENV_NAME)
    args = parser.parse_args()
    print(args)
    time = args.time
    print('time:', time)
    training_steps = args.training_step
    save = args.save
    show = args.show
    abc_epsilon = args.epsilon
    seed = args.seed
    MCMC_SHOW_DISABLE=args.MCMC
    STEPSIZE = args.stepsize
    warmup_steps = int(training_steps * WARMUP_RATIO)
    GREEDY = args.Greedy
    env_name = args.Env

    N_PARTICLE = 10
    random.seed(seed)
    pyro.set_rng_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if env_name == 'GridWorld':
        env = GridWorld((3,4), obstacles=True)
        env.plot_env()
    if env_name == 'Maze':
        env = Maze()
    dim = env.observation_space.n * env.action_space.n
    model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal')
    
    S = []
    if len(env.n_cell) == 1:
        S = [(i, ) for i in range(env.n_cell[0])]
    else:
        for i in range(env.n_cell[0]):
            for j in range(env.n_cell[1]):
                S.append((i,j)) 
    Q = np.ones(shape=(env.n_cell + (env.action_space.n, )))/env.observation_space.n / env.action_space.n
    A = range(env.action_space.n)
    pi_star, Q_star, V_star = DynamicProgramming(Q, A, S, env, gamma=GAMMA, show=show)
    
    env.reset()
    results = []
    if ONLINE_LEARNING:
        r_all_iter = []
        for repeat in range(REPEAT_EXPERIMENT):
            STEPSIZE = INITIAL_STEPSIZE
            model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal')
            r_all_epi = []
            obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
            s0, _ = env.reset()
            posterior_samples = torch.tensor(model.get_parameter())
            for e in range(EPISODES):
                s0, _ = env.reset()
                para = model.sample_para()
                print(f'Episode {e} in repeat {repeat}')
                R = 0
                R_star = 0
                h = 0
                # while True: #Turn on h += 1
                for h in tqdm(range(HORIZON)):
                    action = model.act(s0, para, greedy=GREEDY)
                    s1, r, done, *info = env.step(action)
                    #Optimal action
                    # R_star = V_star[s0] + gamma * R_star#env.R[tuple(env.P[s0 + (int(pi_star[s0]), )])] #for regret
                    R += r#sum([r * gamma ** i for i in range(h + 1)])
                    obs.insert({'state0': s0, 'state1': s1, 'action': int(action), 'rewards': r, 'done': done}, unique=UNIQUE_OBS)
                    s0 = s1
                    # if e==0 and h==0:
                    #     print(posterior_samples[:, 0, 0], h, posterior_samples.shape)
                    #     print(np.argmax(posterior_samples[:, 0, 0], axis=-1))
                    
                    if done or ( h + 1)  % FROZEN_T == 0:
                        # print(obs._buffers['state0'][-FROZEN_T:])
                        # print(obs._buffers)
                        #MCMC
                        if BATCH_TRAINING:
                            batch_indices = random.sample(range(min(len(obs._buffers['state0']), BUFFER_SIZE)), k=min(BATCH_SIZE, len(obs._buffers['state0']))) #TODO: what is this?
                        else:
                            batch_indices = slice(None)
                        r_hat = partial(generate_samples, model=model, obs=obs._buffers,  batch_indices=batch_indices)

                        def tabular_indicator(para, model, obs):
                            s0 = obs['state0']
                            s1 = obs['state1']
                            a = obs['action']
                            done = np.array(obs['done'])
                            s01, s02 = np.array(s0).T
                            s11, s12 = np.array(s1)[done == False].T
                            a_prime = np.argmax(para[s11, s12], axis=-1)
                            Indicator = np.zeros(shape=(len(a), ) + para.shape) #TxTheta
                            Indicator[range(len(a)), s01, s02, a] = 1.
                            Indicator[range(len(a_prime)), s11, s12, a_prime] -= model.gamma
                            return torch.tensor(Indicator, dtype=torch.float32).reshape((len(a),-1))
                        
                        llh_transform_grad_fn = lambda parameter:  tabular_indicator(para=parameter.reshape(env.n_cell + (env.action_space.n, )), model=model, obs=obs._buffers)#standard form

                        prior = IsotropicGaussianPrior(sd=PRIOR_SIGMA)
                        abclikelihood = GaussianABCLikelihood(epsilon=EPSILON)
                        data = torch.tensor(obs._buffers["rewards"])[-BUFFER_SIZE:][batch_indices]
                        Model = DeterministicSRModel(prior=prior, abclikelihood=abclikelihood, data=data, llh_transform_fn=r_hat, llh_transform_grad_fn=llh_transform_grad_fn)

                        def fn(parameter):
                            current_logtarget_density, _ = Model.logtarget_density(parameter=parameter, llh_info_dict=dict())
                            return current_logtarget_density
                        hessian = torch.autograd.functional.hessian(fn, posterior_samples[0].reshape(-1))
                        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE)
                        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE, covariance_matrix=-torch.linalg.inv(hessian))
                        #kernel = pCN(model=Model, stepsize=STEPSIZE)
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=None)
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=-torch.linalg.inv(hessian))
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=False)
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=False, precondition_matrix=-torch.linalg.inv(hessian))
                        #kernel = HMC_pyro(model=Model, stepsize=STEPSIZE, full_mass=FULL_MASS, adapt_step_size=ADAPT_STEP_SIZE, adapt_mass_matrix=ADAPT_MASS_MATRIX, target_accept_prob=TARGET_ACCEPT_PROB, num_steps=NUM_STEPS)
                        #kernel = HMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=False)
                        kernel = HMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=False, precondition_matrix=-torch.linalg.inv(hessian), traj_len=None)
                        #kernel = HMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=True, precondition_matrix=-torch.linalg.inv(hessian), traj_len=None)
                        #kernel = mMALA(model=Model, stepsize=STEPSIZE, use_autograd=False, use_autohess=False)
                        #kernel = mMALA(model=Model, stepsize=STEPSIZE, use_autograd=True, use_autohess=True)
                        #kernel = mHMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=False, use_autohess=False, traj_len=None, fp_iterations=50)
                        accept_probs = None

                        if kernel.original is True:
                            mcmc = MCMC(num_samples=training_steps, kernel=kernel, initial_params=posterior_samples[-1].reshape(-1), warmup_steps=np.int64(np.floor(training_steps*WARMUP_RATIO)), warup_settings=dict(target_prob=0.7, auto_init_stepsize=True))
                            posterior_samples = mcmc.run().reshape((-1, ) + env.n_cell + (env.action_space.n, ))
                            logdensities = mcmc.get_logdensities()
                            proposed_logdensities = mcmc.get_proposed_logdensities()
                            accept_probs = mcmc.get_accept_prob()

                        else:
                            mcmc = MCMC_pyro(num_samples=training_steps, kernel=kernel, initial_params=posterior_samples[-1].reshape(-1), warmup_steps=np.int64(np.floor(training_steps*WARMUP_RATIO)), disable_progbar=MCMC_SHOW_DISABLE)
                            posterior_samples = mcmc.run().reshape((-1, ) + env.n_cell + (env.action_space.n, ))

                        STEPSIZE *= DECREASING_FACTOR
                        # posterior_samples = new_posterior_samples
                        model.plot_policy(paras=posterior_samples.numpy(), title=f'policy_T{training_steps}_{time}', additional_info = env.R, save=save, show=show)
                        # model.plot_value(paras=posterior_samples.numpy(), title=f'value_T{training_steps}_{time}')
                        plt.imshow(torch.round(torch.max(torch.mean(posterior_samples, 0), -1).values, decimals=2))
                        if show:
                            plt.show()
                        data_plot = posterior_samples.numpy().reshape(posterior_samples.shape[0], -1)[:, OBSERVE_DATA_START:OBSERVE_DATA_END]#Change the indices of names and Q_star below as well
                        f = mcp.plot_chain_panel(chains=data_plot, names=env.names[OBSERVE_DATA_START:OBSERVE_DATA_END],
                                                                        settings=dict(add_pm2std=True, fig=dict(figsize=(10,10), dpi=250),
                                                                        mean=dict(color='y', label='mean'),
                                                                        plot=dict(color='k', label='trace')))
                        ax = f.get_axes()
                        for i, ai in enumerate(ax):
                            ai.axhline(y = Q_star.flatten()[OBSERVE_DATA_START:OBSERVE_DATA_END][i], linestyle=':', linewidth=5, color = 'g',  label = 'true q')
                            q = np.percentile(data_plot[:, i], [PLOT_THRESHOLD, 100 - PLOT_THRESHOLD])
                            ai.set_ylim(q)   
                        f.tight_layout()
                        handles, labels = ai.get_legend_handles_labels()
                        ai.legend(handles, labels, bbox_to_anchor=(2, 0.2), loc='right')
                        if show:
                            plt.show()

                        if accept_probs is not None:
                            fig, ax = plt.subplots(1,1,sharex=True)

                            ax.plot(proposed_logdensities.numpy(), label="proposed samples")
                            ax.plot(logdensities.numpy(), label="accepted samples")
                            ax.set_xlabel("samples")
                            ax.set_ylabel("log density")

                            ax2 = ax.twinx()
                            ax2.plot(accept_probs.numpy(), label="log acceptance probability", c="tab:green")
                            ax2.set_ylabel("log acceptance probability")

                            fig.legend()
                            fig.tight_layout()

                            if show:
                                plt.show()

                    if done:
                        print("done with", h + 1, 'steps')
                        print('Return', R)
                        break
                    # h += 1
                r_all_epi.append(R)
                with open(f'Returns/MCMC/Episode{e}T{training_steps}_Gdy{GREEDY}_Ep{EPSILON}_Stp{INITIAL_STEPSIZE}_Dcrs{DECREASING_FACTOR}_{time}.npy', 'wb') as f:
                    np.save(f, r_all_epi)
                    print(f'EPISODES return for repeat {repeat} saved at', f'Returns/MCMC/Episode{e}T{training_steps}_Gdy{GREEDY}_Ep{EPSILON}_Stp{INITIAL_STEPSIZE}_Dcrs{DECREASING_FACTOR}_{time}.npy')
            r_all_iter.append(r_all_epi)
            with open(f'Returns/MCMC/T{training_steps}_Gdy{GREEDY}_Ep{EPSILON}_Stp{INITIAL_STEPSIZE}_Dcrs{DECREASING_FACTOR}_{time}.npy', 'wb') as f:
                np.save(f, r_all_iter)
                print('return saved at', f'Returns/MCMC/T{training_steps}_Gdy{GREEDY}_Ep{EPSILON}_Stp{INITIAL_STEPSIZE}_Dcrs{DECREASING_FACTOR}_{time}.npy')
            
        plt.plot(R)
        if show:
            plt.show()
            
    else:
        env.uniform_policy()
        obs = env.uniform_obs._buffers
        r_hat = partial(generate_samples, model=model, obs=obs)

        posterior_samples = model.get_parameter()
        def tabular_indicator(para, model, obs):
            s0 = obs['state0']
            s1 = obs['state1']
            a = obs['action']
            done = np.array(obs['done'])
            s01, s02 = np.array(s0).T
            s11, s12 = np.array(s1)[done == False].T
            a_prime = np.argmax(para[s11, s12], axis=-1)
            Indicator = np.zeros(shape=(len(a), ) + para.shape) #TxTheta
            Indicator[range(len(a)), s01, s02, a] = 1.
            Indicator[range(len(a)), s11, s12, a_prime] -= model.gamma
            return torch.tensor(Indicator, dtype=torch.float32).reshape((len(a),-1))
        
        llh_transform_grad_fn = lambda parameter:  tabular_indicator(para=parameter.reshape(env.n_cell + (env.action_space.n, )), model=model, obs=obs._buffers)#standard form

        prior = IsotropicGaussianPrior()
        abclikelihood = GaussianABCLikelihood(epsilon=EPSILON)
        data = torch.tensor(obs["rewards"])
        Model = DeterministicSRModel(prior=prior, abclikelihood=abclikelihood, data=data, llh_transform_fn=r_hat, llh_transform_grad_fn=llh_transform_grad_fn)

        def fn(parameter):
            current_logtarget_density, _ = Model.logtarget_density(parameter=parameter, llh_info_dict=dict())
            return current_logtarget_density

        hessian = torch.autograd.functional.hessian(fn,posterior_samples[-3].reshape(-1)) + 1.e-6
        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE)
        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE, covariance_matrix=-torch.linalg.inv(hessian))
        #kernel = pCN(model=Model, stepsize=STEPSIZE)
        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=None)
        kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=-torch.linalg.inv(hessian))
        #kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=False)
        #kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=False, precondition_matrix=-torch.linalg.inv(hessian))

        mcmc = MCMC(num_samples=training_steps, kernel=kernel, initial_params=posterior_samples[-1].reshape(-1))
        posterior_samples = mcmc.run().reshape((-1, ) + env.n_cell + (env.action_space.n, ))

        
        #mcmc_run = mcmc(torch.tensor(obs['rewards']), torch.tensor(model.get_parameter()[0].reshape(-1)), num_samples=training_steps, warmup_steps=training_steps//10)
        #posterior_samples = mcmc_run.get_samples()["prior_parameter"]
        model.plot_policy(paras=posterior_samples.numpy().reshape((-1, ) + env.n_cell + (env.action_space.n, )), title=f'policy_T{training_steps}_{time}', additional_info = env.R, save=save, show=show)
        # print('ESS:', ess(chain.T))
        posterior_samples = posterior_samples.reshape(len(posterior_samples),-1)
        f = mcp.plot_chain_panel(chains=posterior_samples.numpy()[training_steps // 5:, :4], names=env.names,
                                                                        settings=dict(add_pm2std=True, fig=dict(figsize=(10,10), dpi=250),
                                                                        mean=dict(color='y', label='mean'),
                                                                        plot=dict(color='k', label='trace')))
        ax = f.get_axes()
        for i, ai in enumerate(ax):
            ai.axhline(y = Q_star.flatten()[i], linestyle=':', linewidth=5, color = 'g',  label = 'true q')
        # reset positions to avoid overlap    
        f.tight_layout()
        handles, labels = ai.get_legend_handles_labels()
        ai.legend(handles, labels, bbox_to_anchor=(2, 0.2), loc='right')
        if show:
            plt.show()
        

        
    
    

    