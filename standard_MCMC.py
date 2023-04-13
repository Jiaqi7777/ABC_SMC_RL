import pyro
import pyro.distributions as dist
import torch
import matplotlib.pyplot as plt
from parameter import *
from MCMC import *
from functools import partial
from QLearning import *


def likelihood(data):
    obs, model, dim, batch_indicies, abc_epsilon = data
    if not BATCH_TRAINING:
        batch_indicies = range(min(len(obs['state0']), BUFFER_SIZE))
    r_hat = torch.tensor(obs['rewards'])[-BUFFER_SIZE:][batch_indicies]
    
    prior_parameter = pyro.sample("prior_parameter", dist.MultivariateNormal(torch.zeros(dim), prior_sigma **2 * torch.eye(dim)))
    mean = generate_samples(prior_parameter, model, obs, batch_indicies=batch_indicies).reshape(-1)
    with pyro.plate("data_plate"):
        pyro.sample("obs", dist.MultivariateNormal(mean, abc_epsilon ** 2 * torch.eye(len(r_hat))), obs=r_hat)


def mcmc(data, prior_parameter, num_samples=MCMC_SAMPLE, warmup_steps=MCMC_T//10, MCMC_SHOW_DISABLE=MCMC_SHOW_DISABLE, stepsize=stepsize):
    pyro.clear_param_store()
    kernel = pyro.infer.mcmc.HMC(likelihood, full_mass=full_mass, step_size=stepsize, adapt_step_size=adapt_step_size, adapt_mass_matrix=adapt_mass_matrix, target_accept_prob=target_accept_prob)
    # kernel = pyro.infer.mcmc.HMC(likelihood, full_mass=True, step_size=stepsize, adapt_step_size=adapt_step_size, adapt_mass_matrix=adapt_mass_matrix, target_accept_prob=target_accept_prob)
    mcmc_run = pyro.infer.mcmc.MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, initial_params={'prior_parameter': prior_parameter}, disable_progbar=MCMC_SHOW_DISABLE)
    mcmc_run.run(data)

    return mcmc_run


if __name__ == '__main__':
    from GridWorld import *
    from model import *
    from QLearning import *
    from tqdm import tqdm
    from mcmcplot import mcmcplot as mcp
    from arviz import ess, plot_autocorr, plot_trace
    import datetime
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--training_step', default=MCMC_T, type=int)
    parser.add_argument('-t', '--time', default=datetime.datetime.now().strftime("%f"))
    parser.add_argument('-s', '--save', default=save)
    parser.add_argument('-p', '--show', default=show)
    parser.add_argument('-e', '--epsilon', default=epsilon, type=float)
    parser.add_argument('--seed', default=SEED, type=int)
    parser.add_argument('--MCMC', default=True, action='store_false', help='Bool type')
    parser.add_argument('-g', '--Greedy', default=GREEDY, action='store_true', help='Bool type')
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
    warmup_steps = int(training_steps * warmup_ratio)
    GREEDY = args.Greedy

    n_particle = 10
    random.seed(seed)
    pyro.set_rng_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = GridWorld((3,4), obstacles=True)
    dim = env.observation_space.n * env.action_space.n
    model = Tabular(env=env, n_particle=n_particle, prior='normal')
    
    S = []
    for i in range(env.n_cell[0]):
        for j in range(env.n_cell[1]):
            S.append((i,j)) 
    Q = np.ones(shape=(env.n_cell + (env.action_space.n, )))/env.observation_space.n / env.action_space.n
    A = range(env.action_space.n)
    pi_star, Q_star, V_star = DynamicProgramming(Q, A, S, env)
    
    env.reset()
    results = []
    if ONLINE_LEARNING:
        r_all_iter = []
        for repeat in range(REPEAT_EXPERIMENT):
            model = Tabular(env=env, n_particle=n_particle, prior='normal')
            r_all_epi = []
            obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
            s0, _ = env.reset()
            posterior_samples = torch.tensor(model.get_parameter())
            for e in range(EPISODES):
                env.reset()
                print(f'Episode {e} in repeat {repeat}')
                R = 0
                R_star = 0
                h = 0
                # while True: #Turn on h += 1
                for h in tqdm(range(HORIZON)):
                    action = model.act(s0, posterior_samples, GREEDY=GREEDY)
                    s1, r, done, *info = env.step(action)
                    #Optimal action
                    # R_star = V_star[s0] + gamma * R_star#env.R[tuple(env.P[s0 + (int(pi_star[s0]), )])] #for regret
                    R += r#sum([r * gamma ** i for i in range(h + 1)])
                    obs.insert({'state0': s0, 'state1': s1, 'action': action, 'rewards': r, 'done': done})
                    s0 = s1
                    # if e==0 and h==0:
                    #     print(posterior_samples[:, 0, 0], h, posterior_samples.shape)
                    #     print(np.argmax(posterior_samples[:, 0, 0], axis=-1))
                    
                    if ( h + 1)  % FROZEN_T == 0 or done:
                        print(obs._buffers['state0'][-FROZEN_T:])
                        #MCMC
                        batch_indicies = random.sample(range(min(len(obs._buffers['state0']), BUFFER_SIZE)), k=min(BATCH_SIZE, len(obs._buffers['state0'])))
                        r_hat = partial(generate_samples, model=model, obs=obs._buffers,  batch_indicies=batch_indicies)
                        if BATCH_TRAINING:
                            mcmc_run = mcmc([obs._buffers, model, dim, batch_indicies, abc_epsilon], torch.tensor(posterior_samples[-1].reshape(-1)), num_samples=training_steps,  warmup_steps=training_steps//5, MCMC_SHOW_DISABLE=MCMC_SHOW_DISABLE)
                        else:
                            mcmc_run = mcmc([obs._buffers, model, dim, batch_indicies, abc_epsilon], torch.tensor(posterior_samples[-1].reshape(-1)), num_samples=training_steps,  warmup_steps=warmup_steps, MCMC_SHOW_DISABLE=MCMC_SHOW_DISABLE)
                        posterior_samples = mcmc_run.get_samples()["prior_parameter"].reshape((-1, ) + env.n_cell + (env.action_space.n, ))
                        stepsize *= decreasing_factor
                        # posterior_samples = new_posterior_samples
                        model.plot_policy(paras=posterior_samples.numpy(), title=f'policy_T{training_steps}_{time}', additional_info = env.R, save=save, show=show)
                        # model.plot_value(paras=posterior_samples.numpy(), title=f'value_T{training_steps}_{time}')
                        print('Mean Q values', torch.round(torch.mean(posterior_samples, 0), decimals=2))
                        data_plot = posterior_samples.numpy().reshape(posterior_samples.shape[0], -1)[:, :8]
                        f = mcp.plot_chain_panel(chains=data_plot, names=env.names,
                                                                        settings=dict(add_pm2std=True, fig=dict(figsize=(10,10), dpi=250),
                                                                        mean=dict(color='y', label='mean'),
                                                                        plot=dict(color='k', label='trace')))
                        ax = f.get_axes()
                        for i, ai in enumerate(ax):
                            ai.axhline(y = Q_star.flatten()[i], linestyle=':', linewidth=5, color = 'g',  label = 'true q')
                            q = np.percentile(data_plot[:, i], [plot_threshold, 100 - plot_threshold])
                            ai.set_ylim(q)   
                        f.tight_layout()
                        handles, labels = ai.get_legend_handles_labels()
                        ai.legend(handles, labels, bbox_to_anchor=(2, 0.2), loc='right')
                        if show:
                            plt.show()
                    if done:
                        print("done with", h + 1, 'steps')
                        break
                    # h += 1
                r_all_epi.append(R)
                with open(f'Returns/MCMC/T{training_steps}_Gdy{GREEDY}_Ep{epsilon}_Stp{initial_stepsize}_Dcrs{decreasing_factor}_{time}.npy', 'wb') as f:
                    np.save(f, r_all_epi)
                    print(f'EPISODES return for repeat {repeat} saved at', f'Returns/MCMC/T{training_steps}_Gdy{GREEDY}_Ep{epsilon}_Stp{initial_stepsize}_Dcrs{decreasing_factor}_{time}.npy')
            r_all_iter.append(r_all_epi)
        with open(f'Returns/MCMC/T{training_steps}_Gdy{GREEDY}_Ep{epsilon}_Stp{initial_stepsize}_Dcrs{decreasing_factor}_{time}.npy', 'wb') as f:
            np.save(f, r_all_iter)
            print('return saved at', f'Returns/MCMC/T{training_steps}_Gdy{GREEDY}_Ep{epsilon}_Stp{initial_stepsize}_Dcrs{decreasing_factor}_{time}.npy')
            
        plt.plot(R)
        if show:
            plt.show()
            
    else:
        env.uniform_policy()
        obs = env.uniform_obs._buffers
        r_hat = partial(generate_samples, model=model, obs=obs)
        mcmc_run = mcmc(torch.tensor(obs['rewards']), torch.tensor(model.get_parameter()[0].reshape(-1)), num_samples=training_steps, warmup_steps=training_steps//10)
        posterior_samples = mcmc_run.get_samples()["prior_parameter"]
        print(posterior_samples.shape)

        model.plot_policy(paras=posterior_samples.numpy().reshape((-1, ) + env.n_cell + (env.action_space.n, )), title=f'policy_T{training_steps}_{time}', additional_info = env.R, save=save, show=show)
        # print('ESS:', ess(chain.T))
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
        

        
    
    

    