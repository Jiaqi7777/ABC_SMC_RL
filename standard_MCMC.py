import pyro
import pyro.distributions as dist
import torch
import matplotlib.pyplot as plt
from parameter import *
from MCMC import *
from functools import partial
from QLearning import *


def likelihood(data):
    global r_hat
    prior_parameter = pyro.sample("prior_parameter", dist.MultivariateNormal(torch.zeros(dim), prior_sigma **2 * torch.eye(dim)))
    mean = r_hat(prior_parameter).reshape(-1)
    with pyro.plate("data_plate"):
        pyro.sample("obs", dist.MultivariateNormal(mean, abc_epsilon ** 2 * torch.eye(len(data))), obs=data)


def mcmc(data, prior_parameter, num_samples=MCMC_SAMPLE, warmup_steps=MCMC_T//10):
    pyro.clear_param_store()
    # kernel = pyro.infer.mcmc.HMC(likelihood, step_size=stepsize, adapt_step_size=False, adapt_mass_matrix=True, target_accept_prob=target_accept_prob)
    kernel = pyro.infer.mcmc.NUTS(likelihood, step_size=stepsize, adapt_step_size=False, adapt_mass_matrix=True, target_accept_prob=target_accept_prob)
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
    parser.add_argument('--seed', default=seed, type=int)
    parser.add_argument('--MCMC', default=True, action='store_false', help='Bool type')
    args = parser.parse_args()
    time = args.time
    print('time:', time)
    training_steps = args.training_step
    save = args.save
    show = args.show
    abc_epsilon = args.epsilon
    seed = args.seed
    MCMC_SHOW_DISABLE=args.MCMC

    n_particle = 1
    random.seed(seed)
    pyro.set_rng_seed(seed)
    np.random.seed(seed)
    
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
        for repeat in range(repeat_experiment):
            model = Tabular(env=env, n_particle=n_particle, prior='normal')
            r_all_epi = []
            obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
            s0, _ = env.reset()
            posterior_samples = torch.tensor(model.get_parameter())
            for e in range(episodes):
                env.reset()
                print(f'Episode {e} in repeat {repeat}')
                R = 0
                R_star = 0
                h = 0
                while True:
                # for h in tqdm(range(horizon)):
                    action = model.act(s0, posterior_samples)
                    s1, r, done, *info = env.step(action)
                    #Optimal action
                    R_star = V_star[s0] + gamma * R_star#env.R[tuple(env.P[s0 + (int(pi_star[s0]), )])]
                    R += sum([r * gamma ** i for i in range(h + 1)])
                    obs.insert({'state0': s0, 'state1': s1, 'action': action, 'rewards': r, 'done': done})
                    s0 = s1
                    if ( h + 1)  % FROZEN_T == 0 or done:
                        #MCMC
                        batch_indicies = random.sample(range(min(len(obs._buffers['state0']), buffer_size)), k=batch_size)
                        r_hat = partial(generate_samples, model=model, obs=obs._buffers,  batch_indicies=batch_indicies)
                        mcmc_run = mcmc(torch.tensor(obs._buffers['rewards'])[batch_indicies], torch.tensor(posterior_samples[-1].reshape(-1)), num_samples=training_steps,  warmup_steps=training_steps//10)
                        posterior_samples = mcmc_run.get_samples()["prior_parameter"].reshape((-1, ) + env.n_cell + (env.action_space.n, ))
                        # posterior_samples = new_posterior_samples
                        # model.plot_policy(paras=posterior_samples.numpy(), title=f'policy_T{training_steps}_{time}', additional_info = env.R, save=save, show=show)
                        # model.plot_value(paras=posterior_samples.numpy(), title=f'value_T{training_steps}_{time}')
                        f = mcp.plot_chain_panel(chains=posterior_samples.numpy().reshape(posterior_samples.shape[0], -1)[:, :4], names=env.names,
                                                                        settings=dict(add_pm2std=True,
                                                                        mean=dict(color='b'),
                                                                        plot=dict(color='k')))
                        ax = f.get_axes()
                        for i, ai in enumerate(ax):
                            ai.axhline(y = Q_star.flatten()[i], linestyle=':', color = 'g',  label = 'true q')
                        # reset positions to avoid overlap    
                        f.tight_layout()
                        if show:
                            plt.show()
                    if done:
                        print("done with", h + 1, 'steps')
                        break
                    h += 1
                r_all_epi.append(R_star - R)
                if e % FROZEN_T == 0 :
                    with open(f'Models/MCMC/chains_T{training_steps}_{time}.npy', 'wb') as f:
                        np.save(f, r_all_iter)
                        print('model saved at', f'Models/MCMC/chains_T{training_steps}_{time}.npy')
            r_all_iter.append(r_all_epi)
        with open(f'Models/MCMC/chains_T{training_steps}_{time}.npy', 'wb') as f:
            np.save(f, r_all_iter)
            print('model saved at', f'Models/MCMC/chains_T{training_steps}_{time}.npy')
            
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
        f = mcp.plot_chain_panel(chains=posterior_samples.numpy()[0 * training_steps // 10:, :4],settings=dict(add_pm2std=True,
                                                            mean=dict(color='b'),
                                                            plot=dict(color='k')))
        if show:
            plt.show()
        

        
    
    

    