import pyro
import pyro.distributions as dist
import torch
import matplotlib.pyplot as plt
from parameter import *
from MCMC import *
from functools import partial



def likelihood(data):
    global r_hat
    # print(obs._buffers)
    prior_parameter = pyro.sample("prior_parameter", dist.MultivariateNormal(torch.zeros(dim), prior_sigma **2 * torch.eye(dim)))
    mean = r_hat(prior_parameter).reshape(-1)
    with pyro.plate("data_plate"):
        pyro.sample("obs", dist.MultivariateNormal(mean, abc_epsilon ** 2 * torch.eye(len(data))), obs=data)


def mcmc(data, prior_parameter, num_samples=MCMC_T, warmup_steps=MCMC_T//10):
    pyro.clear_param_store()

    kernel = pyro.infer.mcmc.NUTS(likelihood, adapt_step_size=True, adapt_mass_matrix=True)
    mcmc_run = pyro.infer.mcmc.MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, initial_params={'prior_parameter': prior_parameter})
    mcmc_run.run(data)

    return mcmc_run


if __name__ == '__main__':
    from GridWorld import *
    from model import *
    from QLearning import *
    from tqdm import tqdm
    from mcmcplot import mcmcplot as mcp
    # from arviz import ess, plot_autocorr, plot_trace
    import datetime
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--training_step', default=MCMC_T, type=int)
    parser.add_argument('-t', '--time', default=datetime.datetime.now().strftime("%f"))
    parser.add_argument('-s', '--save', default=False)
    parser.add_argument('-p', '--show', default=False)
    parser.add_argument('-e', '--epsilon', default=epsilon, type=float)
    args = parser.parse_args()
    time = args.time
    training_steps = args.training_step
    save = args.save
    show = args.show
    abc_epsilon = args.epsilon
    repeat = 100
    n_particle = 1
    r = []
    random.seed(10)
    
    env = GridWorld((3,4), obstacles=True)
    dim = env.observation_space.n * env.action_space.n
    model = Tabular(env=env, n_particle=n_particle, prior='normal')
    print(env.reset())
    if ONLINE_LEARNING:
        obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
        s0, _ = env.reset()
        posterior_samples = model.get_parameter()
        for h in tqdm(range(horizon)):
            action = model.act(s0, posterior_samples)
            s1, r, done, *info = env.step(action)
            obs.insert({'state0': s0, 'state1': s1, 'action': action, 'rewards': r, 'done': done})
            r_hat = partial(generate_samples, model=model, obs=obs._buffers)
            mcmc_run = mcmc(torch.tensor(obs._buffers['rewards']), torch.tensor(posterior_samples[-1].reshape(-1)), num_samples=training_steps)
            new_posterior_samples = mcmc_run.get_samples()["prior_parameter"].reshape((-1, ) + env.n_cell + (env.action_space.n, ))
            if h // FROZEN_T == 0:
                posterior_samples = new_posterior_samples
            s0 = s1
            if done:
                print("done with", h + 1, 'steps')
                break
            
    else:
        env.uniform_policy()
        obs = env.uniform_obs._buffers
        r_hat = partial(generate_samples, model=model, obs=obs)
        mcmc_run = mcmc(torch.tensor(obs['rewards']), torch.tensor(model.get_parameter()[0].reshape(-1)), num_samples=training_steps)
        posterior_samples = mcmc_run.get_samples()["prior_parameter"]
        print(posterior_samples.shape)

        model.plot_policy(paras=posterior_samples.numpy().reshape((-1, ) + env.n_cell + (env.action_space.n, )), title=f'policy_T{training_steps}_{time}', additional_info = env.R, save=save, show=show)
        # print('ESS:', ess(chain.T))
        f = mcp.plot_chain_panel(chains=posterior_samples.numpy()[training_steps // 10:, :4],settings=dict(add_pm2std=True,
                                                            mean=dict(color='b'),
                                                            plot=dict(color='k')))
        if show:
            plt.show()
        

        
    
    

    