import numpy as np
import scipy.stats as stats
from copy import deepcopy
from parameter import *
   
def generate_samples(model, obs, para):
    samples = []
    for s0, a, s1 in zip(obs['state0'], obs['action'], obs['state1']):
        samples.append(model.q_value(para, s0, a) - model.gamma * model.v_value(para, s1))
    return np.array(samples)

def get_log_prior(parameters, sigma=1):
        """log(p(para))"""
        parameters = parameters.reshape(-1)
        sigma = sigma * np.identity(len(parameters))
        return stats.multivariate_normal.logpdf(parameters, cov=sigma).sum()

def get_log_likelihood(obs, samples, sigma=1, tractability=False):
    """log(p(evidence|para))"""
    if tractability:
        raise NotImplementedError("Not implemented for tractable likelihood")
        likelihood = stats.norm.pdf(evidence, loc=9.8, scale=sigma)
        log_likelihood = np.log(likelihood).sum()
        return log_likelihood
    else:
        #sigma = sigma * np.identity(len(samples))
        data_length = min(len(obs['rewards']), len(samples))
        lld = stats.norm.logpdf(obs['rewards'][:data_length], loc=samples[:data_length], scale=sigma).sum()
        return lld
        
    
def get_log_posterior(para, *args):
    log_prior = get_log_prior(para)
    log_likelihood = get_log_likelihood(*args)
    return log_prior + log_likelihood

class Kernel:
    def __init__(self, model=None, sigma=0.1, prior=get_log_prior, likelihood=get_log_likelihood, tractability=False):
        self.sigma = sigma
        self.model=model
        self.prior = prior
        self.likelihood = likelihood
        self.tractability = tractability

    def posterior(self, para, *args):
        return self.prior(para) + self.likelihood(*args, tractability=self.tractability)

    def move(self):
        raise NotImplementedError

class RandomWalk(Kernel):
    def __init__(self, *args, model=None, sigma=0.5):
        print('sigma', sigma)
        super(RandomWalk, self).__init__(*args)   
        self.model=model
        self.sigma = sigma

    def move(self, current_para):
        move_ratio = 1
        return current_para + np.random.normal(size=(current_para.shape), scale=self.sigma), move_ratio

    def accept(self, current_para, obs, samples):
        proposed_para, move_ratio = self.move(current_para)
        proposed_samples = generate_samples(self.model, obs, proposed_para)
        #print(proposed_samples-samples)
        current_log_posterior = self.posterior(current_para, obs, samples)
        proposed_log_posterior = self.posterior(proposed_para, obs, proposed_samples)
        return proposed_log_posterior - current_log_posterior - np.log(move_ratio), proposed_para, proposed_samples
        

class MCMC:
    def __init__(self, steps = 10, kernal=RandomWalk(sigma=sigma)):
        self.steps = steps
        self.kernel = kernal
        self.accepted = 0

    def reset(self):
        self.accepted = 0

    def update(self, paras, obs, samples):
        new_paras = deepcopy(paras)
        for i, current_para in enumerate(paras):
            acceptance_ratio, proposed_para, proposed_samples = self.kernel.accept(current_para, obs, samples[:, i])
            if acceptance_ratio > 1:
                accept = True
            else:
                alpha = np.random.uniform(0, 1)
                accept = alpha < np.exp(acceptance_ratio)
            if accept:
                new_paras[i] = proposed_para
                self.accepted += 1
                samples[:, i] = proposed_samples
                # print('accept with ratio ', np.exp(acceptance_ratio))
        return new_paras, samples
        

if __name__ == '__main__':
    from GridWorld import *
    from model import *
    from QLearning import *
    import mcmcplot
    from mcmcplot import mcmcplot as mcp
    from tqdm import tqdm
    training_steps = 1000
    n_particle = 1
    sigma = 0.3
    r = []
    random.seed(10)
    env = GridWorld((3,4), obstacles=True)
    model = Tabular(env=env, n_particle=n_particle, prior='normal')
    kernel = RandomWalk(model=model, sigma=sigma)
    mcmc = MCMC(kernal=kernel)
    chain = np.zeros([training_steps, env.observation_space.n * env.action_space.n])
    
    print(env.reset())
    # env.plot_env()
    env.expert(reset=True)
    # env.plot_env(env.expert_traj + env.R)
    
    obs = env.expert_obs._buffers
    R = [obs['rewards'][0]]
    samples_l = []
    paras = []
    for j in range(len(obs['state0'])):
        samples_l.append(model.r_hat(obs['state0'][j], obs['state1'][j], obs['action'][j]))
    samples_l = np.array(samples_l)
    for t in tqdm(range(training_steps)):
        new_parameter, samples_l = mcmc.update(model.get_parameter(), obs, samples_l)
        model.set_parameter(new_parameter)
        chain[t] = new_parameter.flat
        if t > training_steps * 0.1:
            if t % 10 == 0:
                paras.append(new_parameter)
    print('accepted ratio:', mcmc.accepted / training_steps)
    model.plot_policy(paras=np.array(paras), show=True, additional_info = env.R)
    #mcmc chain plots
    f = mcp.plot_chain_panel(chains=chain[:, :6],settings=dict(add_pm2std=True,
                                                        mean=dict(color='b'),
                                                        plot=dict(color='k')))
    plt.show()
    
    #density panel
    user_settings = dict(
    plot=dict(
        marker='s',
        mfc='none',
        linestyle='none'),
    fig=dict(figsize=(6, 6)))
    names = ['a', 'b', 'c']
    f = mcp.plot_density_panel(
        chains=chain[:, :6],
        names=names,
        settings=user_settings)
    plt.show()
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(chain.shape[1]).columns, fontsize=14, rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()
    #correlation
    settings = dict(
    add_5095_contours=True,
    plot_95=dict(
        color='r',
        linewidth=3),
    plot_50=dict(
        color='c',
        linewidth=3),
    add_legend=True,
    legend=dict(
        loc='upper right',
        fontsize=10,
        bbox_to_anchor=(0.85, 0.75)),
    fig=dict(figsize=(4,4)))
    fp = mcp.plot_pairwise_correlation_panel(
        chains=chain[:, :6],
        settings=settings)
    plt.show()

    
    
        
    