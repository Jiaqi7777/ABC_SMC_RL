import numpy as np
import scipy.stats as stats
from copy import deepcopy
from parameter import *
   
def generate_samples(model, obs, para):
    samples = []
    for s0, a, s1 in zip(obs['state0'], obs['action'], obs['state1']):
        samples.append(model.q_value(para, s0, a) - model.gamma * model.v_value(para, s1))
    return np.array(samples)

class Prior:
    def __init__(self, sigma=1):
        """log(p(para))"""
        self.sigma = sigma 
        
    def get_log_prior(self, parameters):
        parameters = parameters.reshape(-1)
        sigma = self.sigma * np.identity(len(parameters))
        return stats.multivariate_normal.logpdf(parameters, cov=sigma).sum()

def get_log_likelihood(obs, samples, epsilon=1, tractability=False):
    """log(p(evidence|para))"""
    if tractability:
        raise NotImplementedError("Not implemented for tractable likelihood")
        likelihood = stats.norm.pdf(evidence, loc=9.8, scale=epsilon)
        log_likelihood = np.log(likelihood).sum()
        return log_likelihood
    else:
        #epsilon = epsilon * np.identity(len(samples))
        data_length = min(len(obs['rewards']), len(samples))
        lld = stats.norm.logpdf(obs['rewards'][:data_length], loc=samples[:data_length], scale=epsilon).sum()
        return lld
        
    
# def get_log_posterior(para, *args):
#     log_prior = get_log_prior(para)
#     log_likelihood = get_log_likelihood(*args)
#     return log_prior + log_likelihood

class Kernel:
    def __init__(self, model=None, stepsize=0.1, prior=Prior(sigma=1), likelihood=get_log_likelihood, tractability=False):
        self.stepsize = stepsize
        self.model=model
        self.prior = prior
        self.likelihood = likelihood
        self.tractability = tractability

    def posterior(self, para, *args):
        return self.prior.get_log_prior(para) + self.likelihood(*args, tractability=self.tractability)

    def move(self):
        raise NotImplementedError

class RandomWalk(Kernel):
    def __init__(self, *args, model=None, stepsize=0.5):
        print('stepsize', stepsize)
        super(RandomWalk, self).__init__(*args)   
        self.model=model
        self.stepsize = stepsize

    def move(self, current_para):
        move_ratio = 1
        return current_para + np.random.normal(size=(current_para.shape), scale=self.stepsize), move_ratio

    def accept(self, current_para, obs, samples):
        proposed_para, move_ratio = self.move(current_para)
        proposed_samples = generate_samples(self.model, obs, proposed_para)
        #print(proposed_samples-samples)
        current_log_posterior = self.posterior(current_para, obs, samples)
        proposed_log_posterior = self.posterior(proposed_para, obs, proposed_samples)
        return proposed_log_posterior - current_log_posterior - np.log(move_ratio), proposed_para, proposed_samples
    
class pCN(Kernel):
    def __init__(self, *args, model=None, stepsize=0.5):
        print('stepsize', stepsize)
        super(pCN, self).__init__(*args)   
        self.model=model
        self.stepsize = stepsize
        self.sigma = self.prior.sigma
    
    def accept(self, current_para, obs, samples):
        proposed_para = np.sqrt(1 - self.stepsize ** 2) * current_para + self.stepsize * np.random.normal(size=(current_para.shape), scale=self.sigma)
        proposed_samples = generate_samples(self.model, obs, proposed_para)
        current_log_posterior = self.likelihood(obs, samples)
        proposed_log_posterior = self.likelihood(obs, proposed_samples)
        return proposed_log_posterior - current_log_posterior, proposed_para, proposed_samples
    
class MALA(Kernel):
    def __init__(self, *args, model=None, stepsize=0.5):
        print('stepsize', stepsize)
        super(pCN, self).__init__(*args)   
        self.model=model
        self.stepsize = stepsize
        self.sigma = self.prior.sigma
        
    def gradient(self, para):
        return
    
    def move(self, current_para):
        move_ratio = 1
        return np.sqrt(2 * self.stepsize) *  np.random.normal(size=(current_para.shape), scale=self.sigma) + current_para + self.gradient(current_para), move_ratio
    
    def accept(self, current_para, obs, samples):
        proposed_para, move_ratio = self.move(current_para)
        proposed_samples = generate_samples(self.model, obs, proposed_para)
        current_log_posterior = self.posterior(current_para, obs, samples)
        proposed_log_posterior = self.posterior(proposed_para, obs, proposed_samples)
        return proposed_log_posterior - current_log_posterior - np.log(move_ratio), proposed_para, proposed_samples
        

class MCMC:
    def __init__(self, steps = 10, kernal=pCN(stepsize=stepsize)):
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
    from arviz import ess, plot_autocorr, plot_trace
    import datetime
    time =  datetime.datetime.now()
    time = time.strftime("%f")
    training_steps = 1000
    repeat = 100
    n_particle = 1
    stepsize = 0.03
    r = []
    random.seed(10)
    env = GridWorld((3,4), obstacles=True)
    model = Tabular(env=env, n_particle=n_particle, prior='normal')
    kernel = pCN(model=model, stepsize=stepsize)
    mcmc = MCMC(kernal=kernel)
    chain = np.zeros([training_steps, env.observation_space.n * env.action_space.n])
    
    print(env.reset())
    # env.plot_env()
    env.expert(reset=True)
    for _ in range(repeat):
    # env.plot_env(env.expert_traj + env.R)
        env.expert()
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
    print('ESS:', ess(chain[:,0]))
    with open(f'Models/MCMC/chains_T{training_steps}_{time}.npy', 'wb') as f:
        np.save(f, chain)
    #mcmc chain plots
    # f = mcp.plot_chain_panel(chains=chain[training_steps // 10:, :6],settings=dict(add_pm2std=True,
    #                                                     mean=dict(color='b'),
    #                                                     plot=dict(color='k')))
    # plt.show()
    
    #density panel
    # user_settings = dict(
    # plot=dict(
    #     marker='s',
    #     mfc='none',
    #     linestyle='none'),
    # fig=dict(figsize=(6, 6)))
    # names = ['a', 'b', 'c']
    # f = mcp.plot_density_panel(
    #     chains=chain[training_steps // 10:, :6],
    #     names=names,
    #     settings=user_settings)
    # plt.show()

    #correlation
    # settings = dict(
    # add_5095_contours=True,
    # plot_95=dict(
    #     color='r',
    #     linewidth=3),
    # plot_50=dict(
    #     color='c',
    #     linewidth=3),
    # add_legend=True,
    # legend=dict(
    #     loc='upper right',
    #     fontsize=10,
    #     bbox_to_anchor=(0.85, 0.75)),
    # fig=dict(figsize=(4,4)))
    # fp = mcp.plot_pairwise_correlation_panel(
    #     chains=chain[training_steps // 10:, :6],
    #     settings=settings)
    # plt.show()
    figure_path = 'Figures/MCMC/'
    plot_trace(chain[training_steps // 10:, :].T, compact=False)
    # plt.show()
    plt.savefig(f'{figure_path}trace_T{training_steps}_{time}.png')
    plot_autocorr(chain[training_steps // 10 :: training_steps // 100, :].T)
    # plt.show()
    plt.savefig(f'{figure_path}corrT{training_steps}_{time}.png')

    
    
        
    