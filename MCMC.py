import numpy as np
import scipy.stats as stats
from copy import deepcopy
from parameter import *
   
def generate_samples(para, model, obs, batch_indicies=None):
    global batch_size
    para = para.reshape(model.state_size + (model.action_size, ))
    if not batch_training:
        batch_indicies = range(min(len(obs['state0']), buffer_size))
    
    s0 = np.array(obs['state0'][-buffer_size:])[batch_indicies]
    s1 = np.array(obs['state1'][-buffer_size:])[batch_indicies]
    a = np.array(obs['action'][-buffer_size:])[batch_indicies]
    return model.q_value(para, s0.T, a) - model.gamma * model.v_value(para, s1.T).values #time 

    samples = []
    for s0, a, s1 in zip(obs['state0'], obs['action'], obs['state1']):
        samples.append(model.q_value(para, s0, a) - model.gamma * model.v_value(para, s1))
    return np.array(samples)# t



class Prior:
    def __init__(self, sigma=1):
        """log(p(para))"""
        self.sigma = sigma 
        
    def get_log_prior(self, parameters):
        parameters = parameters.reshape(-1)
        sigma = self.sigma * np.identity(len(parameters))
        return stats.multivariate_normal.logpdf(parameters, cov=sigma**2)

class Likelihood:
    """log(p(evidence|para))"""
    def __init__(self, epsilon=epsilon):
        self.epsilon = epsilon 
        
    def get_log_likelihood(self):
        raise NotImplementedError
        
class ABCLikelihood(Likelihood):
    def __init__(self, epsilon=epsilon):
        super().__init__(epsilon)
        print('abc epsilon', epsilon)
    
    def get_log_likelihood(self, obs, samples, tractability=False):
        """log(p(evidence|para))"""
        if tractability:
            raise NotImplementedError("Not implemented for tractable likelihood")
            likelihood = stats.norm.pdf(evidence, loc=9.8, scale=epsilon)
            log_likelihood = np.log(likelihood).sum()
            return log_likelihood
        else:
            #epsilon = epsilon * np.identity(len(samples))
            data_length = min(len(obs['rewards']), len(samples))
            lld = stats.norm.logpdf(obs['rewards'][:data_length], loc=samples[:data_length], scale=self.epsilon).sum()
            return lld
        
    
# def get_log_posterior(para, *args):
#     log_prior = get_log_prior(para)
#     log_likelihood = get_log_likelihood(*args)
#     return log_prior + log_likelihood

class Kernel:
    def __init__(self, model=None, stepsize=0.1, prior=Prior(sigma=prior_sigma), likelihood=ABCLikelihood(epsilon=epsilon), tractability=False):
        self.stepsize = stepsize
        self.model=model
        self.prior = prior
        self.likelihood = likelihood
        self.tractability = tractability

    def posterior(self, para, *args):
        return self.prior.get_log_prior(para) + self.likelihood.get_log_likelihood(*args, tractability=self.tractability)

    def move(self):
        raise NotImplementedError

class RandomWalk(Kernel):
    def __init__(self, *args, model=None, stepsize=0.5):
        print('RandomWalk stepsize', stepsize)
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
        print('pCN stepsize', stepsize)
        super(pCN, self).__init__(*args)   
        self.model=model
        self.stepsize = stepsize
        self.sigma = self.prior.sigma
        
    def move(self, current_para):
        return np.sqrt(1 - self.stepsize ** 2) * current_para + self.stepsize * np.random.normal(size=(current_para.shape), scale=self.sigma)
    
    def accept(self, current_para, obs, samples):
        proposed_para = self.move(current_para)
        proposed_samples = generate_samples(self.model, obs, proposed_para)
        current_log_posterior = self.likelihood.get_log_likelihood(obs, samples)
        proposed_log_posterior = self.likelihood.get_log_likelihood(obs, proposed_samples)
        return proposed_log_posterior - current_log_posterior, proposed_para, proposed_samples
    
class MALA(Kernel):
    def __init__(self, *args, model=None, stepsize=0.5):
        print('MALA stepsize', stepsize)
        super(MALA, self).__init__(*args)   
        self.model=model
        self.stepsize = stepsize
        self.sigma = self.prior.sigma
        
    def gradient(self, para, obs, samples):
        s0 = obs['state0']
        s1 = obs['state1']
        a = obs['action']
        s01, s02 = np.array(s0).T
        s11, s12 = np.array(s1).T
        a_prime = np.argmax(para[s11, s12], axis=-1)
        Indicator = np.zeros(shape=(len(samples), ) + para.shape) #TxTheta
        Indicator[range(len(samples)), s01, s02, a] = 1
        Indicator[range(len(samples)), s11, s12, a_prime] -= self.model.gamma
        Sum = np.matmul(Indicator.T, (np.array(obs['rewards']) - samples)).T #Theta x T, Tx1
        return - para / self.sigma ** 2 + 1 / self.likelihood.epsilon ** 2 * Sum
    
    def move(self, current_para, obs, samples):
        current_gradient = self.gradient(current_para, obs, samples)
        proposed_para = np.sqrt(2 * self.stepsize) *  np.random.normal(size=(current_para.shape), scale=1) + current_para + self.stepsize * current_gradient
        proposed_gradient = self.gradient(proposed_para, obs, samples)
        move_ratio = stats.norm.logpdf(proposed_para, loc=current_para + self.stepsize * current_gradient, scale=np.sqrt(2*self.stepsize)).sum() - \
                                                                    stats.norm.logpdf(current_para, loc=proposed_para + self.stepsize * proposed_gradient, scale=np.sqrt(2*self.stepsize)).sum()
        # plt.imshow(proposed_gradient.reshape(3,16))
        # plt.show()
        # print('move ratio')
        # print(stats.norm.logpdf(proposed_para, loc=current_para + self.stepsize * current_gradient, scale=2*self.stepsize).sum(), stats.norm.logpdf(current_para, loc=proposed_para + self.stepsize * proposed_gradient, scale=2*self.stepsize).sum())
        return proposed_para, move_ratio
    
    def accept(self, current_para, obs, samples):
        proposed_para, move_ratio = self.move(current_para, obs, samples)
        proposed_samples = generate_samples(self.model, obs, proposed_para)
        current_log_posterior = self.posterior(current_para, obs, samples)
        proposed_log_posterior = self.posterior(proposed_para, obs, proposed_samples)
        # print(proposed_log_posterior, current_log_posterior,  - move_ratio)
        return proposed_log_posterior - current_log_posterior - move_ratio, proposed_para, proposed_samples
        
class AM(Kernel):
    def __init__(self, model=None, stepsize=0.1, prior=Prior(sigma=prior_sigma), likelihood=ABCLikelihood(epsilon=epsilon), tractability=False, sd=1, am_epsilon=1e-5):
        super().__init__(model, stepsize, prior, likelihood, tractability)
        self.sd = sd
        self.am_epsilon=am_epsilon
        
    def move(self, current_para, para_history):
        shape = current_para.shape
        # print(current_para)
        current_para = current_para.reshape(-1)
        if para_history == []:
            paras = [current_para]
        else:
            paras = np.array(para_history).reshape(len(para_history), -1)
        current_cov = self.cov(paras)
        proposed_para = current_para + np.random.multivariate_normal(current_para, cov=current_cov)
        paras[-1] = proposed_para
        proposed_cov = self.cov(paras)
        move_ratio = stats.multivariate_normal.logpdf(proposed_para, mean=current_para, cov=current_cov) - \
                                        stats.multivariate_normal.logpdf(current_para, mean=proposed_para, cov=proposed_cov)
        return proposed_para.reshape(shape), move_ratio                                
        
    def cov(self, para_history):
        if len(para_history) == 1:
            return self.stepsize ** 2 * np.eye(len(para_history[0]))
        return self.sd * np.cov(para_history, rowvar=False) + self.sd * self.am_epsilon * np.eye(len(para_history[0]))
        
    def accept(self, current_para, obs, samples, para_history):
        proposed_para, move_ratio = self.move(current_para, para_history)
        proposed_samples = generate_samples(self.model, obs, proposed_para)
        current_log_posterior = self.posterior(current_para, obs, samples)
        proposed_log_posterior = self.posterior(proposed_para, obs, proposed_samples)
        return proposed_log_posterior - current_log_posterior - move_ratio, proposed_para, proposed_samples
        

class MCMC:
    def __init__(self, steps = 10, kernal=pCN(stepsize=stepsize)):
        self.steps = steps
        self.kernel = kernal
        self.accepted = 0

    def reset(self):
        self.accepted = 0

    def update(self, paras, obs, samples, para_history=[]):
        new_paras = deepcopy(paras)
        for i, current_para in enumerate(paras):
            acceptance_ratio, proposed_para, proposed_samples = self.kernel.accept(current_para, obs, samples[:, i])#para_history for AM
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
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--training_step', default=MCMC_T, type=int)
    parser.add_argument('-t', '--time', default=datetime.datetime.now().strftime("%f"))
    parser.add_argument('-s', '--save', default=False)
    parser.add_argument('-p', '--show', default=False)
    args = parser.parse_args()
    time = args.time
    training_steps = args.training_step
    save = args.save
    show = args.show
    repeat = 100
    n_particle = 1
    r = []
    random.seed(10)
    env = GridWorld((3,4), obstacles=True)
    model = Tabular(env=env, n_particle=n_particle, prior='normal')
    kernel = pCN(model=model, stepsize=stepsize)
    mcmc = MCMC(kernal=kernel)
    chain = np.zeros([training_steps, env.observation_space.n * env.action_space.n])
    
    print(env.reset())
    # env.plot_env()
    # env.expert(reset=True)
    # for _ in range(repeat):
    #     env.expert(reset=False)
    env.uniform_policy()
    obs = env.uniform_obs._buffers
    R = [obs['rewards'][0]]
    samples_l = []
    paras_history = []
    for j in range(len(obs['state0'])):
        samples_l.append(model.r_hat(obs['state0'][j], obs['state1'][j], obs['action'][j]))
    samples_l = np.array(samples_l)
    for t in tqdm(range(training_steps)):
        new_parameter, samples_l = mcmc.update(model.get_parameter(), obs, samples_l)#paras_history for AM
        model.set_parameter(new_parameter)
        chain[t] = new_parameter.flatten()
        if t > training_steps * BURN_IN:
            if t % skip == 0:
                paras_history.append(new_parameter)
    print('accepted ratio:', mcmc.accepted / training_steps)
    model.plot_policy(paras=np.array(paras_history), title=f'policy_T{training_steps}_{time}', additional_info = env.R, save=save, show=show)
    print('ESS:', ess(chain.T))

    if save:
        with open(f'Models/MCMC/chains_T{training_steps}_{time}.npy', 'wb') as f:
            np.save(f, chain)
            print('model saved at', f'Models/MCMC/chains_T{training_steps}_{time}.npy')
    figure_path = 'Figures/MCMC/'
    #mcmc chain plots
    f = mcp.plot_chain_panel(chains=chain[training_steps // 10:, :4],settings=dict(add_pm2std=True,
                                                        mean=dict(color='b'),
                                                        plot=dict(color='k')))
    if show:
        plt.show()
    if save:
        plt.savefig(f'{figure_path}traces_T{training_steps}_{time}')
        print('Figure saved at ', f'{figure_path}traces_T{training_steps}_{time}')
        plt.clf()
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

    plot_trace(chain[training_steps // 10:, :].T, compact=False)
    if show:
        plt.show()
    if save:
        plt.savefig(f'{figure_path}trace_T{training_steps}_{time}.png')
        print('Figure saved at ', f'{figure_path}trace_T{training_steps}_{time}')
        plt.clf()
    corr_chain = chain[training_steps // 10 :: training_steps // 100, :].T
    plot_autocorr(corr_chain, max_lag=min(200, len(corr_chain)))
    if show:
        plt.show()
    if save:
        plt.savefig(f'{figure_path}corr_T{training_steps}_{time}.png')
        print('Figure saved at ', f'{figure_path}corr_T{training_steps}_{time}')
        plt.clf()

    
    
        
    