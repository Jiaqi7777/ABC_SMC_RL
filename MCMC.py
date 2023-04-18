import numpy as np
import scipy.stats as stats
import torch
from copy import deepcopy
from parameter import *
from functools import partial
import pyro
from tqdm.notebook import tqdm

def generate_samples(para, model, obs, batch_indices=None, buffer_size=BUFFER_SIZE, batch_training=BATCH_TRAINING):
    para = para.reshape(model.state_size + (model.action_size, ))
    if not batch_training:
        batch_indices = range(min(len(obs['state0']), buffer_size))
    
    s0 = np.array(obs['state0'])[-buffer_size:][batch_indices]
    s1 = np.array(obs['state1'])[-buffer_size:][batch_indices]
    a = np.array(obs['action'])[-buffer_size:][batch_indices]
    dones = torch.tensor(np.array(obs['done'])[-buffer_size:][batch_indices].astype(int))

    return model.q_value(para, s0.T, a) - torch.where(dones == 1, torch.zeros(len(s0)), model.gamma * model.v_value(para, s1.T).values) #time 

    # samples = []
    # for s0, a, s1 in zip(obs['state0'], obs['action'], obs['state1']):
    #     samples.append(model.q_value(para, s0, a) - model.gamma * model.v_value(para, s1))
    # return np.array(samples)# t



class IsotropicGaussianPrior:
    def __init__(self, sd=1., mean=0):
        """log(p(para))"""
        self.sigma = np.float(sd)
        self.mean = mean
        
    def logprior(self, parameter, return_gradient=False):
        return torch.distributions.normal.Normal(loc=0., scale=self.sigma).log_prob(parameter).sum(axis=-1)
    
    def logprior_gradient(self, parameter):
        return - parameter / self.sigma ** 2


class Likelihood:
    """log(p(evidence|para))"""
    def __init__(self, epsilon=EPSILON):
        self.epsilon = epsilon 
        
    def get_log_likelihood(self):
        raise NotImplementedError
        
# class ABCLikelihood(Likelihood):
#     def __init__(self, epsilon=EPSILON):
#         super().__init__(epsilon)
#         print('abc epsilon', epsilon)
    
#     def get_log_likelihood(self, obs, samples, tractability=False):
#         """log(p(evidence|para))"""
#         if tractability:
#             raise NotImplementedError("Not implemented for tractable likelihood")
#             likelihood = stats.norm.pdf(evidence, loc=9.8, scale=epsilon)
#             log_likelihood = np.log(likelihood).sum()
#             return log_likelihood
#         else:
#             #epsilon = epsilon * np.identity(len(samples))
#             data_length = min(len(obs['rewards']), len(samples))
#             lld = stats.norm.logpdf(obs['rewards'][:data_length], loc=samples[:data_length], scale=self.epsilon).sum()
#             return lld
        
class GaussianABCLikelihood():
    def __init__(self, epsilon=EPSILON):
        self.epsilon = epsilon

    def llh(self, data, parameter=None, para_extra_info=None, llh_transform_fn=None): #standard form
        return self.llh_(data=data, parameter=parameter, mean=para_extra_info, mean_fn=llh_transform_fn)


    def llh_(self, data, parameter=None, mean=None, mean_fn=None):
        """if mean is provided, mean_fn is not used"""
        assert mean is not None or mean_fn is not None, "either mean or mean_fn of the form mean_fn(parameter) -> mean should be provided"
        if mean is None:
            mean = mean_fn(parameter)
        llh = torch.distributions.normal.Normal(loc=mean,scale=self.epsilon).log_prob(data).sum()

        return llh, mean

    def llh_gradient(self, data, parameter, para_extra_info=None, llh_transform_fn=None, llh_transform_grad_fn=None): #standard form
        return self.llh_gradient_(data=data, parameter=parameter, mean=para_extra_info, mean_fn=llh_transform_fn, mean_jacobian_fn=llh_transform_grad_fn)

    def llh_gradient_(self, data, parameter, mean=None, mean_fn=None, mean_jacobian_fn=None):
        
        assert mean is not None or mean_fn is not None, "either mean or mean_fn of the form mean_fn(parameter) -> mean should be provided"

        if mean is None:
            mean = mean_fn(parameter)

        mean_jacobian = mean_jacobian_fn(data=data, parameter=parameter)
        gradient = 1/ self.epsilon **2 *  torch.linalg.matmul(torch.t(mean_jacobian),(data - mean))

        return gradient

# def get_log_posterior(para, *args):
#     log_prior = get_log_prior(para)
#     log_likelihood = get_log_likelihood(*args)
#     return log_prior + log_likelihood

class DeterministicSRModel():
    def __init__(self, prior, abclikelihood, llh_transform_fn, llh_transform_grad_fn=None, *args):
        self.prior = prior
        self.abclikelihood = abclikelihood
        self.llh_transform_fn = llh_transform_fn
        self.llh_transform_grad_fn  = llh_transform_grad_fn

    def logprior(self, parameter):
        return self.prior.logprior(parameter=parameter)

    def llh(self, data, parameter, para_extra_info=None):
        llh, para_extra_info = self.abclikelihood.llh(data=data, parameter=parameter, para_extra_info=para_extra_info, llh_transform_fn=self.llh_transform_fn)
        return llh, para_extra_info

    def logtarget_density(self, data, parameter, para_extra_info):
        logprior = self.logprior(parameter=parameter)
        llh, para_extra_info = self.llh(data=data, parameter=parameter, para_extra_info=para_extra_info)
        return logprior + llh, para_extra_info
    
    def logtarget_gradient(self, data, parameter, para_extra_info):
        logprior_grad = self.prior.logprior_gradient(parameter=parameter)
        llh_grad = self.abclikelihood.llh_gradient(data=data, parameter=parameter, para_extra_info=para_extra_info, llh_transform_fn=self.llh_transform_fn, llh_transform_grad_fn=self.llh_transform_grad_fn)
        return logprior_grad + llh_grad


class Kernel:
    def __init__(self, model):
        self.model = model

    # def posterior(self, para, *args):
    #     return self.prior.get_log_prior(para) + self.likelihood.get_log_likelihood(*args, tractability=self.tractability)

    # def move(self):
    #     raise NotImplementedError

class RandomWalk(Kernel):
    def __init__(self, model, stepsize=0.5, covariance_matrix=None, *args):
        print('RandomWalk stepsize', stepsize)
        super(RandomWalk, self).__init__(model=model)   
        self.stepsize = stepsize
        self.covariance_matrix = covariance_matrix

    def move(self, current_para):
        if self.covariance_matrix is not None:
            return torch.distributions.multivariate_normal.MultivariateNormal(loc=current_para, covariance_matrix=np.sqrt(self.stepsize) * self.covariance_matrix).sample()
        else:
            return torch.normal(mean=current_para, std=self.stepsize)

    def propose_accept(self, current_para, data, current_para_extra_info=None):
        proposed_para = self.move(current_para)

        current_logtarget_density, _ = self.model.logtarget_density(data=data, parameter=current_para, para_extra_info=current_para_extra_info)
        proposed_logtarget_density, proposed_para_extra_info = self.model.logtarget_density(data=data, parameter=proposed_para, para_extra_info=None)
        accept_prob = proposed_logtarget_density - current_logtarget_density

        return accept_prob, proposed_para, proposed_para_extra_info #acceptance_prob, proposal, extra_stat_for_proposal
    
class pCN(Kernel):
    def __init__(self, model, stepsize=0.5, *args):
        print('pCN stepsize', stepsize)
        super(pCN, self).__init__(model=model)   
        self.stepsize = stepsize
        self.sigma = self.model.prior.sigma
        
    def move(self, current_para):
        return np.sqrt(1 - self.stepsize ** 2) * current_para + self.stepsize * torch.normal(mean=torch.zeros(current_para.size()), std=self.sigma)
    
    def propose_accept(self, current_para, data, current_para_extra_info=None):
        proposed_para = self.move(current_para)

        current_llh, _ = self.model.llh(data=data, parameter=current_para, para_extra_info=current_para_extra_info)
        proposed_llh, proposed_para_extra_info = self.model.llh(data=data, parameter=proposed_para, para_extra_info=None)
        accept_prob = proposed_llh - current_llh

        return accept_prob, proposed_para, proposed_para_extra_info #acceptance_prob, proposal, extra_stat_for_proposal
    
class MALA(Kernel):
    def __init__(self, model, stepsize=0.5, precondition_matrix=None, use_autograd=True, *args):
        print('MALA stepsize', stepsize)
        super(MALA, self).__init__(model=model)   
        self.stepsize = stepsize
        #self.sigma = self.prior.sigma
        self.precondition = precondition_matrix
        #self.gradient = gradient_fn #should take parameter, data as input, gradient as output
        self.use_autograd = use_autograd

    def move(self, current_para, current_gradient):
        if self.precondition is None:
            proposed_para = current_para + self.stepsize * current_gradient + np.sqrt(2 * self.stepsize) *  torch.normal(mean=torch.zeros(current_para.size()), std=1.)
        else:
            noise = torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.zeros(current_para.size()), covariance_matrix=self.precondition).sample()
            proposed_para = current_para + self.stepsize * (self.precondition @ current_gradient) + np.sqrt(2 * self.stepsize) * noise
        return proposed_para

    def propose_accept(self, current_para, data, current_para_extra_info=None):

        if self.use_autograd:
            #print("using autograd")
            current_para_density_info = current_para_extra_info["density_info"] if current_para_extra_info is not None else None

            if current_para_extra_info is not None:
                current_gradient = current_para_extra_info["gradient"]
                current_logtarget_density, _ = self.model.logtarget_density(data=data, parameter=current_para, para_extra_info=current_para_density_info)
            else:
                current_para.requires_grad = True
                current_logtarget_density, _ = self.model.logtarget_density(data=data, parameter=current_para, para_extra_info=current_para_density_info)
                current_logtarget_density.backward()
                current_gradient = current_para.grad.clone()
                current_para.grad.zero_()
                current_para.requires_grad = False

            proposed_para = self.move(current_para=current_para, current_gradient=current_gradient)
            proposed_para = proposed_para.detach().requires_grad_(True)

            proposed_logtarget_density, proposed_para_density_info = self.model.logtarget_density(data=data, parameter=proposed_para, para_extra_info=None)

            proposed_logtarget_density.backward()
            proposed_gradient = proposed_para.grad.clone()
            proposed_para.grad.zero_()
            proposed_para.requires_grad = False

        else:
            #print("using manual gradient")
            current_para_density_info = current_para_extra_info["density_info"] if current_para_extra_info is not None else None
            current_logtarget_density, current_para_density_info = self.model.logtarget_density(data=data, parameter=current_para, para_extra_info=current_para_density_info)
            if current_para_extra_info is not None:
                current_gradient = current_para_extra_info["gradient"]
            else:
                current_gradient = self.model.logtarget_gradient(data=data, parameter=current_para, para_extra_info=current_para_density_info)

            proposed_para = self.move(current_para=current_para, current_gradient=current_gradient)

            proposed_logtarget_density, proposed_para_density_info = self.model.logtarget_density(data=data, parameter=proposed_para, para_extra_info=None)
            proposed_gradient = self.model.logtarget_gradient(parameter=proposed_para, data=data, para_extra_info=proposed_para_density_info)
        # print("current_grad", current_gradient)
        # print("proposed_gradient", proposed_gradient)

        move_ratio = torch.distributions.normal.Normal(loc=current_para + self.stepsize * current_gradient, scale=np.sqrt(2*self.stepsize)).log_prob(proposed_para).sum() - \
            torch.distributions.normal.Normal(loc=proposed_para + self.stepsize * proposed_gradient, scale=np.sqrt(2*self.stepsize)).log_prob(current_para).sum()
        accept_prob = proposed_logtarget_density - current_logtarget_density - move_ratio

        proposed_para_extra_info = {"density_info": proposed_para_density_info, "gradient": proposed_gradient}

        return accept_prob, proposed_para, proposed_para_extra_info #acceptance_prob, proposal, extra_stat_for_proposal
    
    # def inverse_riemann_mass(self, Indicator):
    #     Indicator_flatten = Indicator.reshape(Indicator.shape[0],-1)
    #     fisher = 1 / (self.likelihood.epsilon ** 2) * Indicator_flatten.T @ Indicator_flatten
    #     fisher +=  self.prior.sigma ** 2 * np.identity(fisher.shape[0])
    #     pass

        
# class AM(Kernel):
#     def __init__(self, model=None, stepsize=0.1, prior=Prior(sigma=PRIOR_SIGMA), likelihood=ABCLikelihood(epsilon=EPSILON), tractability=False, sd=1, am_epsilon=1e-5):
#         super().__init__(model, stepsize, prior, likelihood, tractability)
#         self.sd = sd
#         self.am_epsilon=am_epsilon
        
#     def move(self, current_para, para_history):
#         shape = current_para.shape
#         # print(current_para)
#         current_para = current_para.reshape(-1)
#         if para_history == []:
#             paras = [current_para]
#         else:
#             paras = np.array(para_history).reshape(len(para_history), -1)
#         current_cov = self.cov(paras)
#         proposed_para = current_para + np.random.multivariate_normal(current_para, cov=current_cov)
#         paras[-1] = proposed_para
#         proposed_cov = self.cov(paras)
#         move_ratio = stats.multivariate_normal.logpdf(proposed_para, mean=current_para, cov=current_cov) - \
#                                         stats.multivariate_normal.logpdf(current_para, mean=proposed_para, cov=proposed_cov)
#         return proposed_para.reshape(shape), move_ratio                                
        
#     def cov(self, para_history):
#         if len(para_history) == 1:
#             return self.stepsize ** 2 * np.eye(len(para_history[0]))
#         return self.sd * np.cov(para_history, rowvar=False) + self.sd * self.am_epsilon * np.eye(len(para_history[0]))
        
#     def accept(self, current_para, obs, samples, batch_indices, para_history):
#         proposed_para, move_ratio = self.move(current_para, para_history)
#         proposed_samples = generate_samples(proposed_para, self.model, obs, batch_indices)
#         current_log_posterior = self.posterior(current_para, obs, samples)
#         proposed_log_posterior = self.posterior(proposed_para, obs, proposed_samples)
#         return proposed_log_posterior - current_log_posterior - move_ratio, proposed_para, proposed_samples
        

class MCMC:
    def __init__(self, kernel, num_samples=MCMC_SAMPLE, initial_params=None, params_dim=None):
        self.num_samples = num_samples
        self.kernel = kernel
        assert initial_params is not None or params_dim is not None, "Should either specify initial_params or params_dim"
        if initial_params is not None:
            self.initial_params = initial_params
            self.params_dim = len(initial_params)
        else:
            self.initial_params = torch.tensor(params_dim)
            self.params_dim = params_dim


    #def run(self, paras, obs, samples, batch_indices=None):
    def run(self, data):
        accepted = 0
        samples = torch.zeros((self.num_samples, self.params_dim))

        current_para = self.initial_params if self.initial_params is not None else torch.zeros(self.params_dim)
        current_para_extra_info = None

        pbar = tqdm(range(self.num_samples))
        for i in pbar:
            accept_prob, proposed_para, proposed_para_extra_info  = self.kernel.propose_accept(current_para=current_para, 
                                                                         data=data, 
                                                                         current_para_extra_info=current_para_extra_info)

            if np.log(np.random.uniform(0,1)) < accept_prob:
                current_para = proposed_para
                current_para_extra_info = proposed_para_extra_info
                accepted += 1

            pbar.set_description("Acceptance probability {}".format(np.round(accepted/(i+1), 2)))

            samples[i] = (current_para)

        return samples

        #     acceptance_ratio, proposed_para, proposed_samples = self.kernel.accept(current_para, obs, samples[:, i], batch_indices)#para_history for AM
        #     if acceptance_ratio > 1:
        #         accept = True
        #     else:
        #         alpha = np.random.uniform(0, 1)
        #         accept = alpha < np.exp(acceptance_ratio)
        #     if accept:
        #         new_paras[i] = proposed_para
        #         self.accepted += 1
        #         samples[batch_indices, i] = proposed_samples
        # # print('accept with ratio ', np.exp(acceptance_ratio))
        
        # # print([s for s in torch.chunk(samples, samples.size(0))])
        # return new_paras, [s for s in torch.chunk(samples, samples.size(0))]
        


if __name__ == '__main__':
    from GridWorld import *
    from model import *
    from QLearning import *
    from tqdm.notebook import tqdm
    from mcmcplot import mcmcplot as mcp
    from arviz import ess, plot_autocorr, plot_trace
    import datetime
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--training_step', default=MCMC_T, type=int)
    parser.add_argument('-t', '--time', default=datetime.datetime.now().strftime("%f"))
    parser.add_argument('-s', '--save', default=SAVE)
    parser.add_argument('-p', '--show', default=SHOW)
    parser.add_argument('-e', '--epsilon', default=EPSILON, type=float)
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
    warmup_steps = int(training_steps * WARMUP_RATIO)
    GREEDY = args.Greedy

    N_PARTICLE = 10
    random.seed(seed)
    pyro.set_rng_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = GridWorld((3,4), obstacles=True)
    dim = env.observation_space.n * env.action_space.n
    model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal')
    
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
            model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal')
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
                        print(*zip(obs._buffers['state0'][-FROZEN_T:],obs._buffers['action'][-FROZEN_T:]))
                        #MCMC
                        if BATCH_TRAINING:
                            batch_indices = random.sample(range(min(len(obs._buffers['state0']), BUFFER_SIZE)), k=min(BATCH_SIZE, len(obs._buffers['state0']))) #TODO: what is this?
                        else:
                            batch_indices = slice(None)
                        r_hat = partial(generate_samples, model=model, obs=obs._buffers,  batch_indices=batch_indices)


                        # if BATCH_TRAINING:
                        #     mcmc_run = mcmc([obs._buffers, model, dim, batch_indices, abc_epsilon], torch.tensor(posterior_samples[-1].reshape(-1)), num_samples=training_steps,  warmup_steps=training_steps//5, MCMC_SHOW_DISABLE=MCMC_SHOW_DISABLE)
                        # else:
                        #     mcmc_run = mcmc([obs._buffers, model, dim, batch_indices, abc_epsilon], torch.tensor(posterior_samples[-1].reshape(-1)), num_samples=training_steps,  warmup_steps=warmup_steps, MCMC_SHOW_DISABLE=MCMC_SHOW_DISABLE)
                        # posterior_samples = mcmc_run.get_samples()["prior_parameter"].reshape((-1, ) + env.n_cell + (env.action_space.n, ))

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
                        
                        llh_transform_grad_fn = lambda data, parameter:  tabular_indicator(para=parameter.reshape(env.n_cell + (env.action_space.n, )), model=model, obs=obs._buffers)#standard form

                        prior = IsotropicGaussianPrior(sd=PRIOR_SIGMA)
                        abclikelihood = GaussianABCLikelihood(epsilon=EPSILON)
                        Model = DeterministicSRModel(prior=prior, abclikelihood=abclikelihood, llh_transform_fn=r_hat, llh_transform_grad_fn=llh_transform_grad_fn)

                        # def gradient(para, Model, model ,obs, samples):
                        #     Indicator = tabular_indicator(para=para, model=model, obs=obs, samples=samples)
                        #     Sum = np.matmul(Indicator.T, (np.array(obs['rewards']) - np.array(samples))).T #Theta x T, Tx1
                        #     return - para / Model.prior.sigma ** 2 + 1 / Model.abclikelihood.epsilon ** 2 * Sum

                        def fn(parameter):
                            current_logtarget_density, _ = Model.logtarget_density(data=torch.tensor(obs._buffers["rewards"])[-BUFFER_SIZE:][batch_indices], parameter=parameter, para_extra_info=None)
                            return current_logtarget_density
        
                        hessian = torch.autograd.functional.hessian(fn,torch.tensor(posterior_samples[-3].reshape(-1)))
                        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE)
                        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE, covariance_matrix=-torch.linalg.inv(hessian))
                        #kernel = pCN(model=Model, stepsize=STEPSIZE)
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=None)
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=-torch.linalg.inv(hessian))
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=False)
                        kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=False, precondition_matrix=-torch.linalg.inv(hessian))

                        mcmc = MCMC(num_samples=training_steps, kernel=kernel, initial_params=torch.tensor(posterior_samples[-1].reshape(-1)))
                        posterior_samples = mcmc.run(torch.tensor(obs._buffers["rewards"])[-BUFFER_SIZE:][batch_indices]).reshape((-1, ) + env.n_cell + (env.action_space.n, ))

                        # STEPSIZE *= DECREASING_FACTOR
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
                            q = np.percentile(data_plot[:, i], [PLOT_THRESHOLD, 100 - PLOT_THRESHOLD])
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
                if save:
                    with open(f'Returns/MCMC/T{training_steps}_Gdy{GREEDY}_Ep{EPSILON}_Stp{INITIAL_STEPSIZE}_Dcrs{DECREASING_FACTOR}_{time}.npy', 'wb') as f:
                        np.save(f, r_all_epi)
                        print(f'EPISODES return for repeat {repeat} saved at', f'Returns/MCMC/T{training_steps}_Gdy{GREEDY}_Ep{EPSILON}_Stp{INITIAL_STEPSIZE}_Dcrs{DECREASING_FACTOR}_{time}.npy')
            r_all_iter.append(r_all_epi)
        if save:
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

        posterior_samples = torch.tensor(model.get_parameter())
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
        
        llh_transform_grad_fn = lambda data, parameter:  tabular_indicator(para=parameter.reshape(env.n_cell + (env.action_space.n, )), model=model, obs=obs._buffers)#standard form

        prior = IsotropicGaussianPrior()
        abclikelihood = GaussianABCLikelihood(epsilon=EPSILON)
        Model = DeterministicSRModel(prior=prior, abclikelihood=abclikelihood, llh_transform_fn=r_hat, llh_transform_grad_fn=llh_transform_grad_fn)

        def fn(parameter):
            current_logtarget_density, _ = Model.logtarget_density(data=torch.tensor(obs["rewards"]), parameter=parameter, para_extra_info=None)
            return current_logtarget_density

        hessian = torch.autograd.functional.hessian(fn,torch.tensor(posterior_samples[-3].reshape(-1)))
        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE)
        kernel = RandomWalk(model=Model, stepsize=STEPSIZE, covariance_matrix=-torch.linalg.inv(hessian))
        #kernel = pCN(model=Model, stepsize=STEPSIZE)
        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=None)
        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=-torch.linalg.inv(hessian))
        #kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=False)
        #kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=False, precondition_matrix=-torch.linalg.inv(hessian))

        mcmc = MCMC(num_samples=training_steps, kernel=kernel, initial_params=torch.tensor(posterior_samples[-1].reshape(-1)))
        posterior_samples = mcmc.run(torch.tensor(obs["rewards"])).reshape((-1, ) + env.n_cell + (env.action_space.n, ))

        
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



# if __name__ == '__main__':
#     from GridWorld import *
#     from model import *
#     from QLearning import *
#     import mcmcplot
#     from mcmcplot import mcmcplot as mcp
#     from tqdm import tqdm
#     from arviz import ess, plot_autocorr, plot_trace
#     import datetime
#     import argparse
#     import pandas as pd
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-T', '--training_step', default=MCMC_T, type=int)
#     parser.add_argument('-t', '--time', default=datetime.datetime.now().strftime("%f"))
#     parser.add_argument('-s', '--save', default=SAVE)
#     parser.add_argument('-p', '--show', default=SHOW)
#     parser.add_argument('-e', '--epsilon', default=EPSILON, type=float)
#     parser.add_argument('--seed', default=SEED, type=int)
#     parser.add_argument('--MCMC', default=True, action='store_false', help='Bool type')
#     args = parser.parse_args()
#     time = args.time
#     print('time:', time)
#     training_steps = args.training_step
#     save = args.save
#     show = args.show
#     abc_epsilon = args.epsilon
#     seed = args.seed
#     mcmc_show_disable = args.MCMC
#     save = False

#     random.seed(seed)

#     np.random.seed(seed)
#     r = []

#     env = GridWorld((3,4), obstacles=True)
#     model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal')
#     kernel = MALA(model=model, stepsize=STEPSIZE)
#     mcmc = MCMC(kernal=kernel)
#     chain = np.zeros([training_steps, env.observation_space.n * env.action_space.n])
    
#     print(env.reset())
#     # env.plot_env()
#     # env.expert(reset=True)
#     # for _ in range(repeat):
#     #     env.expert(reset=False)
    
#     #True Q
#     S = []
#     for i in range(env.n_cell[0]):
#         for j in range(env.n_cell[1]):
#             S.append((i,j)) 
#     Q = np.ones(shape=(env.n_cell + (env.action_space.n, )))/env.observation_space.n / env.action_space.n
#     A = range(env.action_space.n)
#     pi_star, Q_star, V_star = DynamicProgramming(Q, A, S, env)
    
#     if ONLINE_LEARNING:
#         r_all_iter = []
#         for repeat in range(REPEAT_EXPERIMENT):
#             samples_l = []
#             model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal')
#             r_all_epi = []
#             obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
#             s0, _ = env.reset()
#             posterior_samples = torch.tensor(model.get_parameter())
#             for e in range(EPISODES):
#                 env.reset()
#                 print(f'Episode {e} in repeat {repeat}')
#                 R = 0
#                 R_star = 0
#                 h = 0
#                 while True:
#                 # for h in tqdm(range(HORIZON)):
#                     action = model.act(s0, posterior_samples)
#                     s1, r, done, *info = env.step(action)
#                     samples = model.r_hat(s0, s1, action)
#                     samples_l.append(samples)
#                     # print(samples_l)
#                     #Optimal action
#                     R_star = V_star[s0] + GAMMA * R_star#env.R[tuple(env.P[s0 + (int(pi_star[s0]), )])]
#                     R += sum([r * GAMMA ** i for i in range(h + 1)])
#                     obs.insert({'state0': s0, 'state1': s1, 'action': action, 'rewards': r, 'done': done})
#                     s0 = s1
#                     if ( h + 1)  % FROZEN_T == 0 or done:
#                         #MCMC
#                         batch_indices = random.sample(range(min(len(obs._buffers['state0']), BUFFER_SIZE)), k=min(BATCH_SIZE, len(obs._buffers['state0'])))
    
#                         torch_sample = torch.vstack(samples_l)
#                         new_parameter, samples_l = mcmc.update(model.get_parameter(), obs._buffers, torch_sample, batch_indices)#paras_history for AM
#                         model.set_parameter(new_parameter)
#                         chain = np.array(new_parameter).reshape(new_parameter.shape[0],-1)                        # posterior_samples = new_posterior_samples
#                         print("accprob:", mcmc.accepted / N_PARTICLE)


#                         #model.plot_policy(paras=posterior_samples.numpy(), title=f'policy_T{training_steps}_{time}', additional_info = env.R, save=save, show=show)
#                         #model.plot_value(paras=posterior_samples.numpy(), title=f'value_T{training_steps}_{time}')
#                         f = mcp.plot_chain_panel(chains=chain[N_PARTICLE // 10:, :8], names=env.names,
#                                                                         settings=dict(add_pm2std=True, fig=dict(figsize=(10,10), dpi=250),
#                                                                         mean=dict(color='y', label='mean'),
#                                                                         plot=dict(color='k', label='trace')))
#                         ax = f.get_axes()
#                         for i, ai in enumerate(ax):
#                             ai.axhline(y = Q_star.flatten()[i], linestyle=':', linewidth=5, color = 'g',  label = 'true q')
#                         # reset positions to avoid overlap    
#                         f.tight_layout()
#                         handles, labels = ai.get_legend_handles_labels()
#                         ai.legend(handles, labels, bbox_to_anchor=(2, 0.2), loc='right')
#                         if show:
#                             plt.show()
#                     if done:
#                         print("done with", h + 1, 'steps')
#                         break
#                     h += 1
#                 r_all_epi.append(R_star - R)
#                 if e % FROZEN_T == 0 and save:
#                     with open(f'Models/MCMC/chains_T{training_steps}_{time}.npy', 'wb') as f:
#                         np.save(f, r_all_iter)
#                         print('model saved at', f'Models/MCMC/chains_T{training_steps}_{time}.npy')
#             r_all_iter.append(r_all_epi)
#         if save:
#             with open(f'Models/MCMC/chains_T{training_steps}_{time}.npy', 'wb') as f:
#                 np.save(f, r_all_iter)
#                 print('model saved at', f'Models/MCMC/chains_T{training_steps}_{time}.npy')
            
#         plt.plot(R)
#         if show:
#             plt.show()
#     else:
#         env.uniform_policy()
#         obs = env.uniform_obs._buffers
#         R = [obs['rewards'][0]]
#         samples_l = []
#         paras_history = []
#         for j in range(len(obs['state0'])):
#             samples_l.append(model.r_hat(obs['state0'][j], obs['state1'][j], obs['action'][j]))
#         samples_l = np.array(samples_l)
#         for t in tqdm(range(training_steps)):
#             new_parameter, samples_l = mcmc.update(model.get_parameter(), obs, samples_l)#paras_history for AM
#             model.set_parameter(new_parameter)
#             chain[t] = new_parameter.flatten()
#             if t > training_steps * BURN_IN:
#                 if t % SKIP == 0:
#                     paras_history.append(new_parameter)
#         print('accepted ratio:', mcmc.accepted / training_steps)
#         model.plot_policy(paras=np.array(paras_history), title=f'policy_T{training_steps}_{time}', additional_info = env.R, save=save, show=show)
#         print('ESS:', ess(chain.T))

#     if save:
#         with open(f'Models/MCMC/chains_T{training_steps}_{time}.npy', 'wb') as f:
#             np.save(f, chain)
#             print('model saved at', f'Models/MCMC/chains_T{training_steps}_{time}.npy')
#     figure_path = 'Figures/MCMC/'
#     #mcmc chain plots
#     f = mcp.plot_chain_panel(chains=chain[training_steps // 10:, :4],settings=dict(add_pm2std=True,
#                                                         mean=dict(color='b'),
#                                                         plot=dict(color='k')))
#     if show:
#         plt.show()
#     if save:
#         plt.savefig(f'{figure_path}traces_T{training_steps}_{time}')
#         print('Figure saved at ', f'{figure_path}traces_T{training_steps}_{time}')
#         plt.clf()
#     #density panel
#     # user_settings = dict(
#     # plot=dict(
#     #     marker='s',
#     #     mfc='none',
#     #     linestyle='none'),
#     # fig=dict(figsize=(6, 6)))
#     # names = ['a', 'b', 'c']
#     # f = mcp.plot_density_panel(
#     #     chains=chain[training_steps // 10:, :6],
#     #     names=names,
#     #     settings=user_settings)
#     # plt.show()

#     #correlation
#     # settings = dict(
#     # add_5095_contours=True,
#     # plot_95=dict(
#     #     color='r',
#     #     linewidth=3),
#     # plot_50=dict(
#     #     color='c',
#     #     linewidth=3),
#     # add_legend=True,
#     # legend=dict(
#     #     loc='upper right',
#     #     fontsize=10,
#     #     bbox_to_anchor=(0.85, 0.75)),
#     # fig=dict(figsize=(4,4)))
#     # fp = mcp.plot_pairwise_correlation_panel(
#     #     chains=chain[training_steps // 10:, :6],
#     #     settings=settings)
#     # plt.show()

#     plot_trace(chain[training_steps // 10:, :].T, compact=False)
#     if show:
#         plt.show()
#     if save:
#         plt.savefig(f'{figure_path}trace_T{training_steps}_{time}.png')
#         print('Figure saved at ', f'{figure_path}trace_T{training_steps}_{time}')
#         plt.clf()
#     corr_chain = chain[training_steps // 10 :: training_steps // 100, :].T
#     plot_autocorr(corr_chain, max_lag=min(200, len(corr_chain)))
#     if show:
#         plt.show()
#     if save:
#         plt.savefig(f'{figure_path}corr_T{training_steps}_{time}.png')
#         print('Figure saved at ', f'{figure_path}corr_T{training_steps}_{time}')
#         plt.clf()

    
    
        
    