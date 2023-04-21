import numpy as np
import scipy.stats as stats
import torch
from copy import deepcopy
from parameter import *
from functools import partial
import pyro
import pyro.distributions as dist
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
        self.sigma = float(sd)
        self.mean = mean
        
    def logprior(self, parameter, return_gradient=False):
        return torch.distributions.normal.Normal(loc=0., scale=self.sigma).log_prob(parameter).sum(axis=-1)
    
    def logprior_gradient(self, parameter):
        return - parameter / self.sigma ** 2
    
    def covariance_matrix(self, parameter_len):
        return (self.sigma ** 2) * torch.eye(parameter_len)


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

    def llh(self, data, parameter=None, llh_info_dict=dict(), llh_transform_fn=None): #standard form
        return self.llh_(data=data, parameter=parameter, llh_info_dict=llh_info_dict, mean_fn=llh_transform_fn)


    def llh_(self, data, parameter=None, mean_fn=None, llh_info_dict=dict()):

        mean = self.compute_mean(parameter=parameter, mean_fn=mean_fn, llh_info_dict=llh_info_dict)
        llh = torch.distributions.normal.Normal(loc=mean,scale=self.epsilon).log_prob(data).sum()
        return llh, {"mean":mean.detach()}
    
    def compute_mean(self, parameter=None, mean_fn=None, llh_info_dict=dict()):
        """if mean is provided, mean_fn is not used"""
        assert llh_info_dict.get("mean") is not None or mean_fn is not None, "either mean or mean_fn of the form mean_fn(parameter) -> mean should be provided"
        if llh_info_dict.get("mean") is None:
            mean = mean_fn(parameter)
        else:
            mean = llh_info_dict["mean"]
        return mean

    def llh_gradient(self, data, parameter, llh_info_dict=dict(), llh_transform_fn=None, llh_transform_grad_fn=None): #standard form
        return self.llh_gradient_(data=data, parameter=parameter, llh_info_dict=llh_info_dict, mean_fn=llh_transform_fn, mean_jacobian_fn=llh_transform_grad_fn)

    def llh_gradient_(self, data, parameter, mean_fn=None, mean_jacobian_fn=None, llh_info_dict=dict()):
        
        mean = self.compute_mean(parameter=parameter, mean_fn=mean_fn, llh_info_dict=llh_info_dict)
        mean_jacobian = mean_jacobian_fn(data=data, parameter=parameter)
        
        gradient = 1/ self.epsilon **2 *  torch.mv(torch.t(mean_jacobian),(data - mean))

        return gradient
    
    def covariance_matrix(self, data_len):
        return (self.epsilon ** 2) * torch.eye(data_len)

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

    def llh(self, data, parameter, llh_info_dict=dict()):
        llh, llh_info_dict = self.abclikelihood.llh(data=data, parameter=parameter, llh_info_dict=llh_info_dict, llh_transform_fn=self.llh_transform_fn)
        return llh, llh_info_dict

    def logtarget_density(self, data, parameter, llh_info_dict=dict()):
        logprior = self.logprior(parameter=parameter)
        llh, llh_info_dict = self.llh(data=data, parameter=parameter, llh_info_dict=llh_info_dict)
        return logprior + llh, llh_info_dict
    
    def logtarget_gradient(self, data, parameter, llh_info_dict=dict()):
        logprior_grad = self.prior.logprior_gradient(parameter=parameter)
        llh_grad = self.abclikelihood.llh_gradient(data=data, parameter=parameter, llh_info_dict=llh_info_dict, llh_transform_fn=self.llh_transform_fn, llh_transform_grad_fn=self.llh_transform_grad_fn)
        return logprior_grad + llh_grad
    
    def logtarget_auto_gradient(self, data, parameter, llh_info_dict=dict()):
        parameter = parameter.clone()
        parameter.requires_grad = True
        logtarget_density, llh_info_dict = self.logtarget_density(data=data, parameter=parameter, llh_info_dict=llh_info_dict)
        logtarget_density.backward()
        gradient = parameter.grad.clone()
        parameter.grad.zero_()
        parameter.requires_grad = False
        logtarget_density = logtarget_density.detach()
        return logtarget_density, gradient, llh_info_dict

    def pyro_model(self, data, parameter_len):
        prior_parameter = pyro.sample("prior_parameter", dist.MultivariateNormal(torch.zeros(parameter_len), self.prior.covariance_matrix(parameter_len=parameter_len)))
        mean = self.abclikelihood.compute_mean(parameter=prior_parameter, mean_fn=self.llh_transform_fn, llh_info_dict=dict())
        with pyro.plate("data_plate"):
            pyro.sample("obs", dist.MultivariateNormal(mean, self.abclikelihood.covariance_matrix(data_len=len(data))), obs=data)


class Kernel:
    def __init__(self, model):
        self.model = model

    def move(self):
        raise NotImplementedError

    def propose_accept(self):
        raise NotImplementedError
    
    def gradient(self, data, parameter, info_dict=dict(), llh_info_dict=dict(), return_logtarget_density=True):
        if self.use_autograd:
            if info_dict.get("gradient") is not None:
                logtarget_density, llh_info_dict = self.model.logtarget_density(data=data, parameter=parameter, llh_info_dict=llh_info_dict)
                gradient = info_dict["gradient"]
            else:
                logtarget_density, gradient, llh_info_dict = self.model.logtarget_auto_gradient(data=data, parameter=parameter, llh_info_dict=llh_info_dict)

        else:
            logtarget_density, llh_info_dict = None, dict()
            if info_dict.get("gradient") is not None:
                gradient = info_dict["gradient"]
            else:
                gradient = self.model.logtarget_gradient(data=data, parameter=parameter, llh_info_dict=llh_info_dict)
        
        if return_logtarget_density is True:
            if logtarget_density is None:
                logtarget_density, llh_info_dict = self.model.logtarget_density(data=data, parameter=parameter, llh_info_dict=llh_info_dict)
            return gradient, logtarget_density, llh_info_dict
        
        return gradient

class RandomWalk(Kernel):
    def __init__(self, model, stepsize=0.5, covariance_matrix=None, *args):
        print('RandomWalk stepsize', stepsize)
        super(RandomWalk, self).__init__(model=model)   
        self.stepsize = stepsize
        self.covariance_matrix = covariance_matrix

    def move(self, current_para):
        if self.covariance_matrix is not None:
            return torch.distributions.multivariate_normal.MultivariateNormal(loc=current_para, covariance_matrix= (self.stepsize ** 2) * self.covariance_matrix).sample()
        else:
            return torch.normal(mean=current_para, std=self.stepsize)

    def propose_accept(self, current_para, data, current_para_info_dict=dict()):

        current_para_llh_info_dict = current_para_info_dict["llh_info_dict"] if current_para_info_dict.get("llh_info_dict") is not None else dict()
        proposed_para = self.move(current_para)

        current_logtarget_density, _ = self.model.logtarget_density(data=data, parameter=current_para, llh_info_dict=current_para_llh_info_dict)
        proposed_logtarget_density, proposed_para_llh_info_dict = self.model.logtarget_density(data=data, parameter=proposed_para, llh_info_dict=dict())
        accept_prob = proposed_logtarget_density - current_logtarget_density

        proposed_para_info_dict = {"llh_info_dict": proposed_para_llh_info_dict, "logdensities":proposed_logtarget_density}

        return accept_prob, proposed_para, proposed_para_info_dict #acceptance_prob, proposal, extra_stat_for_proposal
    
class pCN(Kernel):
    def __init__(self, model, stepsize=0.5, *args):
        print('pCN stepsize', stepsize)
        super(pCN, self).__init__(model=model)   
        self.stepsize = stepsize
        self.sigma = self.model.prior.sigma
        
    def move(self, current_para):
        return np.sqrt(1 - self.stepsize ** 2) * current_para + self.stepsize * torch.normal(mean=torch.zeros(current_para.size()), std=self.sigma)
    
    def propose_accept(self, current_para, data, current_para_info_dict=dict()):

        current_para_llh_info_dict = current_para_info_dict["llh_info_dict"] if current_para_info_dict.get("llh_info_dict") is not None else dict()
    
        proposed_para = self.move(current_para)

        current_llh, _ = self.model.llh(data=data, parameter=current_para, llh_info_dict=current_para_llh_info_dict)
        proposed_llh, proposed_para_llh_info_dict = self.model.llh(data=data, parameter=proposed_para, llh_info_dict=dict())
        accept_prob = proposed_llh - current_llh

        proposed_para_info_dict = {"llh_info_dict": proposed_para_llh_info_dict, "logdensities":proposed_llh}

        return accept_prob, proposed_para, proposed_para_info_dict #acceptance_prob, proposal, extra_stat_for_proposal
    
class MALA(Kernel):
    def __init__(self, model, stepsize=0.5, precondition_matrix=None, use_autograd=True, *args):
        print('MALA stepsize', stepsize)
        super(MALA, self).__init__(model=model)   
        self.stepsize = stepsize
        self.precondition = precondition_matrix
        self.use_autograd = use_autograd

    def move(self, current_para, current_gradient):
        if self.precondition is None:
            proposed_para = current_para + ((self.stepsize ** 2) / 2) * current_gradient + self.stepsize *  torch.normal(mean=torch.zeros(current_para.size()), std=1.)
        else:
            noise = torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.zeros(current_para.size()), covariance_matrix=self.precondition).sample()
            proposed_para = current_para + ((self.stepsize ** 2) / 2) * (self.precondition @ current_gradient) + self.stepsize * noise
            #print("gradient step", ((self.stepsize ** 2) / 2) * (self.precondition @ current_gradient))
            #print("noise",  self.stepsize * noise)
            #print("current_para", current_para)
        return proposed_para
    
    def move_ratio(self, current_para, current_gradient, proposed_para, proposed_gradient):

        if self.precondition is None:
            curr_to_prop = torch.distributions.normal.Normal(loc=current_para + ((self.stepsize ** 2) / 2) * current_gradient, scale=self.stepsize).log_prob(proposed_para).sum()
            prop_to_curr = torch.distributions.normal.Normal(loc=proposed_para + ((self.stepsize ** 2) / 2) * proposed_gradient, scale=self.stepsize).log_prob(current_para).sum()
        else:
            curr_to_prop = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=current_para + ((self.stepsize ** 2) / 2) * (self.precondition @ current_gradient), 
                covariance_matrix=self.precondition * (self.stepsize ** 2)
                ).log_prob(proposed_para).sum()
            prop_to_curr = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=proposed_para + ((self.stepsize ** 2) / 2) * (self.precondition @ proposed_gradient), 
                covariance_matrix=self.precondition * (self.stepsize ** 2)
                ).log_prob(current_para).sum()
        return curr_to_prop - prop_to_curr

    def propose_accept(self, current_para, data, current_para_info_dict=dict()):

        current_para_llh_info_dict = current_para_info_dict["llh_info_dict"] if current_para_info_dict.get("llh_info_dict") is not None else dict()
        current_gradient, current_logtarget_density, current_para_llh_info_dict = self.gradient(data=data, parameter=current_para, info_dict=current_para_info_dict, llh_info_dict=current_para_llh_info_dict, return_logtarget_density=True)

        proposed_para = self.move(current_para=current_para, current_gradient=current_gradient)
        proposed_gradient, proposed_logtarget_density, proposed_para_llh_info_dict = self.gradient(data=data, parameter=proposed_para, info_dict=dict(), llh_info_dict=dict(), return_logtarget_density=True)

        # print("current_grad", current_gradient)
        # print("proposed_gradient", proposed_gradient)

        move_ratio = self.move_ratio(current_para=current_para, current_gradient=current_gradient, proposed_para=proposed_para, proposed_gradient=proposed_gradient)
        accept_prob = proposed_logtarget_density - current_logtarget_density - move_ratio
        #print("diff a", proposed_logtarget_density - current_logtarget_density, "diff b", move_ratio)
        proposed_para_info_dict = {"llh_info_dict": proposed_para_llh_info_dict, "gradient": proposed_gradient, "logdensities":proposed_logtarget_density}

        return accept_prob, proposed_para, proposed_para_info_dict #acceptance_prob, proposal, extra_stat_for_proposal
    
    # def inverse_riemann_mass(self, Indicator):
    #     Indicator_flatten = Indicator.reshape(Indicator.shape[0],-1)
    #     fisher = 1 / (self.likelihood.epsilon ** 2) * Indicator_flatten.T @ Indicator_flatten
    #     fisher +=  self.prior.sigma ** 2 * np.identity(fisher.shape[0])
    #     pass


class HMC(Kernel):
    def __init__(self, model, num_steps=10, traj_len=2*np.pi, stepsize=0.5, precondition_matrix=None, use_autograd=True, *args):
        print('HMC stepsize', stepsize)
        super(HMC, self).__init__(model=model)   
        self.stepsize = stepsize
        self.precondition = precondition_matrix
        self.use_autograd = use_autograd
    
        if num_steps is None:
            num_steps = np.ceil(traj_len / stepsize)
        self.L = num_steps

    def move(self, current_para, current_gradient, data):
        if self.precondition is not None:
            p0 = torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.zeros(current_para.size()), precision_matrix=self.precondition).sample()
        else:
            p0 = torch.normal(mean=torch.zeros(current_para.size()), std=1.)

        p = p0 + self.stepsize * current_gradient * 0.5
        q = current_para

        for i in range(self.L):
            q_move = torch.mv(self.precondition, p) if self.precondition is not None else p
            q = q + self.stepsize * q_move
            if i != (self.L-1):
                p = p + self.stepsize * self.gradient(data=data, parameter=q, info_dict=dict(), llh_info_dict=dict(), return_logtarget_density=False)
        
        proposed_gradient, proposed_logtarget_density, proposed_para_llh_info_dict = self.gradient(data=data, parameter=q, info_dict=dict(), llh_info_dict=dict(), return_logtarget_density=True)
        p = p + self.stepsize * proposed_gradient * 0.5
        q_info_dict = {"logtarget_density":proposed_logtarget_density, "gradient":proposed_gradient, "llh_info_dict":proposed_para_llh_info_dict}
        return q, p0, p, q_info_dict

    def propose_accept(self, current_para, data, current_para_info_dict=dict()):

        current_para_llh_info_dict = current_para_info_dict["llh_info_dict"] if current_para_info_dict.get("llh_info_dict") is not None else dict()
        current_gradient, current_logtarget_density, current_para_llh_info_dict = self.gradient(data=data, parameter=current_para, info_dict=current_para_info_dict, llh_info_dict=current_para_llh_info_dict, return_logtarget_density=True)

        proposed_para, p0, p, q_info_dict = self.move(current_para=current_para, current_gradient=current_gradient, data=data)
        proposed_logtarget_density, proposed_gradient, proposed_para_llh_info_dict = q_info_dict["logtarget_density"], q_info_dict["gradient"], q_info_dict["llh_info_dict"]

        H_old = self.hamiltonian(q=current_para, p=p0, q_logtarget_density=current_logtarget_density)
        H_new =  self.hamiltonian(q=proposed_para, p=p, q_logtarget_density=proposed_logtarget_density)
        accept_prob = H_old - H_new

        proposed_para_info_dict = {"llh_info_dict": proposed_para_llh_info_dict, "gradient": proposed_gradient, "logdensities":proposed_logtarget_density}

        return accept_prob, proposed_para, proposed_para_info_dict

    def hamiltonian(self, q, p, q_logtarget_density=None):
        if q_logtarget_density is None:
            q_logtarget_density = self.model.logtarget_density(q)
        p_logdensity = 0.5 * torch.dot(p, torch.mv(self.precondition, p)) if self.precondition is not None else 0.5 * torch.dot(p, p)
        return p_logdensity - q_logtarget_density


def HMC_pyro(model, parameter_len, stepsize=0.5, *args, **kwargs):
        print('HMC stepsize', stepsize)
        pyro.clear_param_store()
        pyro_model = lambda data: model.pyro_model(data=data, parameter_len=parameter_len)
        return pyro.infer.mcmc.HMC(model=pyro_model, step_size=stepsize, **kwargs)


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
    def __init__(self, kernel, num_samples=MCMC_SAMPLE, initial_params=None, params_dim=None, **kwargs):
        self.num_samples = num_samples
        self.kernel = kernel
        assert initial_params is not None or params_dim is not None, "Should either specify initial_params or params_dim"
        if initial_params is not None:
            self.initial_params = initial_params
            self.params_dim = len(initial_params)
        else:
            self.initial_params = torch.tensor(params_dim)
            self.params_dim = params_dim

        self.reset()

    #def run(self, paras, obs, samples, batch_indices=None):
    def run(self, data):

        self.reset()

        current_para = self.initial_params if self.initial_params is not None else torch.zeros(self.params_dim)
        current_logtarget_density, _ = self.kernel.model.logtarget_density(data=data, parameter=current_para, llh_info_dict=dict())
        current_para_info_dict = {"logdensities":current_logtarget_density}

        self.samples[0] = current_para
        self.logdensities[0] = current_logtarget_density
        self.proposed_logdensities[0] = current_logtarget_density

        pbar = tqdm(range(self.num_samples))
        for i in pbar:
            accept_prob, proposed_para, proposed_para_info_dict  = self.kernel.propose_accept(current_para=current_para, 
                                                                         data=data, 
                                                                         current_para_info_dict=current_para_info_dict)

            if np.log(np.random.uniform(0,1)) < accept_prob:
                current_para = proposed_para
                current_para_info_dict = proposed_para_info_dict
                self.accepted += 1

            pbar.set_description("Acceptance probability {}".format(np.round(self.accepted/(i+1), 2)))

            self.samples[i+1] = current_para
            self.logdensities[i+1] = current_para_info_dict["logdensities"]
            self.proposed_logdensities[i+1] = proposed_para_info_dict["logdensities"]
            self.accept_prob[i+1] = accept_prob

        return self.samples
    
    def get_samples(self):
        return self.samples
    
    def get_logdensities(self):
        return self.logdensities
    
    def get_proposed_logdensities(self):
        return self.proposed_logdensities
    
    def get_accept_prob(self):
        return self.accept_prob
    
    def reset(self):
        self.samples = torch.zeros((self.num_samples+1, self.params_dim))
        self.logdensities = torch.zeros(self.num_samples+1)
        self.proposed_logdensities = torch.zeros(self.num_samples+1)
        self.accepted = 0
        self.accept_prob = torch.zeros(self.num_samples+1)

class MCMC_pyro(MCMC):
    def __init__(self, kernel, num_samples=MCMC_SAMPLE, initial_params=None, params_dim=None, warmup_steps=MCMC_T//5, disable_progbar=MCMC_SHOW_DISABLE, **kwargs):
        super(MCMC_pyro, self).__init__(kernel=kernel, num_samples=num_samples, initial_params=initial_params, params_dim=params_dim, **kwargs)
        self.pyro_mcmc = pyro.infer.mcmc.MCMC(kernel=kernel, num_samples=num_samples, initial_params={'prior_parameter': initial_params}, warmup_steps=warmup_steps, disable_progbar=disable_progbar, **kwargs)

    def run(self, data):

        self.reset()

        self.pyro_mcmc.run(data)
        self.samples = self.pyro_mcmc.get_samples()["prior_parameter"]
        return self.samples
    
    def get_logdensities(self):
        raise NotImplementedError

    def get_proposed_logdensities(self):
        raise NotImplementedError
    
    def get_accept_prob(self):
        raise NotImplementedError




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
    parser.add_argument('-n', '--stepsize', default=STEPSIZE, type=float)
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
    STEPSIZE = args.stepsize
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
            model = Tabular(env=env, n_particle=training_steps, prior='normal')
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
                        
                        llh_transform_grad_fn = lambda data, parameter:  tabular_indicator(para=parameter.reshape(env.n_cell + (env.action_space.n, )), model=model, obs=obs._buffers)#standard form

                        prior = IsotropicGaussianPrior(sd=PRIOR_SIGMA)
                        abclikelihood = GaussianABCLikelihood(epsilon=EPSILON)
                        Model = DeterministicSRModel(prior=prior, abclikelihood=abclikelihood, llh_transform_fn=r_hat, llh_transform_grad_fn=llh_transform_grad_fn)

                        def fn(parameter):
                            current_logtarget_density, _ = Model.logtarget_density(data=torch.tensor(obs._buffers["rewards"])[-BUFFER_SIZE:][batch_indices], parameter=parameter, llh_info_dict=dict())
                            return current_logtarget_density
                        hessian = torch.autograd.functional.hessian(fn,torch.tensor(posterior_samples[0].reshape(-1)))
                        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE)
                        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE, covariance_matrix=-torch.linalg.inv(hessian))
                        #kernel = pCN(model=Model, stepsize=STEPSIZE)
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=None)
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=-torch.linalg.inv(hessian))
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=False)
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=False, precondition_matrix=-torch.linalg.inv(hessian))
                        kernel = HMC_pyro(model=Model, parameter_len=len(posterior_samples[0].reshape(-1)), stepsize=STEPSIZE, full_mass=FULL_MASS, adapt_step_size=ADAPT_STEP_SIZE, adapt_mass_matrix=ADAPT_MASS_MATRIX, target_accept_prob=TARGET_ACCEPT_PROB, num_steps=NUM_STEPS)
                        #kernel = HMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=False)
                        #kernel = HMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=False, precondition_matrix=-torch.linalg.inv(hessian))
                        #kernel = HMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=True, precondition_matrix=-torch.linalg.inv(hessian))
                        accept_probs = None

                        if isinstance(kernel, Kernel):
                            mcmc = MCMC(num_samples=training_steps, kernel=kernel, initial_params=torch.tensor(posterior_samples[-1].reshape(-1)))
                            posterior_samples = mcmc.run(torch.tensor(obs._buffers["rewards"])[-BUFFER_SIZE:][batch_indices]).reshape((-1, ) + env.n_cell + (env.action_space.n, ))
                            logdensities = mcmc.get_logdensities()
                            proposed_logdensities = mcmc.get_proposed_logdensities()
                            accept_probs = mcmc.get_accept_prob()

                        else:
                            mcmc = MCMC_pyro(num_samples=training_steps, kernel=kernel, initial_params=torch.tensor(posterior_samples[-1].reshape(-1)), warmup_steps=np.int64(np.floor(training_steps*WARMUP_RATIO)), disable_progbar=MCMC_SHOW_DISABLE)
                            posterior_samples = mcmc.run(torch.tensor(obs._buffers["rewards"])[-BUFFER_SIZE:][batch_indices]).reshape((-1, ) + env.n_cell + (env.action_space.n, ))

                        STEPSIZE *= DECREASING_FACTOR
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
            current_logtarget_density, _ = Model.logtarget_density(data=torch.tensor(obs["rewards"]), parameter=parameter, llh_info_dict=dict())
            return current_logtarget_density

        hessian = torch.autograd.functional.hessian(fn,torch.tensor(posterior_samples[-3].reshape(-1))) + 1.e-6
        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE)
        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE, covariance_matrix=-torch.linalg.inv(hessian))
        #kernel = pCN(model=Model, stepsize=STEPSIZE)
        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=None)
        kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=-torch.linalg.inv(hessian))
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

    
    
        
    