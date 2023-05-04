import numpy as np
import scipy.stats as stats
import torch
from copy import deepcopy
from parameter import *
from functools import partial
import pyro
import pyro.distributions as dist
from tqdm.notebook import tqdm
from utils import torch_max_0, is_diagonal

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
    def __init__(self, sd=1., mean=0.):
        """log(p(para)) independent gaussian prior
        sd: float
            - standard deviaition of each dimension
        mean: float
            - mean of each dimension
        """
        self.sigma = float(sd)
        self.mean = mean
        
    def logprior(self, parameter):
        """return the log prior of a given parameter
        parameter: torch.tensor
            - the parameter for which the log prior is computed"""
        return torch.distributions.normal.Normal(loc=self.mean, scale=self.sigma).log_prob(parameter).sum(axis=-1)
    
    def logprior_gradient(self, parameter):
        """return the gradient of the log prior with respect to the parameter
        parameter: torch.tensor
            - the parameter for which the log prior gradient is computed
        """
        return - parameter / (self.sigma ** 2)
    
    def logprior_hessian(self, parameter):
        """return the hessian of the log prior with respect to the parameter
        parameter: torch.tensor
            - the parameter for which the log prior hessian is computed
        """
        return - 1 / (self.sigma ** 2) * torch.eye(parameter.size()[-1])
    
    def covariance_matrix(self, parameter_len):
        """return the overall covariance matrix of the prior given the dimension of the parameter
        parameter_len: int
            - the dimension of the parameter
        """
        return (self.sigma ** 2) * torch.eye(parameter_len)

        
class GaussianABCLikelihood():
    """log(p(evidence|para)) of the Gaussian ABC model with provided mean or mean function with respect to the parameter"""
    def __init__(self, epsilon=EPSILON):
        """
        epsilon: float
            - the ABC error for epsilon
        """
        self.epsilon = epsilon

    def llh(self, data, parameter=None, llh_info_dict=dict(), llh_transform_fn=None): #standard form
        """return the log likelihood for the parameter and data given
        data: torch.tensor
            - the target of the ABC likelihood
        parameter: torch.tensor, optional
            - the parameter for the model, see llh_info_dict
        llh_info_dict: dict, of the form {"mean":torch.tensor}, optional
            - an optional dictionary that supplies to the function with the mean of the Gaussian ABC likelihood. If not provided, llh_transform_fn will be used to compute the mean
        llh_transform_fn: function torch.tensor -> torch.tensor, optional 
            - a function that transforms the parameter into the mean of the Gaussian ABC likelihood. If the mean is supplied by llh_info_dict, this function is not used.
        return:
            - llh: torch.tensor
                - log likelihood for the parameter and data given
            - llh_info_dict: dict of the form {"mean": torch.tensor}
                - mean of the Gaussian ABC likelihood function in a dictionary
        """
        return self.llh_(data=data, parameter=parameter, mean_fn=llh_transform_fn, llh_info_dict=llh_info_dict,)

    def llh_(self, data, parameter=None, mean_fn=None, llh_info_dict=dict()):
        """see self.llh, where mean_fn is the llh_transform_fn of self.llh"""
        mean = self.compute_mean(parameter=parameter, mean_fn=mean_fn, llh_info_dict=llh_info_dict)
        llh = torch.distributions.normal.Normal(loc=mean, scale=self.epsilon).log_prob(data).sum(axis=-1)
        return llh, {"mean":mean.detach()}
    
    def compute_mean(self, parameter=None, mean_fn=None, llh_info_dict=dict()):
        """function to compute the mean of the Gaussian ABC likelihood, see self.llh and self.llh_"""
        assert llh_info_dict.get("mean") is not None or mean_fn is not None, "either mean or mean_fn of the form mean_fn(parameter) -> mean should be provided"
        if llh_info_dict.get("mean") is None:
            mean = mean_fn(parameter)
        else:
            mean = llh_info_dict["mean"]
        return mean

    def llh_gradient(self, data, parameter, llh_transform_grad_fn, llh_info_dict=dict(), llh_transform_fn=None): #standard form
        """compute the gradient of the log likelihood function with respect to the parameter
        data: see self.llh
        parameter: see self.llh
        llh_info_dict: see self.llh
        llh_transform_fn: see self.llh
        llh_transform_grad_fn: function torch.tensor -> torch.tensor
            - a function that takes the parameter and output the gradient (Jacobian) of the mean of the Gaussian ABC likelihood function with respect to the parameter
        """
        return self.llh_gradient_(data=data, parameter=parameter, mean_jacobian_fn=llh_transform_grad_fn, mean_fn=llh_transform_fn, llh_info_dict=llh_info_dict)

    def llh_gradient_(self, data, parameter, mean_jacobian_fn=None, mean_fn=None, llh_info_dict=dict()):
        """see self.llh_gradient, in which mean_jacobian_fn is the llh_transform_grad_fn"""
        mean = self.compute_mean(parameter=parameter, mean_fn=mean_fn, llh_info_dict=llh_info_dict)
        mean_jacobian = mean_jacobian_fn(parameter=parameter)
        
        gradient = 1. / (self.epsilon**2) *  torch.mv(torch.t(mean_jacobian),(data - mean))
        return gradient, {"mean_jacobian":mean_jacobian}
    
    def llh_hessian(self, parameter, llh_transform_grad_fn=None, llh_transform_hessian_fn=None, llh_grad_info_dict=dict(), **kwargs):
        """compute the hessian of the log likelihood with respect to the parameter at the parameter
        parameter: see self.llh
        llh_transform_grad_fn: see self.llh_gradient
        llh_transform_hessian_fn: None
            - a function that takes the parameter and output the hessian of the mean of the Gaussian ABC likelihood function with respect to the parameter. Currently not implemented, hence only accept None
        llh_grad_info_dict: dict, of the form {"mean_jacobian": torch.tensor}
            - the dictionary output by self.llh_gradient that contains the jacobian of the mean of the Gaussian ABC likelihood function with respect to the parameter    
        """
        return self.llh_hessian_(parameter=parameter, mean_jacobian_fn=llh_transform_grad_fn, mean_hessian_fn=llh_transform_hessian_fn, llh_grad_info_dict=llh_grad_info_dict, **kwargs)
    
    def llh_hessian_(self, parameter, mean_jacobian_fn=None, mean_hessian_fn=None, llh_grad_info_dict=dict(), **kwargs):
        """see self.llh_hessian, in which mean_jacobian_fn is the llh_transform_grad_fn, and mean_hessian_fn is the llh_transform_hessian_fn"""
        if mean_hessian_fn is None:
            if llh_grad_info_dict.get("mean_jacobian") is None:
                mean_jacobian = mean_jacobian_fn(parameter=parameter)
            else:
                mean_jacobian = llh_grad_info_dict.get("mean_jacobian")
            return - 1. / (self.epsilon**2) * torch.matmul(mean_jacobian.T, mean_jacobian)
        else:
            raise NotImplementedError
    
    def covariance_matrix(self, data_len):
        """return the overall covariance matrix of the ABC Gaussian likelihood given the dimension of the data
        data_len: int
            - the dimension of the data
        """
        return (self.epsilon ** 2) * torch.eye(data_len)


class DeterministicSRModel():
    """the overall model of the ABC likelihood model with prior and likelihood (with deterministic reward and state transition)"""
    def __init__(self, prior, abclikelihood, data, llh_transform_fn=None, llh_transform_grad_fn=None, llh_transform_hessian_fn=None, *args):
        """
        prior: a prior class instance
            - a prior class instance that contains a method logprior that takes a parameter and output the log prior density, 
              and a method logprior_gradient that takes a parameter and output the gradient of the logprior with respect to the parameter
              The logprior method should take input: parameter (torch.tensor); and output torch.tensor
              The logprior_gradient should take input: parameter(torch.tensor); and output torch.tensor
        abclikelihood: an abclikelihood class instance
            - a likelihood class instance that contains a method llh that takes a parameter and data and output the log likelihood density, 
              and a method llh_gradient that takes a parameter and data and output the gradient of the loglikelihood with respect to the parameter.
              The llh method is allowed to take input: data (torch.tensor), parameter (torch.tensor), llh_info_dict, llh_transform_fn; and output torch.tensor
              The llh_gradient is allowed to take input: data (torch.tensor), parameter (torch.tensor), llh_info_dict, llh_transform_fn, llh_transform_grad_fn; and output torch.tensor
        data: torch.tensor
            - the data for the target of the abclikelihood
        llh_transform_fn: function torch.tensor -> torch.tensor, optional 
            - a function that is required by the abclikelihood function to transform the parameter
        llh_transform_grad_fn: function torch.tensor -> torch.tensor, optional
            - a function that is required by the abclikelihood function to transform the parameter into the gradient of a component of the abclikelihood function
        """
        self.prior = prior
        self.abclikelihood = abclikelihood
        self.llh_transform_fn = llh_transform_fn
        self.llh_transform_grad_fn  = llh_transform_grad_fn
        self.llh_transform_hessian_fn = llh_transform_hessian_fn
        self.data = data

    def logprior(self, parameter):
        """compute the logprior"""
        return self.prior.logprior(parameter=parameter)

    def llh(self, parameter, llh_info_dict=dict()):
        """compute the loglikelihood given the abclikelihood and return the loglikelihood with the llh_info_dict"""
        llh, llh_info_dict = self.abclikelihood.llh(data=self.data, parameter=parameter, llh_info_dict=llh_info_dict, llh_transform_fn=self.llh_transform_fn)
        return llh, llh_info_dict

    def logtarget_density(self, parameter, llh_info_dict=dict()):
        """compute the log target density (logprior + llh) given the abclikelihood and prior and return the log target density with the llh_info_dict"""
        logprior = self.logprior(parameter=parameter)
        llh, llh_info_dict = self.llh(parameter=parameter, llh_info_dict=llh_info_dict)
        return logprior + llh, llh_info_dict
    
    def logtarget_gradient(self, parameter, llh_info_dict=dict()):
        """compute the gradient of the log target density with respect to the parameter and return the gradient, using the explicit derivation of the gradient"""
        logprior_grad = self.prior.logprior_gradient(parameter=parameter)
        llh_grad, llh_grad_info = self.abclikelihood.llh_gradient(data=self.data, parameter=parameter, llh_info_dict=llh_info_dict, llh_transform_fn=self.llh_transform_fn, llh_transform_grad_fn=self.llh_transform_grad_fn)
        return logprior_grad + llh_grad, llh_grad_info
    
    def logtarget_auto_gradient(self, parameter):
        """compute the gradient of the log target density with respect to the parameter and return the gradient, using automatic differentiation with pytorch
        parameter: torch.tensor (no gradient needed)
            - the parameter the gradient is computed at
        return:
            - logtarget_density: torch.tensor
                - the log target density of the target density with respect to the input parameter
            - gradient: torch.tensor
                - the gradient of the log target density with respect to and at the input parameter
            - llh_info_dict: dict
                - a dictionary of info output by the abclikelihood.llh function
        """
        parameter = parameter.clone()
        parameter.requires_grad = True
        logtarget_density, llh_info_dict = self.logtarget_density(parameter=parameter, llh_info_dict=dict()) # must use the llh_transform_fn to compute the density
        logtarget_density.backward()
        gradient = parameter.grad.clone()
        parameter.grad.zero_()
        parameter.requires_grad = False
        logtarget_density = logtarget_density.detach()
        return logtarget_density, gradient, llh_info_dict
    
    def logtarget_hessian(self, parameter, llh_info_dict=dict(), llh_grad_info_dict=dict()):
        """compute the hessian of the log target density with respect to the parameter and return the hessian, using explicit derivation of the hessian
        parameter: torch.tensor
            - the parameter the gradient is computed at
        llh_info_dict: dict, optional
            - a dictionary of info returned by abclikelihood.llh function
        llh_grad_info_dict: dict, optional
            - a dictionary of gradient info returned by abclikelihood.gradient function
        return:
            - hessian: torch.tensor
                - the hessian at the parameter
        """
        logprior_hessian = self.prior.logprior_hessian(parameter=parameter)
        llh_hessian = self.abclikelihood.llh_hessian(data=self.data, parameter=parameter, llh_info_dict=llh_info_dict, llh_grad_info_dict=llh_grad_info_dict, llh_transform_fn=self.llh_transform_fn, llh_transform_grad_fn=self.llh_transform_grad_fn, llh_transform_hessian_fn=self.llh_transform_hessian_fn)
        return logprior_hessian + llh_hessian
    
    def logtarget_auto_hessian(self, parameter):
        """compute the hessian of the log target density with respect to the parameter and return the hessian, using automatic differentiation. See self.logtarget_hessian"""
        def fn(parameter):
            logtarget_density, _ = self.logtarget_density(parameter=parameter, llh_info_dict=dict())
            return logtarget_density
        return torch.autograd.functional.hessian(fn, parameter)

    def pyro_model(self, data, parameter_len):
        """the equivalent pyro model, for use in pyro MCMC functions
        data: torch.tensor
            - this input is required as a standard format of a pyro model
        parameter_len: int
            - the dimension of the parameter
        """
        prior_parameter = pyro.sample("prior_parameter", dist.MultivariateNormal(torch.zeros(parameter_len), self.prior.covariance_matrix(parameter_len=parameter_len)))
        mean = self.abclikelihood.compute_mean(parameter=prior_parameter, mean_fn=self.llh_transform_fn, llh_info_dict=dict())
        with pyro.plate("data_plate"):
            pyro.sample("obs", dist.MultivariateNormal(mean, self.abclikelihood.covariance_matrix(data_len=len(self.data))), obs=data)


class Kernel:
    """the class of all MCMC kernels"""
    def __init__(self, model, use_autograd=True, use_autohess=True, *args, **kwargs):
        """
        model: a model of the form DeterministicSRModel
            - model that defines the log target density for the MCMC, and should contain methods: logtarget_density that takes a parameter (torch.tensor) and return a torch.tensor
        autograd: bool, optional
            - For kernels that requires gradient. If True, use autograd to compute gradient, otherwise, use the gradient method defined by the model
              In this case, the model should contain methods: logtarget_auto_gradient (autograd), logtarget_gradient
        autohess: bool, optional
            - For kernels that requires Hessian. If True, use autograd to compute hessian, otherwise, use the hessian method defined by the model
              In this case, the model should contain methods: logtarget_auto_hessian (autograd), logtarget_hessian
        """
        self.model = model
        self.original = True #flag to identify whether the kernel subclass is origin
        self.use_autograd = use_autograd
        self.use_autohess = use_autohess

    def move(self, *args, **kwargs):
        raise NotImplementedError

    def propose_accept(self, *args, **kwargs):
        raise NotImplementedError
    
    def warmup(self, *args, **kwargs):
        raise NotImplementedError
    
    def gradient(self, parameter, info_dict=dict(), return_logtarget_density=True):
        """compute the gradient of the target density with respect to the parameter, with an option to return the logtarget density
        parameter: torch.tensor
            - parameter for which the gradient is computed
        info dict: dict, of the form {"llh_info_dict": dict, "logdensities": torch.tensor, "gradient": torch.tensor}, optional
            - "llh_info_dict": a dict to be passed in to the logtarget_density; "logdensities": log density of the corresponding parameter; "gradient": gradient of the log target density with respect to the parameter
        return_logtarget_density: bool, optional
            - If True, logtarget_density is computed and returned
        return:
            - gradient: torch.tensor
            - logtarget_density: torch.tensor
                - It is returned if return_logtarget_density is true
            - llh_info_dict: dict
                - The llh_info_dict returned by the the logtarget_density as a by-product
        """

        llh_info_dict = info_dict["llh_info_dict"] if info_dict.get("llh_info_dict") is not None else dict()
        logtarget_density = info_dict.get("logdensities")
        llh_grad_info_dict = dict()

        if info_dict.get("gradient") is not None:
            gradient = info_dict["gradient"]
        else:
            if self.use_autograd:
                logtarget_density, gradient, llh_info_dict = self.model.logtarget_auto_gradient(parameter=parameter) #llh_info_dict must be none to compute the gradient correctly
            else:
                gradient, llh_grad_info_dict = self.model.logtarget_gradient(parameter=parameter, llh_info_dict=llh_info_dict)
        
        if return_logtarget_density is True:
            if logtarget_density is None or llh_info_dict is None:
                logtarget_density, llh_info_dict = self.model.logtarget_density(parameter=parameter, llh_info_dict=llh_info_dict)
            return gradient, logtarget_density, llh_info_dict, llh_grad_info_dict
        
        return gradient, llh_grad_info_dict

    def hessian(self, parameter, info_dict=dict()):
        """compute the hessian of the target density with respect to the parameter. See self.gradient"""
        llh_info_dict = info_dict["llh_info_dict"] if info_dict.get("llh_info_dict") is not None else dict()
        llh_grad_info_dict = info_dict["llh_grad_info_dict"] if info_dict.get("llh_grad_info_dict") is not None else dict()
        if self.use_autohess:
            hessian = self.model.logtarget_auto_hessian(parameter=parameter)
        else:
            hessian = self.model.logtarget_hessian(parameter=parameter, llh_info_dict=llh_info_dict, llh_grad_info_dict=llh_grad_info_dict)
        return hessian

    def negative_inverse_hessian(self, parameter, info_dict=dict()):
        """compute the precondition matrix, negative of the inverse of the hessian of the target density with respect to the parameter. See self.gradient"""
        if info_dict.get("precondition") is not None:
            return info_dict.get("precondition")
        else:
            hessian = self.hessian(parameter=parameter, info_dict=info_dict)
            return - torch.linalg.inv(hessian) + torch.eye(hessian.size()[0]) * 1.e-4


class RandomWalk(Kernel):
    """The randomwalk MCMC kernel"""
    def __init__(self, model, stepsize=1., covariance_matrix=None, *args, **kwargs):
        """
        model: see the class Kernel
        stepsize: float, optional
            - the standard deviation of the Gaussian noise of the move, or the multiplicative factor of the squart root of the covariance matrix
        covariance_matrix: torch.tensor, optional
            - the covariance of the gaussian noise, before being multiplied by stepsize ** 2. If None, dimension indpendent noise is used, with standard deviation stepsize
        """
        print('RandomWalk stepsize', stepsize)
        super(RandomWalk, self).__init__(model=model, *args, **kwargs)   
        self.stepsize = stepsize
        self.covariance_matrix = covariance_matrix

    def move(self, current_para):
        """propose a proposal from the current parameter
        current_para: torch.tensor
            - the parameter to be moved
        return:
            - proposed_para: torch.tensor
                - a proposal move from the current_para
        """
        if self.covariance_matrix is not None:
            return torch.distributions.multivariate_normal.MultivariateNormal(loc=current_para, covariance_matrix= (self.stepsize ** 2) * self.covariance_matrix).sample()
        else:
            return torch.normal(mean=current_para, std=self.stepsize)

    def propose_accept(self, current_para, current_para_info_dict=dict()):
        """propose a proposal from the current parameter and accept according to the acceptance probability
        current_para: torch.tensor
            - the parameter to be moved
        current_para_info_dict: dict of the form {"llh_info_dict": dict, "logdensities": torch.tensor}, optional
            - "llh_info_dict": a dict to be passed in to the logtarget_density; "logdensities": log density of the corresponding parameter;
        return:
            - accept_prob: torch.tensor
                - the acceptance probability of the proposal
            - proposed_para: torch.tensor
                - proposed parameter
            - proposed_para_info_dict: the corresponding "current_para_info_dict" for the proposed parameter
        """

        current_para_llh_info_dict = current_para_info_dict["llh_info_dict"] if current_para_info_dict.get("llh_info_dict") is not None else dict()
        proposed_para = self.move(current_para)

        if current_para_info_dict.get("logdensities") is not None:
            current_logtarget_density = current_para_info_dict["logdensities"]
        else:
            current_logtarget_density, _ = self.model.logtarget_density(parameter=current_para, llh_info_dict=current_para_llh_info_dict)

        proposed_logtarget_density, proposed_para_llh_info_dict = self.model.logtarget_density(parameter=proposed_para, llh_info_dict=dict())
        accept_prob = np.exp(torch_max_0(proposed_logtarget_density - current_logtarget_density))

        proposed_para_info_dict = {"llh_info_dict": proposed_para_llh_info_dict, "logdensities": proposed_logtarget_density}

        return accept_prob, proposed_para, proposed_para_info_dict #acceptance_prob, proposal, extra_stat_for_proposal
    
class pCN(Kernel):
    """The pCN kernel"""
    def __init__(self, model, stepsize=1., *args, **kwargs):
        """
        model: see Kernel()
        stepsize: float
            - the multiplicative factor of the squart root of the covariance matrix
        """
        print('pCN stepsize', stepsize)
        super(pCN, self).__init__(model=model, *args, **kwargs)   
        self.stepsize = stepsize
        self.covariance = self.model.prior.covariance_matrix
        if is_diagonal(self.covariance):
            self.multidim = False
        else:
            self.multidim = True
        
    def move(self, current_para):
        """proposed a move according to the pCN kernel"""
        if self.multidim is True:
            return np.sqrt(1 - self.stepsize ** 2) * current_para + self.stepsize * torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.zeros(current_para.size()), covariance_matrix=self.covariance)
        else:
            return np.sqrt(1 - self.stepsize ** 2) * current_para + self.stepsize * torch.normal(mean=torch.zeros(current_para.size()), std=torch.diagonal(self.covariance))
    
    def propose_accept(self, current_para, current_para_info_dict=dict()):
        """propose a proposal from the current parameter and accept according to the acceptance probability"""

        current_para_llh_info_dict = current_para_info_dict["llh_info_dict"] if current_para_info_dict.get("llh_info_dict") is not None else dict()
    
        proposed_para = self.move(current_para)

        if current_para_info_dict.get("logdensities") is not None:
            current_llh = current_para_info_dict["logdensities"]
        else:
            current_llh, _ = self.model.llh(parameter=current_para, llh_info_dict=current_para_llh_info_dict)
        proposed_llh, proposed_para_llh_info_dict = self.model.llh(parameter=proposed_para, llh_info_dict=dict())
        accept_prob = np.exp(torch_max_0(proposed_llh - current_llh))

        proposed_para_info_dict = {"llh_info_dict": proposed_para_llh_info_dict, "logdensities":proposed_llh}

        return accept_prob, proposed_para, proposed_para_info_dict #acceptance_prob, proposal, extra_stat_for_proposal

    
class MALA(Kernel):
    """The MALA Kernel"""
    def __init__(self, model, stepsize=0.5, precondition_matrix=None, use_autograd=True, *args, **kwargs):
        """
        model: see RandomWalk
        stepsize: float
            - the standard deviation of the Gaussian noise of the move, or the multiplicative factor of the squart root of the precondition matrix
        precondition_matrix: torch.tensor
            - A fixed precondition matrix for the Langevin dynamics, M^{-1}
        use_autograd: bool
            - If True, autograd is used to compute the gradient of the target density, otherwise the manually implemented gradient function is used specified in the model
        """
        print('MALA stepsize', stepsize)
        super(MALA, self).__init__(model=model, use_autograd=use_autograd, *args, **kwargs)   
        self.stepsize = stepsize
        self.precondition = precondition_matrix
        self.use_autograd = use_autograd

    def move(self, current_para, current_gradient):
        "see self.move_with_precondition, using self.precondition as the precondition"
        if self.precondition is None:
            proposed_para = current_para + ((self.stepsize ** 2) / 2) * current_gradient + self.stepsize *  torch.normal(mean=torch.zeros(current_para.size()), std=1.)
        else:
            proposed_para = self.move_with_precondition(current_para=current_para, current_gradient=current_gradient, current_precondition=self.precondition)
        return proposed_para
    
    def move_with_precondition(self, current_para, current_gradient, current_precondition):
        """proposed a move according to the MALA kernel, with an option for preconditioning
        current_para: torch.tensor
            - the current parameter
        current_gradient: torch.tensor
            - the gradient of the target density with respect to the parameter at the current parameter
        current_precondition: torch.tensor
            - the precondition matrix of the current parameter
        return: 
            proposed_para: torch.tensor
                - the proposed parameter
        """
        noise = torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.zeros(current_para.size()), covariance_matrix=current_precondition).sample()
        proposed_para = current_para + ((self.stepsize ** 2) / 2) * (current_precondition @ current_gradient) + self.stepsize * noise
        return proposed_para
    
    def move_ratio(self, current_para, current_gradient, proposed_para, proposed_gradient):
        """self.move_ratio_with_precondition with constant precondition, self.precondition. See move_ratio_with_precondition"""
        if self.precondition is None:
            curr_to_prop = torch.distributions.normal.Normal(loc=current_para + ((self.stepsize ** 2) / 2) * current_gradient, scale=self.stepsize).log_prob(proposed_para).sum()
            prop_to_curr = torch.distributions.normal.Normal(loc=proposed_para + ((self.stepsize ** 2) / 2) * proposed_gradient, scale=self.stepsize).log_prob(current_para).sum()
            return curr_to_prop - prop_to_curr
        else:
            return self.move_ratio_with_precondition(current_para=current_para, current_gradient=current_gradient, current_precondition=self.precondition,
                                                     proposed_para=proposed_para, proposed_gradient=proposed_gradient, proposed_precondition=self.precondition)
    
    def move_ratio_with_precondition(self, current_para, current_gradient,  current_precondition, proposed_para, proposed_gradient, proposed_precondition):
        """
        current_para: torch.tensor
            - the current parameter
        current_gradient: torch.tensor
            - the gradient of the target density with respect to the parameter at the current parameter
        current_precondition: torch.tensor
            - the precondition matrix of the current parameter
        proposed_para: torch.tensor
            - the proposed parameter
        proposed_gradient: torch.tensor
            - the gradient of the target density with respect to the parameter at the proposed parameter
        proposed_precondition: torch.tensor
            - the precondition matrix of the proposed parameter
        return:
            - move_ratio: torch.tensor
                - the log ratio log(p(proposed_para|current_para)) - log(p(current_para|proposed_para))
        """

        curr_to_prop = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=current_para + ((self.stepsize ** 2) / 2) * (current_precondition @ current_gradient), 
            covariance_matrix=current_precondition * (self.stepsize ** 2)
            ).log_prob(proposed_para).sum()
        prop_to_curr = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=proposed_para + ((self.stepsize ** 2) / 2) * (proposed_precondition @ proposed_gradient), 
            covariance_matrix=proposed_precondition * (self.stepsize ** 2)
            ).log_prob(current_para).sum()
        return curr_to_prop - prop_to_curr

    def propose_accept(self, current_para, current_para_info_dict=dict()):
        """see RandomWalk"""
        current_gradient, current_logtarget_density, *_ = self.gradient(parameter=current_para, info_dict=current_para_info_dict, return_logtarget_density=True)

        proposed_para = self.move(current_para=current_para, current_gradient=current_gradient)
        proposed_gradient, proposed_logtarget_density, proposed_para_llh_info_dict, proposed_para_llh_grad_info_dict = self.gradient(parameter=proposed_para, info_dict=dict(), return_logtarget_density=True)

        move_ratio = self.move_ratio(current_para=current_para, current_gradient=current_gradient, proposed_para=proposed_para, proposed_gradient=proposed_gradient)
        accept_prob = np.exp(torch_max_0(proposed_logtarget_density - current_logtarget_density - move_ratio))
        proposed_para_info_dict = {"llh_info_dict": proposed_para_llh_info_dict, "gradient": proposed_gradient, "logdensities":proposed_logtarget_density, "llh_grad_info_dict": proposed_para_llh_grad_info_dict}

        return accept_prob, proposed_para, proposed_para_info_dict #acceptance_prob, proposal, extra_stat_for_proposal


class mMALA(MALA):
    """Riemann manifold MALA - only works when the gradient of the precondition is zero"""
    def __init__(self, model, stepsize=0.5, use_autograd=True, use_autohess=True, *args, **kwargs):
        """see MALA and Kernel"""
        print('mMALA stepsize', stepsize)
        super(mMALA, self).__init__(model=model, stepsize=stepsize, use_autograd=use_autograd, use_autohess=use_autohess, *args, **kwargs)

    def move(self, current_para, current_gradient, current_precondition):
        """propose move. See MALA.move_with_precondition"""
        return self.move_with_precondition(current_para=current_para, current_gradient=current_gradient, current_precondition=current_precondition)
    
    def move_ratio(self, current_para, current_gradient,  current_precondition, proposed_para, proposed_gradient, proposed_precondition):
        """see MALA.move_ratio_with_precondition"""
        return self.move_ratio_with_precondition(current_para=current_para, current_gradient=current_gradient, current_precondition=current_precondition,
                                                 proposed_para=proposed_para, proposed_gradient=proposed_gradient, proposed_precondition=proposed_precondition)

    def propose_accept(self, current_para, current_para_info_dict=dict()):
        """see MALA.propose_accept"""
        current_gradient, current_logtarget_density, *_ = self.gradient(parameter=current_para, info_dict=current_para_info_dict, return_logtarget_density=True)
        current_precondition = self.negative_inverse_hessian(parameter=current_para, info_dict=current_para_info_dict)

        proposed_para = self.move(current_para=current_para, current_gradient=current_gradient, current_precondition=current_precondition)
        proposed_gradient, proposed_logtarget_density, proposed_para_llh_info_dict, proposed_para_llh_grad_info_dict = self.gradient(parameter=proposed_para, info_dict=dict(), return_logtarget_density=True)
        proposed_para_info_dict = {"llh_info_dict": proposed_para_llh_info_dict, "gradient": proposed_gradient, "logdensities":proposed_logtarget_density, "llh_grad_info_dict": proposed_para_llh_grad_info_dict}
        proposed_precondition = self.negative_inverse_hessian(parameter=proposed_para, info_dict=proposed_para_info_dict)
        proposed_para_info_dict["precondition"] = proposed_precondition

        move_ratio = self.move_ratio(current_para=current_para, current_gradient=current_gradient, current_precondition=current_precondition,
                                     proposed_para=proposed_para, proposed_gradient=proposed_gradient, proposed_precondition=proposed_precondition)
        accept_prob = np.exp(torch_max_0(proposed_logtarget_density - current_logtarget_density - move_ratio))

        return accept_prob, proposed_para, proposed_para_info_dict #acceptance_prob, proposal, extra_stat_for_proposal


class HMC(Kernel):
    def __init__(self, model, traj_len=2*np.pi, num_steps=None, stepsize=0.5, precondition_matrix=None, use_autograd=True, *args, **kwargs):
        """HMC kernel with precondition matrix
        model: see Kernel()
        traj_len:
            - the trajectory length to integrate over. 
        num_steps: int, optional
            - the number of Leapfrog steps to perform. If None, num_steps = ceil(traj_len / stepsize). If traj_len is not None, this parameter is disabled
        stepsize: float
            - the trajectory length for each Leapfrog step
        precondition_matrix: torch.tensor
            - A fixed precondition matrix for the Hamiltonian dynamics, M^{-1}
        use_autograd: bool
            - If True, autograd is used to compute the gradient of the target density, otherwise the manually implemented gradient function is used specified in the model
        """
        print('HMC stepsize', stepsize)
        super(HMC, self).__init__(model=model, *args, **kwargs)   
        self.stepsize = stepsize
        self.precondition = precondition_matrix
        self.use_autograd = use_autograd
    
        self.set_L(num_steps=num_steps, traj_len=traj_len, stepsize=stepsize, set_L=True)


    def move(self, current_para, current_gradient):
        """see self.move_, with L=self.L, stepsize=self.stepsize"""
        return self.move_(current_para=current_para, current_gradient=current_gradient, L=self.L, stepsize=self.stepsize)

    def move_(self, current_para, current_gradient, L=1, stepsize=0.01):
        """
        L: int
            - number of Leapfrog steps
        return:
            - q: torch.tensor
                - the final proposed position
            - p0: torch.tensor
                - the initial proposed momentum
            - p: torch.tensor
                - the final proposed momentum
            - q_info_dict: dict
                - the info dict returned by the log likelihood function at q, see Kernel().gradient
        """
        if self.precondition is not None:
            p0 = torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.zeros(current_para.size()), precision_matrix=self.precondition).sample()
        else:
            p0 = torch.normal(mean=torch.zeros(current_para.size()), std=1.)

        p = p0 + stepsize * current_gradient * 0.5
        q = current_para

        for i in range(L):
            q_move = torch.mv(self.precondition, p) if self.precondition is not None else p
            q = q + stepsize * q_move
            if i != (L-1):
                gradient, _ = self.gradient(parameter=q, info_dict=dict(), return_logtarget_density=False)
                p = p + stepsize * gradient
        proposed_gradient, proposed_logtarget_density, proposed_para_llh_info_dict, proposed_para_llh_grad_info_dict = self.gradient(parameter=q, info_dict=dict(), return_logtarget_density=True)
        
        p = p + stepsize * proposed_gradient * 0.5
        p = -p
        
        q_info_dict = {"logdensities":proposed_logtarget_density, "gradient":proposed_gradient, "llh_info_dict":proposed_para_llh_info_dict, "llh_grad_info_dict":proposed_para_llh_grad_info_dict}
        return q, p0, p, q_info_dict

    def propose_accept(self, current_para, current_para_info_dict=dict()):
        """see RandomWalk"""
        return self.propose_accept_(current_para=current_para, current_para_info_dict=current_para_info_dict, L=self.L, stepsize=self.stepsize)

    def propose_accept_(self, current_para, L=1, stepsize=0.01, current_para_info_dict=dict()):
        """see RandomWalk"""
        current_gradient, current_logtarget_density, *_ = self.gradient(parameter=current_para, info_dict=current_para_info_dict, return_logtarget_density=True)

        proposed_para, p0, p, q_info_dict = self.move_(current_para=current_para, current_gradient=current_gradient, L=L, stepsize=stepsize)
        proposed_logtarget_density = q_info_dict["logdensities"]
        proposed_para_info_dict = q_info_dict

        H_old = self.hamiltonian(q=current_para, p=p0, q_logtarget_density=current_logtarget_density)
        H_new =  self.hamiltonian(q=proposed_para, p=p, q_logtarget_density=proposed_logtarget_density)
        accept_prob = np.exp(torch_max_0(H_old - H_new))

        return accept_prob, proposed_para, proposed_para_info_dict


    def hamiltonian(self, q, p, q_logtarget_density=None):
        """compute the Hamiltonian for q,p. See self.hamiltonian_with_precondition"""
        return self.hamiltonian_with_precondition(q=q, p=p, q_logtarget_density=q_logtarget_density, precondition=self.precondition)

    def hamiltonian_with_precondition(self, q, p, q_logtarget_density=None, precondition=None):
        """compute the Hamiltonian for q,p
        q: torch.tensor
            - the position
        p: torch.tensor
            - the momentum
        q_logtarget_density: torch.tensor, optional
            - the log target density at q
        precondition: torch.tensor, optional
            - the precision matrix of the Gaussian of p
        return:
            - hamiltonian: torch.tensor
                - the hamiltonian of q and p (up to a constant)
        """
        if q_logtarget_density is None:
            q_logtarget_density = self.model.logtarget_density(q)
        p_logdensity = - torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.zeros(p.size()[0]), precision_matrix=precondition).log_prob(p) if precondition is not None else 0.5 * torch.dot(p, p)
        return p_logdensity - q_logtarget_density

    def find_reasonable_stepsize(self, init_para, init_para_info_dict=dict()): #No-U-Turn paper Algorithm 4 - https://arxiv.org/abs/1111.4246
        """find an initial stepsize as the initial stepsize for warmup (adaptive stepsize)
        init_para: torch.tensor
            the initial parameter of the MCMC chain
        init_para_info_dict: torch.tensor
            the info_dict of init_para, see Kernel().gradient and RandomWalk().propose_accept
        return:
            eps: torch.tensor
                a reasonable stepsize
        """

        eps = 1.
        accept_prob, *_ = self.propose_accept_(current_para=init_para, L=1, stepsize=eps, current_para_info_dict=init_para_info_dict)
        a = 2*np.int32(accept_prob > 0.5) - 1

        while accept_prob ** a > 2. ** (-a):
            eps *= 2. ** a
            accept_prob , *_ = self.propose_accept_(current_para=init_para, L=1, stepsize=eps, current_para_info_dict=init_para_info_dict)

        print("initial_stepsize is found as:", eps)
        return eps
    
    def warmup(self, init_para, init_para_info_dict=dict(), iterations=10, set_stepsize=True, target_prob=0.7, auto_init_stepsize=True):
        """perform adapt_stepsize with the optional of using find_reasonable_stepsize
        init_para: torch.tensor
            - the initial parameter of the MCMC chain
        init_para_info_dict: torch.tensor
            - the info_dict of init_para, see Kernel().gradient and RandomWalk().propose_accept
        iterations: int
            - the number of iterations to perform adapt_stepsize
        set_stepsize: bool
            - if True, the stepsize of the class is overwritten by the final stepsize found after warmup is completed
        target_prob: float
            - the target acceptance probability of the stepsize found in adapt_stepsize
        auto_init_stepsize: bool
            - if True, use find_reasonable_stepsize as the initial stepsize for adapt_stepsize, otherwise, use the initial stepsize defined when initialising the class
        """
        return self.adapt_stepsize(init_para=init_para, init_para_info_dict=init_para_info_dict, target_prob=target_prob, auto_init_stepsize=auto_init_stepsize, iterations=iterations, set_stepsize=set_stepsize)

    def adapt_stepsize(self, init_para, init_para_info_dict=dict(), target_prob=0.7, auto_init_stepsize=True, iterations=10, set_stepsize=True): #No-U-Turn paper Algorithm 5 - https://arxiv.org/abs/1111.4246
        """perform dual averaging to find a reasonable stepsize that has an acceptance probability close to target_prob. num_steps is set as self.traj_len/stepsize of a particular iteration. See self.warmup"""

        if auto_init_stepsize is True:
            stepsize = self.find_reasonable_stepsize(init_para=init_para, init_para_info_dict=init_para_info_dict)
        else:
            stepsize = self.stepsize

        current_para = init_para
        current_para_info_dict = init_para_info_dict

        eps = stepsize
        mu = np.log(10*eps)
        logeps_bar = 0
        H_bar = 0
        gamma = 0.05
        t0 = 10
        kappa = 0.75

        pbar = tqdm(range(iterations))
        for i in pbar:
            L = self.set_L(num_steps=None, traj_len=self.traj_len, stepsize=eps, set_L=False)
            try:
                accept_prob, proposed_para, proposed_para_info_dict  = self.propose_accept_(current_para=current_para, 
                                                                            current_para_info_dict=current_para_info_dict,
                                                                            L=L,
                                                                            stepsize=eps)
            except ValueError: #parameter diverge
                accept_prob = torch.tensor(0.)

            if np.random.uniform(0,1) < accept_prob:
                current_para = proposed_para
                current_para_info_dict = proposed_para_info_dict

            m = i+1
            H_bar = (1 - 1/(m+t0)) * H_bar + (1/(m+t0)) * (target_prob - accept_prob)
            logeps = mu - (m**0.5)/gamma * H_bar
            eps = np.exp(logeps)
            logeps_bar = m**(-kappa) * logeps + (1 - m**(-kappa)) * logeps_bar

            pbar.set_description("Warmup: Most recent alpha {}, with stepsize {}".format(str(np.round(accept_prob.numpy(),2)), str(np.round(eps,2))))

        stepsize = np.exp(logeps_bar)
        if set_stepsize is True:   
            self.set_stepsize(stepsize=stepsize)
            print("HMC stepsize set up {}".format(self.stepsize))

        return current_para, current_para_info_dict, stepsize
    
    def set_stepsize(self, stepsize):
        """set stepsize
        stepsize: float
            - step size
        """
        self.stepsize = stepsize

    def set_L(self, num_steps, traj_len, stepsize, set_L=True):
        """set number of Leapfrog steps (L). 
        num_steps: int
            - number of Leapfrog steps. If traj_len is not None, compute num_steps = np.ceil(traj_len / stepsize), otherwise set L=stepsize
        traj_len: float
            - total length of integration by Leapfrog
        stepsize: float
            - length of one Leapfrog integration
        set_L: bool
            - If true, L is overwritten by the L computed
        return:
            - num_steps: int
        """

        if traj_len is not None:
            num_steps = np.int64(np.ceil(traj_len / stepsize))
            if set_L is True:
                self.L = num_steps
            return num_steps
        else:
            if set_L is True:
                self.L = num_steps
            return num_steps

class mHMC(HMC):
    def __init__(self, model, traj_len=2*np.pi, num_steps=None, stepsize=0.5, fp_iterations=5, use_autograd=True, use_autohess=True, *args, **kwargs):
        """Riemann manifold HMC - only works when the gradient of the precondition is zero
        fp_iterations: int
            - number of fixed point iterations to run for the implicit integrator
        """
        print('mHMC stepsize', stepsize)
        super(mHMC, self).__init__(model=model, traj_len=traj_len, num_steps=num_steps, stepsize=stepsize, use_autograd=use_autograd, use_autohess=use_autohess, *args, **kwargs)
        self.fp_iterations = fp_iterations

    def q_move(self, q, q_precondition, p, fp_iterations=5, stepsize=0.01):
        """propose a new q from an old q via fixed point iterations
        q: torch.tensor
            - the current position
        q_precondition: torch.tensor
            - the precondition matrix of the current position
        p: torch.tensor
            - the current momentum
        fp_iterations: int
            - the number of fixed point iterations to take for the integrator
        """
        q_prime = q
        q_prime_precondition = q_precondition
        halfstep = 0.5 * stepsize * torch.mv(q_precondition, p) 
        for i in range(fp_iterations):
            q_prime = q + halfstep + 0.5 * stepsize * torch.mv(q_prime_precondition, p)
            q_prime_precondition = self.negative_inverse_hessian(parameter=q_prime, info_dict=dict())
        #     suberror = q_prime - halfstep - 0.5 * stepsize * torch.mv(q_prime_precondition, p)
        #     print("suberror", np.sqrt((suberror ** 2).sum().numpy() ))
        #     print(np.sqrt(q_prime **2).sum().numpy())
        
        # error = q_prime - halfstep - 0.5 * stepsize * torch.mv(q_prime_precondition, p)
        # print("error", np.sqrt((error ** 2).sum().numpy() ))
        return q_prime, q_prime_precondition
    
    def move(self, current_para, current_gradient, current_precondition):
        """see self.move_"""
        return self.move(current_para=current_para, current_gradient=current_gradient, current_precondition=current_precondition, L=self.L, stepsize=self.stepsize)

    def move_(self, current_para, current_gradient, current_precondition, L=1, stepsize=0.01):
        """see HMC.move_
        current_precondition: torch.tensor
            - the precondition matrix of the current parameter
        """
        p0 = torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.zeros(current_para.size()), precision_matrix=current_precondition).sample()

        p = p0 + stepsize * current_gradient * 0.5
        q = current_para
        q_precondition = current_precondition

        for i in range(L):
            q, q_precondition = self.q_move(q=q, q_precondition=q_precondition, p=p, fp_iterations=self.fp_iterations, stepsize=stepsize)
            if i != (L-1):
                gradient, _ = self.gradient(parameter=q, info_dict=dict(), return_logtarget_density=False)
                p = p + stepsize * gradient

        proposed_precondition = q_precondition
        proposed_gradient, proposed_logtarget_density, proposed_para_llh_info_dict, proposed_para_llh_grad_info_dict = self.gradient(parameter=q, info_dict=dict(), return_logtarget_density=True)

        p = p + stepsize * proposed_gradient * 0.5
        p = -p
        
        q_info_dict = {"logdensities":proposed_logtarget_density, "gradient":proposed_gradient, "llh_info_dict":proposed_para_llh_info_dict, "llh_grad_info_dict":proposed_para_llh_grad_info_dict, "precondition":proposed_precondition}
        return q, p0, p, q_info_dict

    def propose_accept_(self, current_para, L=1, stepsize=0.01, current_para_info_dict=dict()):
        """see HMC.propose_accept_"""

        current_gradient, current_logtarget_density, *_ = self.gradient(parameter=current_para, info_dict=current_para_info_dict, return_logtarget_density=True)
        current_precondition = self.negative_inverse_hessian(parameter=current_para, info_dict=current_para_info_dict)

        proposed_para, p0, p, q_info_dict = self.move_(current_para=current_para, current_gradient=current_gradient, current_precondition=current_precondition, L=L, stepsize=stepsize)
        proposed_logtarget_density = q_info_dict["logdensities"]
        proposed_para_info_dict = q_info_dict
        proposed_precondition = q_info_dict["precondition"]

        H_old = self.hamiltonian_with_precondition(q=current_para, p=p0, q_logtarget_density=current_logtarget_density, precondition=current_precondition)
        H_new =  self.hamiltonian_with_precondition(q=proposed_para, p=p, q_logtarget_density=proposed_logtarget_density, precondition=proposed_precondition)
        accept_prob = np.exp(torch_max_0(H_old - H_new))

        return accept_prob, proposed_para, proposed_para_info_dict
            

class HMC_pyro(Kernel):
    """The HMC kernel for use in pyro"""
    def __init__(self, model, stepsize=0.5, *args, **kwargs):
        print('HMC stepsize', stepsize)
        super(HMC_pyro, self).__init__(model=model, *args, **kwargs)
        self.stepsize = stepsize
        self.kwargs = kwargs
        self.original = False  #flag to identify whether the kernel subclass is origin
        
    def get_pyro_kernel(self, parameter_len):
        """return the HMC pyro kernel with input parameters specified during initialisation of the class
        parameter_len: len
            - the dimension of the sampling (parameter) space
        """
        pyro.clear_param_store()
        pyro_model = lambda data: self.model.pyro_model(data=data, parameter_len=parameter_len)
        self.pyro_kernel =  pyro.infer.mcmc.HMC(model=pyro_model, step_size=self.stepsize, **self.kwargs)
        return self.pyro_kernel


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
    """the class to run MCMC"""
    def __init__(self, kernel, warmup_steps=0, num_samples=MCMC_SAMPLE, initial_params=None, params_dim=None, warmup_settings=dict(target_prob=0.7, auto_init_stepsize=True), **kwargs):
        """
        kernel: a Kernel() class instance (note that a pyro kernel instance does not work)
            - the kernel instance to propose move for the MCMC and provide acceptance probability of the move
        warmup_steps: int or None
            - the number of dual averaging adaptive stepsize steps to run. It is only available for the HMC kernel
        num_samples: int
            - the number of samples from the MCMC run excluding the warup_steps
        initial_params: None or torch.tensor
            - the initial sample of the MCMC. If None, it is initialised as zero.
        params_dim: None or int
            - the dimension of the sampling space. If initial_params is None, this has to be specified, otherwise it is inferred from initial_params
        warmup_settings: dict, of the form {"target_prob": float, "auto_init_stepsize": int, ...}
            - the extra parameters to be passed into HMC.warmup
        """
        self.num_samples = num_samples
        self.kernel = kernel
        assert initial_params is not None or params_dim is not None, "Should either specify initial_params or params_dim"
        if initial_params is not None:
            self.initial_params = initial_params
            self.params_dim = len(initial_params)
        else:
            self.initial_params = torch.zeros(params_dim)
            self.params_dim = params_dim

        assert warmup_steps is None or isinstance(warmup_steps,int) or isinstance(warmup_steps, np.integer), "warmup_steps must be None or integer"
        self.warmup_steps = 0 if warmup_steps is None else warmup_steps
        self.warmup_settings = warmup_settings

        self.reset_stat()

    def run(self):

        self.reset_stat()

        current_para = self.initial_params
        current_logtarget_density, current_para_llh_info_dict  = self.kernel.model.logtarget_density(parameter=current_para, llh_info_dict=dict())
        current_para_info_dict = {"logdensities":current_logtarget_density, "llh_info_dict":current_para_llh_info_dict}

        if self.warmup_steps > 0:
            try: 
                current_para, current_para_info_dict, _ = self.warmup(init_para=current_para, init_para_info_dict=current_para_info_dict)
            except NotImplementedError:
                print("Warmup is not implemented for the current kernel. Skip to sampling...")

        self.samples[0] = current_para
        self.logdensities[0] = current_logtarget_density
        self.proposed_logdensities[0] = current_logtarget_density

        pbar = tqdm(range(self.num_samples))
        for i in pbar:
            accept_prob, proposed_para, proposed_para_info_dict  = self.kernel.propose_accept(current_para=current_para,
                                                                         current_para_info_dict=current_para_info_dict)

            if np.random.uniform(0,1) < accept_prob:
                current_para = proposed_para
                current_para_info_dict = proposed_para_info_dict
                self.accepted += 1

            pbar.set_description("Acceptance probability {}".format(np.round(self.accepted/(i+1), 2)))

            self.samples[i+1] = current_para
            self.logdensities[i+1] = current_para_info_dict["logdensities"]
            self.proposed_logdensities[i+1] = proposed_para_info_dict["logdensities"]
            self.accept_prob[i+1] = accept_prob

        return self.samples
    
    def warmup(self, init_para, init_para_info_dict=dict()):
        current_para, current_para_info_dict, info = self.kernel.warmup(init_para=init_para, 
                                                                        init_para_info_dict=init_para_info_dict,
                                                                        iterations=self.warmup_steps, 
                                                                        set_stepsize=True,
                                                                        **self.warmup_settings)

        return current_para, current_para_info_dict, info

    def set_initial_params(self, initial_params):
        self.initial_params = initial_params
    
    def get_samples(self):
        return self.samples
    
    def get_logdensities(self):
        return self.logdensities
    
    def get_proposed_logdensities(self):
        return self.proposed_logdensities
    
    def get_accept_prob(self):
        return self.accept_prob
    
    def reset_stat(self):
        self.samples = torch.zeros((self.num_samples+1, self.params_dim))
        self.logdensities = torch.zeros(self.num_samples+1)
        self.proposed_logdensities = torch.zeros(self.num_samples+1)
        self.accepted = 0
        self.accept_prob = torch.zeros(self.num_samples+1)


class MCMC_pyro(MCMC):
    """the class to run MCMC for pyro kernel"""
    def __init__(self, kernel, num_samples=MCMC_SAMPLE, initial_params=None, params_dim=None, warmup_steps=MCMC_T//5, disable_progbar=MCMC_SHOW_DISABLE, **kwargs):
        """
        kernel: class instance of HMC_pyro kernel
            - MCMC kernel from pyro
        See MCMC()
        """
        super(MCMC_pyro, self).__init__(kernel=kernel, warmup_steps=warmup_steps, num_samples=num_samples, initial_params=initial_params, params_dim=params_dim, **kwargs)
        self.kernel_ = kernel
        self.pyro_kernel = kernel.get_pyro_kernel(parameter_len=self.params_dim)
        self.pyro_mcmc = pyro.infer.mcmc.MCMC(kernel=self.pyro_kernel, num_samples=num_samples, initial_params={'prior_parameter': initial_params}, warmup_steps=warmup_steps, disable_progbar=disable_progbar, **kwargs)
        self.data = self.kernel.model.data

    def run(self):

        self.reset_stat()

        self.pyro_mcmc.run(self.data)
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
            posterior_samples = model.get_parameter()
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
                        
                        llh_transform_grad_fn = lambda parameter:  tabular_indicator(para=parameter.reshape(env.n_cell + (env.action_space.n, )), model=model, obs=obs._buffers)#standard form

                        prior = IsotropicGaussianPrior(sd=PRIOR_SIGMA)
                        abclikelihood = GaussianABCLikelihood(epsilon=EPSILON)
                        data = torch.tensor(obs._buffers["rewards"])[-BUFFER_SIZE:][batch_indices]
                        Model = DeterministicSRModel(prior=prior, abclikelihood=abclikelihood, data=data, llh_transform_fn=r_hat, llh_transform_grad_fn=llh_transform_grad_fn)

                        def fn(parameter):
                            current_logtarget_density, _ = Model.logtarget_density(parameter=parameter, llh_info_dict=dict())
                            return current_logtarget_density
                        hessian = torch.autograd.functional.hessian(fn,posterior_samples[0].reshape(-1))
                        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE)
                        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE, covariance_matrix=-torch.linalg.inv(hessian))
                        #kernel = pCN(model=Model, stepsize=STEPSIZE)
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=None)
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=-torch.linalg.inv(hessian))
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=False)
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=False, precondition_matrix=-torch.linalg.inv(hessian))
                        #kernel = HMC_pyro(model=Model, stepsize=STEPSIZE, full_mass=FULL_MASS, adapt_step_size=ADAPT_STEP_SIZE, adapt_mass_matrix=ADAPT_MASS_MATRIX, target_accept_prob=TARGET_ACCEPT_PROB, num_steps=NUM_STEPS)
                        #kernel = HMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=False)
                        #kernel = HMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=False, precondition_matrix=-torch.linalg.inv(hessian), traj_len=None)
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

    
    
        
    