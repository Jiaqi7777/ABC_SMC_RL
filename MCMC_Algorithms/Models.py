import numpy as np
import scipy.stats as stats
import torch
import pyro
import pyro.distributions as dist
from parameter import *
from parameter import *

def generate_samples(para, model, obs, batch_indices=None, buffer_size=BUFFER_SIZE, batch_training=BATCH_TRAINING):
    para = para.reshape(model.state_size + (model.action_size, ))
    if not batch_training:
        batch_indices = range(min(len(obs['state0']), buffer_size))
    
    s0 = np.array(obs['state0'])[-buffer_size:][batch_indices]
    s1 = np.array(obs['state1'])[-buffer_size:][batch_indices]
    a = np.array(obs['action'])[-buffer_size:][batch_indices]
    dones = torch.tensor(np.array(obs['done'])[-buffer_size:][batch_indices].astype(int))

    return model.q_value(para, s0.T, a) - torch.where(dones == 1, torch.zeros(len(s0)), model.gamma * model.v_value(para, s1.T).values) #time 

def generate_z(indices=slice(None), obs=None, env=None):
    s0 = np.array(obs['state0'])[indices]
    a = np.array(obs['action'])[indices]
    propsoed_z = np.swapaxes(np.array(env.step(action=a, state=s0, multiple=M_Z)[0]), 0, 1)
    return propsoed_z

def generate_samples_with_z(para, model, obs, batch_indices=None, buffer_size=BUFFER_SIZE, batch_training=BATCH_TRAINING):
    '''
    para is [para, z]'''
    if not batch_training:
        batch_indices = range(min(len(obs['state0']), buffer_size))
    
    s0 = np.array(obs['state0'])[-buffer_size:][batch_indices]
    a = np.array(obs['action'])[-buffer_size:][batch_indices]

    para, s1_lst = para
    para = para.reshape(model.state_size + (model.action_size, ))
    # else:
    #     s1_lst = np.swapaxes(np.array(obs['state1'])[-buffer_size:][batch_indices], 0, 1)
    dones = torch.tensor(np.array(obs['done'])[-buffer_size:][batch_indices].astype(int))
    s1_value = 0
    for s1 in s1_lst:
        s1_value += model.v_value(para, s1.T).values
    return model.q_value(para, s0.T, a) - torch.where(dones == 1, torch.zeros(len(s0)), model.gamma * s1_value / M_Z)

def tabular_indicator_deterministic(para, model, obs):
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
    return torch.tensor(Indicator, dtype=torch.float32).reshape((len(a), -1))

def tabular_indicator_stochastic(para, model, obs):
    para, s1_lst = para
    s0 = obs['state0']
    a = obs['action']
    # s1_lst = np.swapaxes(np.array(env.step(action=a, state=s0, multiple=M_Z)[0]), 0, 1)
    done = np.array(obs['done'])
    s01, s02 = np.array(s0).T
    Indicator = np.zeros(shape=(len(a), ) + para.shape) #TxTheta
    Indicator[range(len(a)), s01, s02, a] = 1.
    for s1 in s1_lst:
        s11, s12 = np.array(s1)[done == False].T
        a_prime = np.argmax(para[s11, s12], axis=-1)
        Indicator[range(len(a_prime)), s11, s12, a_prime] -= model.gamma / M_Z
    # for i in range(len(s0)):
    #     for s1 in s1_lst:
    #         a_prime = np.argmax(s1[i][0], s1[i][1])
    #         Indicator[i, s1[i][0], s1[i][1], a_prime] -= model.gamma / M_Z
    return torch.tensor(Indicator, dtype=torch.float32).reshape((len(a), -1))

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
        
        gradient = 1. / (self.epsilon**2) *  torch.mv(torch.t(mean_jacobian), (data - mean))
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
    
    def llh_gradient(self, parameter, llh_info_dict=dict()):
        """compute the gradient of the llh with respect to the parameter and return the gradient, using the explicit derivation of the gradient"""
        llh_grad, llh_grad_info = self.abclikelihood.llh_gradient(data=self.data, parameter=parameter, llh_info_dict=llh_info_dict, llh_transform_fn=self.llh_transform_fn, llh_transform_grad_fn=self.llh_transform_grad_fn)
        return llh_grad, llh_grad_info
    
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
            
class StochasticSModel(DeterministicSRModel):
    """the overall model of the ABC likelihood model with prior and likelihood (with stochastic state transition)"""
    def __init__(self, prior, abclikelihood, data, z_transform_fn=None, llh_transform_fn=None, llh_transform_grad_fn=None, llh_transform_hessian_fn=None, *args):
        super(StochasticSModel, self).__init__(prior, abclikelihood, data, llh_transform_fn=llh_transform_fn, llh_transform_grad_fn=llh_transform_grad_fn, llh_transform_hessian_fn=llh_transform_hessian_fn, *args)
        self.z_transform_fn = z_transform_fn

    def set_var(self, var):
        self.var = var
    
    def logtarget_density(self, parameter, llh_info_dict=dict()):
        """compute the log target density (logprior + llh) given the abclikelihood and prior and return the log target density with the llh_info_dict"""
        logprior = self.logprior(parameter=parameter[0])
        llh, llh_info_dict = self.llh(parameter=parameter, llh_info_dict=llh_info_dict)
        return logprior + llh, llh_info_dict
    
    def logtarget_gradient(self, parameter, llh_info_dict=dict()):
        """compute the gradient of the log target density with respect to the parameter and return the gradient, using the explicit derivation of the gradient"""
        logprior_grad = self.prior.logprior_gradient(parameter=parameter[0])
        llh_grad, llh_grad_info = self.abclikelihood.llh_gradient(data=self.data, parameter=parameter, llh_info_dict=llh_info_dict, llh_transform_fn=self.llh_transform_fn, llh_transform_grad_fn=self.llh_transform_grad_fn)
        return logprior_grad + llh_grad, llh_grad_info
    
    def logtarget_auto_gradient(self, parameter):
        parameter_ = parameter[0].clone()
        parameter_.requires_grad = True
        logtarget_density, llh_info_dict = self.logtarget_density(parameter=[parameter_, parameter[1]], llh_info_dict=dict()) # must use the llh_transform_fn to compute the density
        logtarget_density.backward()
        gradient = parameter_.grad.clone()
        parameter_.grad.zero_()
        parameter_.requires_grad = False
        logtarget_density = logtarget_density.detach()
        return logtarget_density, gradient, llh_info_dict