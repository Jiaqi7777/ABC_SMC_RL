import numpy as np
import math
from scipy.stats import truncnorm
from copy import deepcopy
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from parameter import *

def laplacian(self, x):
    # x: (B,1,H,W)
    xpad = F.pad(x, (1,1,1,1), mode="reflect")  # or "replicate"
    return F.conv2d(xpad, self.kernel.view(1,1,3,3), padding=0)


def generate_samples(para, model, obs, batch_indices=None, buffer_size=BUFFER_SIZE, batch_training=BATCH_TRAINING):
    # print(para.shape, 'generate samples')
    if not batch_training:
        batch_indices = range(min(len(obs['state0']), buffer_size))
    
    s0 = np.array(obs['state0'])[-buffer_size:][batch_indices]
    s1 = np.array(obs['state1'])[-buffer_size:][batch_indices]
    a = np.array(obs['action'])[-buffer_size:][batch_indices]
    dones = torch.tensor(np.array(obs['done'])[-buffer_size:][batch_indices].astype(int))
    try:
        para = para.reshape(model.state_size + (model.action_size, ))
        return model.q_value(para, s0.T, a) - torch.where(dones == 1, torch.zeros(len(s0)), model.gamma * model.v_value(para, s1.T)) #time 
    except:
        # para = para.reshape(env.learnable_shape + (model.action_size, ))
        return model.q_value(para, s0.T, a, full=False) - torch.where(dones == 1, torch.zeros(len(s0)), model.gamma * model.v_value(para, s1.T, full=False)) #time 

def generate_z(indices=slice(None), obs=None, env=None):
    s0 = np.array(obs['state0'])[indices]
    a = np.array(obs['action'])[indices]
    propsoed_z = np.swapaxes(np.array(env.step(action=a, state=s0, multiple=M_Z)[0]), 0, 1)
    return propsoed_z

def generate_samples_with_z(para, model, obs, batch_indices=None, buffer_size=BUFFER_SIZE, batch_training=BATCH_TRAINING, gibbs_indices=None):
    '''
    para is {'para': para, 'z': z]'''
    if gibbs_indices is None:
        gibbs_indices = slice(None)
    # if not batch_training:
    #     batch_indices = range(min(len(obs['state0']), buffer_size))
    
    s0 = np.array(obs['state0'])[-buffer_size:][batch_indices][gibbs_indices]
    a = np.array(obs['action'])[-buffer_size:][batch_indices][gibbs_indices]

    para_ = para['para']
    s1_lst = para['z']
    para_ = para_.reshape(model.state_size + (model.action_size, ))
    # else:
    #     s1_lst = np.swapaxes(np.array(obs['state1'])[-buffer_size:][batch_indices], 0, 1)
    dones = torch.tensor(np.array(obs['done'])[-buffer_size:][batch_indices].astype(int))
    s1_value = 0
    for s1 in s1_lst:
        s1_value += model.v_value(para_, s1.T)
    return model.q_value(para_, s0.T, a) - torch.where(dones == 1, torch.zeros(len(s0)), model.gamma * s1_value / M_Z)

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
    Indicator[np.arange(len(a))[done == False], s11, s12, a_prime] -= model.gamma
    return torch.tensor(Indicator, dtype=torch.float32).reshape((len(a), -1))

def tabular_indicator_stochastic(para, model, obs):
    para_ = para['para']
    s1_lst = para['z']
    s0 = obs['state0']
    a = obs['action']
    # s1_lst = np.swapaxes(np.array(env.step(action=a, state=s0, multiple=M_Z)[0]), 0, 1)
    done = np.array(obs['done'])
    s01, s02 = np.array(s0).T
    Indicator = np.zeros(shape=(len(a), ) + para_.shape) #TxTheta
    Indicator[range(len(a)), s01, s02, a] = 1.
    for s1 in s1_lst:
        s11, s12 = np.array(s1)[done == False].T
        a_prime = np.argmax(para_[s11, s12], axis=-1)
        Indicator[np.arange(len(a))[done == False], s11, s12, a_prime] -= model.gamma / M_Z
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
    
    def sample(self, shape=(1,)):
        """return a sample from the prior given the number of particles
        n_particle: int
            - the number of particles to sample
        """
        return torch.distributions.normal.Normal(loc=self.mean, scale=self.sigma).sample(shape)

class TruncatedGaussianPrior:
    def __init__(self, sd=1., mean=0.):
        """log(p(para)) independent gaussian prior
        sd: float
            - standard deviaition of each dimension
        mean: float
            - mean of each dimension
        """
        self.sigma = float(sd)
        self.mean = mean #before truncation
        
    def logprior(self, parameter):
        parameter = TruncatedGaussianABCLikelihood.neg_exp_transform(parameter)
        """return the log prior of a given parameter
        parameter: torch.tensor
            - the parameter for which the log prior is computed"""
        return 4 * torch.where(parameter <= 0, torch.distributions.normal.Normal(loc=self.mean, scale=self.sigma).log_prob(parameter), torch.full_like(parameter, - float('inf'))).sum(axis=-1)
    
    def logprior_gradient(self, parameter):
        parameter = TruncatedGaussianABCLikelihood.neg_exp_transform(parameter)
        """return the gradient of the log prior with respect to the parameter
        parameter: torch.tensor
            - the parameter for which the log prior gradient is computed
        """
        return torch.where(parameter <= 0, - 4 * parameter / (self.sigma ** 2), torch.full_like(parameter, - float('inf'))) * parameter
    
    def logprior_hessian(self, parameter):
        """return the hessian of the log prior with respect to the parameter
        parameter: torch.tensor
            - the parameter for which the log prior hessian is computed
        """
        return torch.where(parameter <= 0, - 4 / (self.sigma ** 2) * torch.eye(parameter.size()[-1]), torch.full_like(parameter, - float('inf'))) * parameter + torch.ones_like(parameter)
    
    
    def covariance_matrix(self, parameter_len):
        """return the overall covariance matrix of the prior given the dimension of the parameter
        parameter_len: int
            - the dimension of the parameter
        """
        return 4 * (self.sigma ** 2) * torch.eye(parameter_len)

class GaussianMRFPrior:
    def __init__(self, shape, sigma=1.0, unary_mu=0.0, unary_sigma=1.0, device="cpu", dtype=torch.float32):
        self.shape = shape  # (H, W)
        self.sigma = float(sigma)
        self.unary_mu = float(unary_mu)
        self.unary_sigma = float(unary_sigma)
        self.device = device
        self.dtype = dtype

        # 4-neighbour Laplacian kernel
        self.kernel = torch.tensor([[0, -1,  0],
                                    [-1,  4, -1],
                                    [0, -1,  0]], dtype=dtype, device=device)

    # ---------- Laplacian consistent with FFT model (periodic / torus) ----------
    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Discrete Laplacian with circular boundary conditions.
        x: (B, 1, H, W)
        """
        xpad = F.pad(x, (1, 1, 1, 1), mode="circular")
        return F.conv2d(xpad, self.kernel.view(1, 1, 3, 3), padding=0)

    # ---------- Unnormalized log prior and its gradient ----------
    def logprior(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unnormalized log p(x) for the GMRF:
          log p(x) = -0.5 * [ tau * x^T L x + tau_u * ||x - mu||^2 ] + const
        x: (B, 1, H, W)
        returns: (B,)
        """
        tau = 1.0 / (self.sigma ** 2)
        tau_u = 1.0 / (self.unary_sigma ** 2)

        lap = self.laplacian(x)  # (B,1,H,W)

        # x^T L x = sum_{i,j} x_{ij} (Lx)_{ij}
        quad_pair = torch.sum(x * lap, dim=(1, 2, 3))

        # ||x - mu||^2
        diff = x - self.unary_mu
        quad_unary = torch.sum(diff * diff, dim=(1, 2, 3))

        energy = 0.5 * (tau * quad_pair + tau_u * quad_unary)
        return -energy  # + const

    def logprior_grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient of logprior wrt x:
          ∇ log p(x) = - ( tau * Lx + tau_u * (x - mu) )
        x: (B, 1, H, W)
        returns: (B, 1, H, W)
        """
        tau = 1.0 / (self.sigma ** 2)
        tau_u = 1.0 / (self.unary_sigma ** 2)

        lap = self.laplacian(x)
        return -(tau * lap + tau_u * (x - self.unary_mu))

    # ---------- FFT eigenvalues for torus Laplacian ----------
    def _laplacian_eigs_torus(self) -> torch.Tensor:
        """
        Eigenvalues of 2D 4-neighbour Laplacian with periodic BC:
          λ(k,l)=4 - 2cos(2πk/H) - 2cos(2πl/W)
        returns: (H, W) real tensor
        """
        H, W = self.shape
        k = torch.arange(H, device=self.device, dtype=self.dtype)
        l = torch.arange(W, device=self.device, dtype=self.dtype)

        ang_k = 2.0 * math.pi * k / H
        ang_l = 2.0 * math.pi * l / W

        cos_k = torch.cos(ang_k).view(H, 1)
        cos_l = torch.cos(ang_l).view(1, W)

        return 4.0 - 2.0 * cos_k - 2.0 * cos_l

    @torch.no_grad()
    def sample(self, n_particle=1, eps=1e-8) -> torch.Tensor:
        """
        Exact sampling for periodic BC:
          x ~ N(mu, Q^{-1}), Q = tau * L + tau_u * I
        returns: (n_particle, H, W)
        """
        H, W = self.shape
        tau = 1.0 / (self.sigma ** 2)
        tau_u = 1.0 / (self.unary_sigma ** 2)

        lam_L = self._laplacian_eigs_torus()          # (H,W)
        lam_Q = tau * lam_L + tau_u                  # (H,W)
        lam_Q = torch.clamp(lam_Q, min=eps)

        z = torch.randn((n_particle, H, W), device=self.device, dtype=self.dtype)
        Z = torch.fft.fft2(z)
        X = Z / torch.sqrt(lam_Q)
        x = torch.fft.ifft2(X).real + self.unary_mu
        return x

        
class GaussianABCLikelihood():
    """log(p(evidence|para)) of the Gaussian ABC model with provided mean or mean function with respect to the parameter"""
    def __init__(self, epsilon=EPSILON):
        """
        epsilon: float
            - the ABC error for epsilon
        """
        self.epsilon = epsilon

    def llh(self, data, parameter=None, llh_info_dict=dict(), llh_transform_fn=None, epsilon=None): #standard form
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
        return self.llh_(data=data, parameter=parameter, mean_fn=llh_transform_fn, llh_info_dict=llh_info_dict, epsilon=epsilon)

    def llh_(self, data, parameter=None, mean_fn=None, llh_info_dict=dict(), epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        """see self.llh, where mean_fn is the llh_transform_fn of self.llh"""
        mean = self.compute_mean(parameter=parameter, mean_fn=mean_fn, llh_info_dict=llh_info_dict)
        llh = torch.distributions.normal.Normal(loc=mean, scale=epsilon).log_prob(data).sum(axis=-1)
        return llh, {"mean": mean.detach()}
    
    def compute_mean(self, parameter=None, mean_fn=None, llh_info_dict=dict()):
        """function to compute the mean of the Gaussian ABC likelihood, see self.llh and self.llh_"""
        assert llh_info_dict.get("mean") is not None or mean_fn is not None, "either mean or mean_fn of the form mean_fn(parameter) -> mean should be provided"
        if llh_info_dict.get("mean") is None:
            mean = mean_fn(parameter).float()
        else:
            mean = llh_info_dict["mean"]
        return mean

    def llh_gradient(self, data, parameter, llh_transform_grad_fn, llh_info_dict=dict(), llh_transform_fn=None, epsilon=None): #standard form
        """compute the gradient of the log likelihood function with respect to the parameter
        data: see self.llh
        parameter: see self.llh
        llh_info_dict: see self.llh
        llh_transform_fn: see self.llh
        llh_transform_grad_fn: function torch.tensor -> torch.tensor
            - a function that takes the parameter and output the gradient (Jacobian) of the mean of the Gaussian ABC likelihood function with respect to the parameter
        """
        return self.llh_gradient_(data=data, parameter=parameter, mean_jacobian_fn=llh_transform_grad_fn, mean_fn=llh_transform_fn, llh_info_dict=llh_info_dict, epsilon=epsilon)

    def llh_gradient_(self, data, parameter, mean_jacobian_fn=None, mean_fn=None, llh_info_dict=dict(), epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        """see self.llh_gradient, in which mean_jacobian_fn is the llh_transform_grad_fn"""
        mean = self.compute_mean(parameter=parameter, mean_fn=mean_fn, llh_info_dict=llh_info_dict)
        mean_jacobian = mean_jacobian_fn(parameter=parameter)
        
        gradient = 1. / (epsilon**2) *  torch.mv(torch.t(mean_jacobian), (data - mean))
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
    
class PartialGaussianABCLikelihood(GaussianABCLikelihood):
    def __init__(self, epsilon):
        super().__init__(epsilon)
        
    def llh(self, data, gibbs_indices=None, parameter=None, llh_info_dict=dict(), llh_transform_fn=None): #standard form
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
        return self.llh_(data=data, parameter=parameter, gibbs_indices=gibbs_indices, mean_fn=llh_transform_fn, llh_info_dict=llh_info_dict,)
    
    def llh_(self, data, parameter=None, gibbs_indices=None, mean_fn=None, llh_info_dict=dict()):
        """see self.llh, where mean_fn is the llh_transform_fn of self.llh"""
        mean = self.compute_mean(parameter=parameter, mean_fn=mean_fn, llh_info_dict=llh_info_dict, gibbs_indices=gibbs_indices)
        data = data if gibbs_indices is None else data[gibbs_indices]
        llh = torch.distributions.normal.Normal(loc=mean, scale=self.epsilon).log_prob(data).sum(axis=-1)
        return llh, {"mean": mean.detach()}
    
    def compute_mean(self, parameter=None, mean_fn=None, llh_info_dict=dict(), gibbs_indices=None):
        """function to compute the mean of the Gaussian ABC likelihood, see self.llh and self.llh_"""
        assert llh_info_dict.get("mean") is not None or mean_fn is not None, "either mean or mean_fn of the form mean_fn(parameter) -> mean should be provided"
        if llh_info_dict.get("mean") is None:
            mean = mean_fn(parameter, gibbs_indices=gibbs_indices)
        else:
            mean = llh_info_dict["mean"]
        return mean

class TruncatedGaussianABCLikelihood(GaussianABCLikelihood):
    def __init__(self, epsilon):
        super().__init__(epsilon)

    @staticmethod
    def neg_exp_transform(parameter):
        return - torch.exp(parameter).to(torch.float32)
    
    @staticmethod
    def log_neg_transform(parameter):
        return torch.log(- parameter).to(torch.float32)
    
    def llh_(self, data, parameter=None, mean_fn=None, llh_info_dict=dict()):
        """see self.llh, where mean_fn is the llh_transform_fn of self.llh"""
        original_parameter = self.neg_exp_transform(parameter)
        mean = self.compute_mean(parameter=original_parameter, mean_fn=mean_fn, llh_info_dict=llh_info_dict)
        if FROZEN:
            parameter = parameter[FROZEN_NO]
        llh = torch.distributions.normal.Normal(loc=mean, scale=self.epsilon).log_prob(data).sum(axis=-1) + torch.sum(parameter)
        return llh, {"mean": mean.detach()}
    
    def llh_gradient_(self, data, parameter, mean_jacobian_fn=None, mean_fn=None, llh_info_dict=dict()):
        """see self.llh_gradient, in which mean_jacobian_fn is the llh_transform_grad_fn"""
        original_parameter = self.neg_exp_transform(parameter)
        mean = self.compute_mean(parameter=original_parameter, mean_fn=mean_fn, llh_info_dict=llh_info_dict)
        mean_jacobian = mean_jacobian_fn(parameter=original_parameter)
        
        gradient = 1. / (self.epsilon**2) *  torch.mv(torch.t(mean_jacobian), (data - mean)) * original_parameter + torch.ones_like(parameter)
        return gradient, {"mean_jacobian":mean_jacobian}
    
    def llh_hessian_(self, parameter, mean_jacobian_fn=None, mean_hessian_fn=None, llh_grad_info_dict=dict(), **kwargs):
        """see self.llh_hessian, in which mean_jacobian_fn is the llh_transform_grad_fn, and mean_hessian_fn is the llh_transform_hessian_fn"""
        if mean_hessian_fn is None:
            if llh_grad_info_dict.get("mean_jacobian") is None:
                mean_jacobian = mean_jacobian_fn(parameter=self.neg_exp_transform(parameter))
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

    def llh(self, parameter, llh_info_dict=dict(), epsilon=None):
        """compute the loglikelihood given the abclikelihood and return the loglikelihood with the llh_info_dict"""
        llh, llh_info_dict = self.abclikelihood.llh(data=self.data, parameter=parameter, llh_info_dict=llh_info_dict, llh_transform_fn=self.llh_transform_fn, epsilon=epsilon)
        return llh, llh_info_dict

    def logtarget_density(self, parameter, llh_info_dict=dict(), epsilon=None):
        """compute the log target density (logprior + llh) given the abclikelihood and prior and return the log target density with the llh_info_dict"""
        logprior = self.logprior(parameter=parameter)
        llh, llh_info_dict = self.llh(parameter=parameter, llh_info_dict=llh_info_dict, epsilon=epsilon)
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
    
    def llh_auto_gradient(self, parameter):
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
        llh, llh_info_dict = self.llh(parameter=parameter, llh_info_dict=dict()) # must use the llh_transform_fn to compute the density
        llh.backward()
        gradient = parameter.grad.clone()
        parameter.grad.zero_()
        parameter.requires_grad = False
        llh = llh.detach()
        return llh, gradient, llh_info_dict
    
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
        prior_parameter = pyro.sample("prior_parameter", dist.MultivariateNormal(torch.zeros(parameter_len), torch.eye(parameter_len)*self.prior.sigma**2))
        # print(prior_parameter, parameter_len)#, pyro.sample("test_para", dist.MultivariateNormal(torch.zeros(parameter_len), self.prior.covariance_matrix(parameter_len=parameter_len))))
        # if full_para is not None:
        #     print('Full para')
            # FREEZE = [2,3]
            # prior_parameter[FREEZE] = prior_parameter[FREEZE].detach()
            # prior_parameter_copy = prior_parameter.clone()
            # print(prior_parameter_copy)
            # prior_parameter_copy[FREEZE] = deepcopy(full_para[FREEZE])
            # print(prior_parameter_copy)
            # full_para[FROZEN_NO] = prior_parameter
            # prior_parameter = full_para.clone()
        mean = self.abclikelihood.compute_mean(parameter=prior_parameter, mean_fn=self.llh_transform_fn, llh_info_dict=dict())
        with pyro.plate("data_plate"):
            pyro.sample("obs", dist.MultivariateNormal(mean, self.abclikelihood.covariance_matrix(data_len=len(self.data))), obs=data)
            
class DeterministicSRModelSMC(DeterministicSRModel):
    def __init__(self, prior, abclikelihood, data, old_data, new_data, data_key=None, llh_transform_fn_old=None, llh_transform_fn_new=None, llh_transform_grad_fn_old=None, llh_transform_grad_fn_new=None, new_epsilon=None, *args):
        super(DeterministicSRModelSMC, self).__init__(prior, abclikelihood, data, *args)
        self.data = data
        self.data_key = data_key
        self.old_data = old_data
        self.new_data = new_data
        self.llh_transform_fn_old = llh_transform_fn_old
        self.llh_transform_fn_new = llh_transform_fn_new
        self.llh_transform_grad_fn_old = llh_transform_grad_fn_old
        self.llh_transform_grad_fn_new = llh_transform_grad_fn_new
        self.new_epsilon = new_epsilon
        
    def llh_new(self, parameter, epsilon, old=False):
        if old:
            data = self.old_data
            llh_transform_fn = self.llh_transform_fn_old
        else:
            data = self.old_data if len(self.new_data) == 0 else self.new_data
            llh_transform_fn = self.llh_transform_fn_old if len(self.new_data) == 0 else self.llh_transform_fn_new
        return self.abclikelihood.llh(data=data, parameter=parameter, llh_transform_fn=llh_transform_fn, epsilon=epsilon)
    
    def llh(self, parameter, epsilon=None, llh_info_dict=dict()):
        """compute the loglikelihood given the abclikelihood and return the loglikelihood with the llh_info_dict"""
        if epsilon is None:
            epsilon = self.new_epsilon
        old_epsilon = self.new_epsilon if len(self.new_data) == 0 else self.abclikelihood.epsilon
        old_llh, _ = self.abclikelihood.llh(data=self.old_data, parameter=parameter, llh_transform_fn=self.llh_transform_fn_old, epsilon=old_epsilon)
        new_llh, _ = self.abclikelihood.llh(data=self.new_data, parameter=parameter, llh_transform_fn=self.llh_transform_fn_new, epsilon=epsilon)
        return old_llh + new_llh, llh_info_dict
    
    def logtarget_gradient(self, parameter, new_epsilon=None, llh_info_dict=dict()):
        """compute the gradient of the log target density with respect to the parameter and return the gradient, using the explicit derivation of the gradient"""
        if new_epsilon is None:
            new_epsilon = self.new_epsilon
        old_epsilon = self.new_epsilon if len(self.new_data) == 0 else self.abclikelihood.epsilon 
        logprior_grad = self.prior.logprior_gradient(parameter=parameter)
        llh_grad_old, llh_grad_info = self.abclikelihood.llh_gradient(data=self.old_data, parameter=parameter, llh_info_dict=llh_info_dict, llh_transform_fn=self.llh_transform_fn_old, llh_transform_grad_fn=self.llh_transform_grad_fn_old, epsilon=old_epsilon)
        llh_grad_new, llh_grad_info = self.abclikelihood.llh_gradient(data=self.new_data, parameter=parameter, llh_info_dict=llh_info_dict, llh_transform_fn=self.llh_transform_fn_new, llh_transform_grad_fn=self.llh_transform_grad_fn_new, epsilon=new_epsilon)
        return logprior_grad + llh_grad_new + llh_grad_old, llh_grad_info

    def pyro_model(self, data, parameter_len, data_key=None):
        """the equivalent pyro model, for use in pyro MCMC functions
        data: torch.tensor
            - this input is required as a standard format of a pyro model
        parameter_len: int
            - the dimension of the parameter
        """
        data_key = data_key if data_key is not None else "rewards"
        old_data = torch.tensor(data._buffers[data_key], dtype=torch.float32)
        new_data = torch.tensor(data._new_data_buffers[data_key], dtype=torch.float32)
        data = torch.cat((old_data, new_data), dim=0)
        prior_parameter = pyro.sample("prior_parameter", dist.MultivariateNormal(torch.zeros(parameter_len), torch.eye(parameter_len)*self.prior.sigma**2))
        mean_old = self.abclikelihood.compute_mean(parameter=prior_parameter, mean_fn=self.llh_transform_fn_old)
        mean_new = self.abclikelihood.compute_mean(parameter=prior_parameter, mean_fn=self.llh_transform_fn_new)
        mean = torch.cat((mean_old, mean_new))
        cov = torch.diag(self.abclikelihood.covariance_matrix(data_len=len(data)))
        if self.new_epsilon is not None:
            cov[len(old_data):] = self.new_epsilon
        # print(mean.shape, torch.diag(cov).shape, data.shape)
        pyro.sample("obs", dist.MultivariateNormal(mean, torch.diag(cov)), obs=data)


class StochasticSModel(DeterministicSRModel):
    """the overall model of the ABC likelihood model with prior and likelihood (with stochastic state transition)"""
    def __init__(self, prior, abclikelihood, data, z_transform_fn=None, llh_transform_fn=None, llh_transform_grad_fn=None, llh_transform_hessian_fn=None, *args):
        super(StochasticSModel, self).__init__(prior, abclikelihood, data, llh_transform_fn=llh_transform_fn, llh_transform_grad_fn=llh_transform_grad_fn, llh_transform_hessian_fn=llh_transform_hessian_fn, *args)
        self.z_transform_fn = z_transform_fn
        self.var = None

    def set_var(self, var):
        self.var = var
        
    def set_samples(self, samples):
        '''samples: dict'''
        self.samples = samples
        
    def llh(self, parameter, llh_info_dict=dict(), gibbs_indices=None):
        """compute the loglikelihood given the abclikelihood and return the loglikelihood with the llh_info_dict"""
        if isinstance(parameter, dict):
            samples = parameter
        else:
            samples = self.samples.copy()
            if self.var is not None:
                samples[self.var] = parameter
        llh, llh_info_dict = self.abclikelihood.llh(data=self.data, parameter=samples, gibbs_indices=gibbs_indices, llh_info_dict=llh_info_dict, llh_transform_fn=self.llh_transform_fn)
        return llh, llh_info_dict
    
    def logtarget_density(self, parameter, llh_info_dict=dict()):
        """compute the log target density (logprior + llh) given the abclikelihood and prior and return the log target density with the llh_info_dict"""
        samples = self.samples.copy()
        if self.var is not None:
            samples[self.var] = parameter
        if self.var == 'z':
            logprior = 0 
        else:
            logprior = self.logprior(parameter=parameter)
        llh, llh_info_dict = self.llh(parameter=samples, llh_info_dict=llh_info_dict)
        return logprior + llh, llh_info_dict
    
    def logtarget_gradient(self, parameter, llh_info_dict=dict()):
        """compute the gradient of the log target density with respect to the parameter and return the gradient, using the explicit derivation of the gradient"""
        samples = self.samples.copy()
        if self.var is not None:
            samples[self.var] = parameter
        # if self.var == 'z':
        #     logprior_grad = 0 
        # else:
        logprior_grad = self.prior.logprior_gradient(parameter=parameter)
        llh_grad, llh_grad_info = self.abclikelihood.llh_gradient(data=self.data, parameter=samples, llh_info_dict=llh_info_dict, llh_transform_fn=self.llh_transform_fn, llh_transform_grad_fn=self.llh_transform_grad_fn)
        return logprior_grad + llh_grad, llh_grad_info
    
    def logtarget_auto_gradient(self, parameter):
        parameter = parameter.clone()
        parameter.requires_grad = True
        samples = self.samples.copy()
        if self.var is not None:
            samples[self.var] = parameter
        logtarget_density, llh_info_dict = self.logtarget_density(parameter=samples, llh_info_dict=dict()) # must use the llh_transform_fn to compute the density
        logtarget_density.backward()
        gradient = parameter.grad.clone()
        parameter.grad.zero_()
        parameter.requires_grad = False
        logtarget_density = logtarget_density.detach()
        return logtarget_density, gradient, llh_info_dict