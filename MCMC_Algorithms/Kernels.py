import numpy as np
import scipy.stats as stats
import torch
from copy import deepcopy
import pyro
from tqdm.notebook import tqdm
'''module import'''
from parameter import *
from utils import *

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
    
    def gradient(self, parameter, info_dict=dict(), return_logtarget_density=True, llh=False):
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
                if llh:
                    gradient, llh_grad_info_dict = self.model.llh_gradient(parameter=parameter, llh_info_dict=llh_info_dict)
                else:
                    gradient, llh_grad_info_dict = self.model.logtarget_gradient(parameter=parameter, llh_info_dict=llh_info_dict)

        if return_logtarget_density is True:
            if logtarget_density is None or llh_info_dict is None:
                if llh: 
                    logtarget_density, llh_info_dict = self.model.llh(parameter=parameter, llh_info_dict=llh_info_dict)
                else:
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
        self.traj_len = traj_len 
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
                print('Value Error occured')
            if np.random.uniform(0,1) < accept_prob:
                current_para = proposed_para
                current_para_info_dict = proposed_para_info_dict

            m = i+1
            H_bar = (1 - 1/(m+t0)) * H_bar + (1/(m+t0)) * (target_prob - accept_prob)
            logeps = mu - (m**0.5)/gamma * H_bar
            eps = np.exp(logeps)
            logeps_bar = m**(-kappa) * logeps + (1 - m**(-kappa)) * logeps_bar

            pbar.set_description("Warmup: Most recent alpha {}, with stepsize {}".format(str(np.round(accept_prob.numpy(), 3)), str(np.round(eps, 3))))

        stepsize = np.exp(logeps_bar)
        if set_stepsize is True:   
            self.set_stepsize(stepsize=stepsize)
            self.set_L(num_steps=None, traj_len=self.traj_len, stepsize=stepsize, set_L=True)
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
        return self.move_(current_para=current_para, current_gradient=current_gradient, current_precondition=current_precondition, L=self.L, stepsize=self.stepsize)

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

class HMC_Z(HMC):
    def __init__(self, model, traj_len=2*np.pi, num_steps=None, stepsize=0.5, use_autograd=True, use_autohess=True, *args, **kwargs):
        super(HMC_Z, self).__init__(model=model, traj_len=traj_len, num_steps=num_steps, stepsize=stepsize, use_autograd=use_autograd, use_autohess=use_autohess, *args, **kwargs)
        self.stepsize = stepsize
        self.kwargs = kwargs
        self.original = True         
    
    def move_(self, current_para, current_gradient, L=1, stepsize=0.01, additional_para=None):
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
                gradient, _ = self.gradient(parameter=[q, additional_para], info_dict=dict(), return_logtarget_density=False)
                p = p + stepsize * gradient
        proposed_gradient, proposed_logtarget_density, proposed_para_llh_info_dict, proposed_para_llh_grad_info_dict = self.gradient(parameter=[q, additional_para], info_dict=dict(), return_logtarget_density=True)
        
        p = p + stepsize * proposed_gradient * 0.5
        p = -p
        
        q_info_dict = {"logdensities":proposed_logtarget_density, "gradient":proposed_gradient, "llh_info_dict":proposed_para_llh_info_dict, "llh_grad_info_dict":proposed_para_llh_grad_info_dict}
        return q, p0, p, q_info_dict
    
    def propose_accept(self, current_para, indices=None, stepsize=0.01, L=1, current_para_info_dict=None):
        current_gradient, current_logtarget_density, *_ = self.gradient(parameter=current_para, info_dict=current_para_info_dict, return_logtarget_density=True, llh=True)

        proposed_para, p0, p, q_info_dict = self.move_(current_para=current_para[0], current_gradient=current_gradient, stepsize=self.stepsize, L=self.L, additional_para=current_z)
        proposed_logtarget_density = q_info_dict["logdensities"]
        proposed_para_info_dict = q_info_dict

        H_old = self.hamiltonian(q=current_para, p=p0, q_logtarget_density=current_logtarget_density)
        H_new =  self.hamiltonian(q=proposed_para, p=p, q_logtarget_density=proposed_logtarget_density)
        accept_prob = np.exp(torch_max_0(H_old - H_new))

        return accept_prob, proposed_para, proposed_para_info_dict
    

class Z(Kernel):
    def __init__(self, model, use_autograd=True, use_autohess=True, *args, **kwargs):
        super(Z, self).__init__(model, use_autograd, use_autohess, *args, **kwargs)
        
    def move_(self, indices):
        proposed_blocked_z = self.model.z_transform_fn(indices)
        return proposed_blocked_z
    
    def propose_accept(self, current_para, indices=None, current_para_info_dict=None):
        current_para, current_z = current_para
        current_z_llh_info_dict = current_para_info_dict["llh_info_dict"] if current_para_info_dict.get("llh_info_dict") is not None else dict()
        proposed_blocked_z = self.move_(indices=indices)
        proposed_z = current_z.copy()
        proposed_z[slice(None), indices] = proposed_blocked_z
        if current_para_info_dict.get("logdensities") is not None:
            current_llh = current_para_info_dict["logdensities"]
        else:
            current_llh, _ = self.model.llh(parameter=[current_para, current_z], llh_info_dict=current_z_llh_info_dict)

        proposed_llh, proposed_para_llh_info_dict = self.model.llh(parameter=[current_para, proposed_z], llh_info_dict=dict())
        accept_prob = np.exp(torch_max_0(proposed_llh - current_llh))

        proposed_z_info_dict = {"llh_info_dict": proposed_para_llh_info_dict, "logdensities": proposed_llh}

        return accept_prob, proposed_z, proposed_z_info_dict

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
        