import os, sys

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.append('/scratch/Rabbit/work/ABC_SMC_RL/')
sys.path.append('/Users/guojiaqi/work/ABC_SMC_RL/')

from MCMC_Algorithms.MCMC import *
from MCMC_Algorithms.Kernels import *
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch
import pyro
from pyro.infer.mcmc import NUTS
from scipy.optimize import bisect
from scipy.special import logsumexp
from statsmodels.regression.quantile_regression import QuantReg
from copy import deepcopy

# def smoothen_max(x,y,alpha=0.1):
#     value = torch.cat([x.view(1),y.view(1)])
#     return torch.dot(torch.softmax(value/alpha,0),value)

def smoothen_max(x,y,alpha=0.1):
    value = (x+y+torch.sqrt((x-y)**2 + alpha))/2
    return value

class Modeltest():
    def __init__(self, eps, sigma, z=0.5, r=-0.5, softmax=False, softmax_alpha=0.1):
        self.z = torch.tensor(z)
        self.eps = eps
        self.sigma = sigma
        self.r = torch.tensor(r)
        self.max_fn = torch.max if softmax is False else partial(smoothen_max, alpha=softmax_alpha)
        
    def logtarget_density(self, parameter, **kwargs):
        return self.logtarget_density_eps(parameter=parameter, eps=self.eps, **kwargs)
        
    def logtarget_density_eps(self, parameter, eps, **kwargs):#para_extra_info):
        x = parameter[0]
        y = parameter[1]
        
        
        lp = torch.distributions.Normal(loc=torch.zeros(len(parameter)), scale=self.sigma).log_prob(parameter).sum()
        llh = torch.distributions.Normal(loc=x-self.max_fn(self.z, y), scale=eps).log_prob(self.r).sum()
        return lp+llh, None


    def logtarget_auto_gradient(self, parameter, llh_info_dict=dict()):
        parameter = parameter.clone()
        parameter.requires_grad = True
        logtarget_density, llh_info_dict = self.logtarget_density(parameter=parameter, llh_info_dict=llh_info_dict)
        logtarget_density.backward()
        gradient = parameter.grad.clone()
        parameter.grad.zero_()
        parameter.requires_grad = False
        logtarget_density = logtarget_density.detach()
        return logtarget_density, gradient, llh_info_dict
    
    def logtarget_auto_hessian(self, parameter):
        def fn(parameter):
            logtarget_density, _ = self.logtarget_density(parameter=parameter, llh_info_dict=dict())
            return logtarget_density
        return torch.autograd.functional.hessian(fn, parameter)
    
    def pyro(self, data):
        para = pyro.sample("para", dist.MultivariateNormal(torch.zeros(2), torch.eye(2)*(self.sigma**2)))
        pyro.sample("obs", dist.Normal(para[0]-self.max_fn(self.z,para[1]), self.eps), obs=data)

def run_mcmc(kernel, eps, sigma, num_samples, initial_params, hessian_params, kernel_settings, softmax=False, softmax_alpha=0.1, return_extra=False, silent=False):
    
    model = Modeltest(eps=eps, sigma=sigma, softmax=softmax, softmax_alpha=softmax_alpha)
    if kernel_settings.get("precondition_matrix") is True:
        precondition_matrix = - torch.linalg.inv(model.logtarget_auto_hessian(hessian_params) - 1e-6 * torch.eye(len(hessian_params)))
        kernel_settings["precondition_matrix"] = precondition_matrix
        print("precondition", precondition_matrix)        

    kernel = kernel(model=model, silent=silent, **kernel_settings)
    mcmc = MCMC(num_samples=num_samples, kernel=kernel, initial_params=initial_params, warmup_steps=0, silent=silent)
    
    if return_extra is True:
        sample, proposed = mcmc.run(idx=None, return_proposed=True)
        sample = sample.numpy()
        proposed = proposed.numpy()
        return sample, {"kernel":kernel, "proposed":proposed, "mcmc":mcmc}

    else:
        sample = mcmc.run(idx=None).numpy()
        return sample

def run_mcmc_pyro(kernel, eps, sigma, num_samples, initial_params, kernel_settings, softmax=False, softmax_alpha=0.1, warmup_steps=100, return_extra=True, silent=False):
    
    model = Modeltest(eps=eps, sigma=sigma, softmax=softmax, softmax_alpha=softmax_alpha)

    kernel = kernel(model=model.pyro, **kernel_settings)
    mcmc = pyro.infer.mcmc.MCMC(kernel=kernel, initial_params={"para":initial_params}, warmup_steps=warmup_steps, num_samples=num_samples, disable_progbar=silent)
    mcmc.run(model.r)
    sample = mcmc.get_samples()["para"].numpy()
    if return_extra:
        return sample, {"kernel":kernel, "mcmc":mcmc}
    else:
        return sample

def ESS(weights):
    ess = 1 / np.sum(weights ** 2)
    if np.isnan(ess):
        return 1
    else:
        return ess

def particles_stat(particles):
    return particles + particles ** 2

def max_zero(array):
    array[np.isnan(array)] = - 1e9
    return np.minimum(0,array)

def test_mcmc_stop(particles0, particles1, prev_corr_array):
    dim = len(particles0[0])
    particles0_stat = particles_stat(particles0)
    particles1_stat = particles_stat(particles1)
    corr_array = np.array([np.corrcoef(particles0_stat[:,i], particles1_stat[:,i])[1,0] for i in range(dim)])
    corr_array = corr_array * prev_corr_array
    decision = True if np.mean(corr_array > 0.1) <= 0.1 else False
    return decision, corr_array

def norm(particles, matrix):
    return np.sqrt(np.sum((particles @ matrix.T) * particles, axis=2))

def estimate_diag_precondition(particles, weights):
    dim = len(particles[0])
    mat = np.diag(np.array([np.dot(weights,particles[:,i].numpy()**2) - np.dot(weights, particles[:,i].numpy())**2 for i in range(dim)]))
    return torch.tensor(mat).float()

def compute_ESJD(particles_ls, proposed_ls, H_change_ls, precondition_matrix, L):
    particles_ls = np.stack(particles_ls)
    proposed_ls = np.stack(proposed_ls)
    H_change_ls = np.stack(H_change_ls)
    mat = np.diag(1/np.diag(np.array(precondition_matrix)))
    
    ESJD =  np.mean(norm(particles_ls-proposed_ls, matrix=mat) / L * np.minimum(np.exp(H_change_ls),1),axis=0)

    ESJD[np.isnan(ESJD)] = 0
    return ESJD

def update_weight_init(model, log_weights, particles, eps):

    n_particles = len(particles)
    log_weights_prime = log_weights + np.array([model.logtarget_density_eps(particles[i], eps=eps)[0].numpy() for i in range(n_particles)])
    log_weights = log_weights_prime - logsumexp(log_weights_prime)
    
    return log_weights

def update_weight_iter(model, log_weights, particles, eps, new_eps):

    n_particles = len(particles)
    log_weights_prime = log_weights + np.array([model.logtarget_density_eps(particles[i], eps=new_eps)[0].numpy() - model.logtarget_density_eps(particles[i], eps=eps)[0].numpy() for i in range(n_particles)])
    log_weights = log_weights_prime - logsumexp(log_weights_prime)
    
    return log_weights

def SMC_NUTS_move(kernel, num_moves, particles, sigma, eps, silent):

    n_particles = len(particles)
    accept_prob = np.zeros(n_particles)
    step_size = np.zeros(n_particles)
    particles = deepcopy(particles)

    pbar = tqdm(range(n_particles))
    for i in pbar:
        kernel_settings = dict(adapt_step_size=True, adapt_mass_matrix=False, max_tree_depth=5)
        samples, mcmc_extra = run_mcmc_pyro(kernel=kernel, warmup_steps=50, eps=eps, sigma=sigma, num_samples=num_moves, initial_params=particles[i], kernel_settings=kernel_settings, return_extra=True, silent=silent)
        particles[i] = torch.tensor(samples[-1])

        kernel_out = mcmc_extra["kernel"] 
        mcmc_cls = mcmc_extra["mcmc"]

        accept_prob[i] = mcmc_cls.diagnostics()['acceptance rate']["chain 0"]
        step_size[i] = kernel_out.step_size

    print("step size", step_size.mean(), "accept_prob", accept_prob.mean(), "num_moves", num_moves)

    return particles, accept_prob, step_size

def SMC_MCMC_move(kernel, num_moves, particles, sigma, eps, step_size, L, precondition_matrix, adaptive_move, silent):

    n_particles = len(particles)
    particles = deepcopy(particles)

    corr_stat = np.ones(2)
    H_change_tmp_ls = []
    particles_tmp_ls = []
    proposed_tmp_ls = []

    counter = 0

    accept_prob = np.zeros(n_particles)

    for j in range(num_moves):

        counter += 1

        proposed_particles = torch.zeros(n_particles,2)
        H_change = np.zeros(n_particles)

        particles_tmp_ls.append(particles.clone().numpy())

        for i in range(n_particles):
            kernel_settings = dict(stepsize=step_size[i], num_steps=L[i], use_autograd=True, mass=1, precondition_matrix=precondition_matrix, traj_len=None)
            samples, mcmc_extra = run_mcmc(kernel=kernel, eps=eps, sigma=sigma, num_samples=1, initial_params=particles[i], hessian_params=particles[i], kernel_settings=kernel_settings, return_extra=True, silent=silent)

            kernel_out = mcmc_extra["kernel"] 
            proposed = mcmc_extra["proposed"]
            mcmc_cls = mcmc_extra["mcmc"]

            particles[i] = torch.tensor(samples[-1])
            proposed_particles[i] = torch.tensor(proposed[-1])
            H_change[i] = kernel_out.H_change[-1]

            accept_prob[i] += mcmc_cls.get_accept_prob().numpy()[-1]

        proposed_tmp_ls.append(proposed_particles.clone().numpy())
        H_change_tmp_ls.append(H_change)

        test_dec, corr_stat = test_mcmc_stop(particles_tmp_ls[-1], particles, corr_stat)
        if test_dec is True and adaptive_move is True:
            break

    accept_prob = accept_prob / counter
    ESJD = compute_ESJD(particles_ls=particles_tmp_ls, proposed_ls=proposed_tmp_ls, H_change_ls=H_change_tmp_ls, precondition_matrix=precondition_matrix, L=L)

    return particles, H_change_tmp_ls, proposed_tmp_ls, particles_tmp_ls, counter, accept_prob, ESJD

def resample_L_stepsize(ESJD, L, step_size):
    n_particles = len(L)
    if np.all(ESJD == 0):
        Lambda = np.ones(len(Lambda)) / len(Lambda)
    Lambda =  ESJD / np.sum(ESJD)
    h_indices = np.random.choice(range(n_particles),size=n_particles,replace=True, p=Lambda)
    L = L[h_indices]
    step_size = step_size[h_indices]
    return L, step_size


def FT_adaptation(L, step_size):
    n_particles = len(step_size)

    L = L + np.random.randint(-1,2, size=n_particles)
    L = np.clip(L, 1, 200)
    step_size = np.random.normal(step_size, 0.015, size=n_particles)
    step_size = np.clip(step_size, 1e-7, 10)
    return L, step_size

def quantile_regression(target, H_change, step_size):

    abslogtarget = np.abs(np.log(target))

    if np.isinf(H_change).any() or np.isnan(H_change).any():
        indices = np.isfinite(H_change)
        H_change = H_change[indices]
        step_size = step_size[indices]

    step_size_max_simple = np.max(step_size[np.abs(H_change)<abslogtarget])

    H_change = np.clip(np.abs(H_change),0,1e6)
    reg = QuantReg(H_change,step_size**2)
    quant_param = reg.fit(0.5).params
    step_size_max_quant = np.sqrt(abslogtarget / quant_param)

    step_size_max = np.maximum(step_size_max_quant, step_size_max_simple)

    return step_size_max


def Pretune_adaptation(H_change, L_origin, L_resampled, step_size_origin, L_max):
    
    #L
    L80 = np.percentile(L_origin, 80)
    L20 = np.percentile(L_origin, 20)

    if (L_resampled > L80).mean() > 0.5:
        L_max = L_max + 5
    elif (L_resampled < L20).mean() > 0.5:
        if L_max > 5:
            L_max = L_max - 5

    #stepsize
    step_size_max = quantile_regression(target=0.9, H_change=H_change, step_size=step_size_origin)

    return L_max, step_size_max


def SMC(kernel, eps0, eps_f, n_particles, sigma, c=0.9, num_moves=100, silent=True, adapt_alg="ft", return_full_hist=False):

    weights_ls = []
    particles_ls = []
    ess_ls = []
    eps_ls = []
    
    #extra stats
    step_size_ls = []
    L_ls = []
    num_moves_ls = []
    precondition_ls = []
    accept_prob_ls = []
    ESJD_ls = []
    
    model = Modeltest(eps=eps0, sigma=sigma)

    #initialise weights and particles
    weights = np.ones(n_particles) / n_particles
    log_weights = np.log(weights)
    particles = torch.distributions.normal.Normal(loc=torch.zeros(2), scale=sigma).sample((n_particles,))
    
    weights_ls.append(weights)
    particles_ls.append(particles.clone().numpy())
    ess_ls.append(n_particles)

    if return_full_hist and (adapt_alg in ["ft", "pretune"]):
        particles_full_ls = []
        proposed_full_ls = []
        H_change_full_ls = []
    

#     test_fn = lambda eps: ESS(np.exp(update_weight_init(model=model, log_weights=log_weights, particles=particles, eps=eps))) - c*n_particles
#     print("here",[test_fn(i) for i in [0.1,0.4,0.7,1,3,5,7,10]])

    #Initial epsilon
    try:
        eps, res = bisect(lambda eps: ESS(np.exp(update_weight_init(model=model, log_weights=log_weights, particles=particles, eps=eps))) - c*n_particles, a=1e-5, b=50, maxiter=100, disp=False, full_output=True)
    except ValueError:
        eps = eps0
        print("fail to bisect")
    eps_ls.append(eps)

    #Reweight
    log_weights = update_weight_init(model=model, log_weights=log_weights, particles=particles, eps=eps)
    weights = np.exp(log_weights)
    print("init epsilon", eps)

    ess = ESS(weights)
    print("ess",ess)

    #Resample
    if ess < 0.5 * n_particles:
        particles = particles[torch.multinomial(torch.tensor(weights), n_particles, replacement=True)]
        weights = np.ones(n_particles) / n_particles
        log_weights = np.log(weights)
        ess = ESS(weights)

    #Tune
    precondition_matrix = torch.tensor(np.eye(2)).float()

    if adapt_alg == "ft":
        step_size = np.random.random(n_particles)*0.1
        L = np.random.randint(1,100, size=n_particles)

    elif adapt_alg == "pretune":
        step_size_pretune = np.random.random(n_particles)*0.1
        L_pretune = np.random.randint(1,100, size=n_particles)
        L_max_pretune = 100
        _, H_change_tmp_ls_pretune, _, _, _, _, ESJD_pretune = SMC_MCMC_move(kernel=kernel, num_moves=1, particles=particles, sigma=sigma, eps=eps, step_size=step_size_pretune, L=L_pretune, precondition_matrix=precondition_matrix, adaptive_move=False, silent=silent)
        L, step_size = resample_L_stepsize(ESJD=ESJD_pretune, L=L_pretune, step_size=step_size_pretune)
        L_max_pretune, step_size_max_pretune = Pretune_adaptation(H_change=H_change_tmp_ls_pretune[0], L_origin=L_pretune, L_resampled=L, step_size_origin=step_size_pretune, L_max=L_max_pretune)

    if adapt_alg in ["ft", "pretune"]:
        #MCMC move
        particles, H_change_tmp_ls, proposed_tmp_ls, particles_tmp_ls, actual_num_moves, accept_prob, ESJD = SMC_MCMC_move(kernel=kernel, num_moves=num_moves, particles=particles, sigma=sigma, eps=eps, step_size=step_size, L=L, precondition_matrix=precondition_matrix, adaptive_move=True, silent=silent)

        #Update stats
        step_size_ls.append(step_size)
        L_ls.append(L)
        num_moves_ls.append(actual_num_moves)
        precondition_ls.append(np.eye(2))
        accept_prob_ls.append(accept_prob)
        ESJD_ls.append(ESJD)

        if return_full_hist:
            particles_full_ls.append(np.array(particles_tmp_ls))
            proposed_full_ls.append(np.array(proposed_tmp_ls))
            H_change_full_ls.append(np.array(H_change_tmp_ls))

    elif adapt_alg == "nuts":
        particles, accept_prob, step_size = SMC_NUTS_move(kernel=kernel, num_moves=num_moves, particles=particles, sigma=sigma, eps=eps, silent=silent)

        accept_prob_ls.append(accept_prob)
        step_size_ls.append(step_size)

    weights_ls.append(weights)
    particles_ls.append(particles.clone().numpy())
    ess_ls.append(ess)



    while eps > eps_f:
        new_eps = eps

#         test_fn = lambda new_eps: ESS(np.exp(update_weight_iter(model=model, log_weights=log_weights, particles=particles, eps=eps, new_eps=new_eps))) - c*ess
#         print("herehere",[test_fn(i) for i in [1e-5, 0.1,0.4,0.7,1,3,5,7,10,eps]])
        
        #Find new epsilon
        try:
            new_eps, res = bisect(lambda new_eps: ESS(np.exp(update_weight_iter(model=model, log_weights=log_weights, particles=particles, eps=eps, new_eps=new_eps))) - c*ess, a=1e-5, b=eps, maxiter=100, disp=False, full_output=True)
            #print(res)
        except ValueError:
            new_eps = new_eps / 2
            print("failed to bisect")
        print("new eps", new_eps)
        eps_ls.append(new_eps)
        
        #Reweight
        log_weights = update_weight_iter(model=model, log_weights=log_weights, particles=particles, eps=eps, new_eps=new_eps)
        weights = np.exp(log_weights)
        
        ess = ESS(weights)
        print("ess,", ess)

        #Resample
        if ess < 0.5 * n_particles:
            particles = particles[torch.multinomial(torch.tensor(weights), n_particles, replacement=True)]
            weights = np.ones(n_particles) / n_particles
            log_weights = np.log(weights)
            ess = ESS(weights)
            print("updated ess,", ess)


        #Tune
        precondition_matrix = estimate_diag_precondition(particles=particles, weights=weights)

        if adapt_alg == "ft":
            L_ft, step_size_ft = resample_L_stepsize(ESJD=ESJD, L=L, step_size=step_size)
            L, step_size = FT_adaptation(L=L_ft, step_size=step_size_ft)
        elif adapt_alg == "pretune":
            step_size_pretune = np.random.random(n_particles)*step_size_max_pretune
            L_pretune = np.random.randint(1,L_max_pretune+1, size=n_particles)
            _, H_change_tmp_ls_pretune, _, _, _, _, ESJD_pretune = SMC_MCMC_move(kernel=kernel, num_moves=1, particles=particles, sigma=sigma, eps=eps, step_size=step_size_pretune, L=L_pretune, precondition_matrix=precondition_matrix, adaptive_move=False, silent=silent)
            L, step_size = resample_L_stepsize(ESJD=ESJD_pretune, L=L_pretune, step_size=step_size_pretune)
            L_max_pretune, step_size_max_pretune = Pretune_adaptation(H_change=H_change_tmp_ls_pretune[0], L_origin=L_pretune, L_resampled=L, step_size_origin=step_size_pretune, L_max=L_max_pretune)

        if adapt_alg in ["ft", "pretune"]:

            #MCMC move
            particles, H_change_tmp_ls, proposed_tmp_ls, particles_tmp_ls, actual_num_moves, accept_prob, ESJD = SMC_MCMC_move(kernel=kernel, num_moves=num_moves, particles=particles, sigma=sigma, eps=eps, step_size=step_size, L=L, precondition_matrix=precondition_matrix, adaptive_move=True, silent=silent)

            #stats
            step_size_ls.append(step_size)
            L_ls.append(L)
            num_moves_ls.append(actual_num_moves)
            precondition_ls.append(precondition_matrix.numpy())
            accept_prob_ls.append(accept_prob)
            ESJD_ls.append(ESJD)

            if return_full_hist:
                particles_full_ls.append(np.array(particles_tmp_ls))
                proposed_full_ls.append(np.array(proposed_tmp_ls))
                H_change_full_ls.append(np.array(H_change_tmp_ls))
        
        elif adapt_alg == "nuts":
            particles, accept_prob, step_size = SMC_NUTS_move(kernel=kernel, num_moves=num_moves, particles=particles, sigma=sigma, eps=eps, silent=silent)

            accept_prob_ls.append(accept_prob)
            step_size_ls.append(step_size)


        weights_ls.append(weights)
        particles_ls.append(particles.clone().numpy())
        ess_ls.append(ess)

        eps = new_eps
    
    if adapt_alg in ["ft", "pretune"]:
        extra_stats = {"step_size": step_size_ls, "L": L_ls, "num_moves": num_moves_ls, "precondition": precondition_ls, "accept_prob_ls": accept_prob_ls, "ESJD_ls": ESJD_ls}
        if return_full_hist:
            extra_stats["particles_full_ls"] = particles_full_ls
            extra_stats["proposed_full_ls"] = proposed_full_ls
            extra_stats["H_change_full_ls"] = H_change_full_ls

    elif adapt_alg == "nuts":
        extra_stats = {"step_size": step_size_ls, "accept_prob_ls": accept_prob_ls}

    return particles.numpy(), weights, weights_ls, particles_ls, ess_ls, eps_ls, extra_stats