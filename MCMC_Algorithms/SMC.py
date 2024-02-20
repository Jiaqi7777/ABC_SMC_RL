import numpy as np
from scipy.optimize import bisect
from scipy.optimize import root
import sys
import os
from statsmodels.regression.quantile_regression import QuantReg
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.append('/scratch/Rabbit/work/ABC_SMC_RL/')
sys.path.append('/Users/guojiaqi/work/ABC_SMC_RL/')
print('system path', sys.path)
from MCMC_Algorithms.MCMC import *
from Environment.MountainCar import *
from model import *
from parameter import *


class SMC:
    """the class to run SMC"""
    def __init__(self, model, Model=None, min_ess=0.5, num_samples=None, initial_params=None, params_dim=None, adapt_alg=ADAPT_ALG):
        assert initial_params is not None or params_dim is not None, "Should either specify initial_params or params_dim"
        if initial_params is not None:
            self.initial_params = initial_params
        else:
            self.initial_params = model.get_learnable_parameter()
        self.n_particle = len(initial_params)
        self.params_dim = np.prod(self.initial_params.shape[1:])
        self.min_ess = min_ess 
        self.model = model
        self._weights = torch.log(model._weights)
        self.num_samples = num_samples
        self.SMC_Model = Model
        self.reset_stat()
        self.adapt_alg = adapt_alg
        # self.kernel = 'HMC' if adapt_alg == 'pretune' else 'NUTS'

    def set_weights(self, weights):
        self._weights = weights
        self.model.set_weights(weights)

    def update(self, alpha, smc_samples, epsilon=None, error_lag=2, error_perc=3e-4, new_data_flag=False, episode='', repeat=''):
        if self.episode != episode:
            if episode != 0:
                self.epsilon_history.append(self.epsilon_epi)
            self.epsilon_epi = []
            self.episode = episode
        self.repeat = repeat
        L_max_pretune, step_size_max_pretune = 100, 0.1
        if epsilon is None:
            new_data_flag = True
            self.loop = 1
            epsilon = self.SMC_Model.abclikelihood.epsilon
            print('epsilon_0 ==========')
            epsilon_0, a, b = self.find_epsilon_0(alpha=alpha, smc_samples=smc_samples, generate_weights_fn=generate_weights_0, epsilon_0=epsilon, a=epsilon, b=epsilon*10, lower_side=False)
            print(epsilon_0)
            self.SMC_Model.new_epsilon = epsilon_0
            self.epsilon_epi.append(epsilon_0)
            # epsilon_0 = epsilon * 2
            weights = generate_weights_0(epsilon=epsilon_0, weights=self._weights, Model=Model, smc_samples=smc_samples, epsilon_0=epsilon_0)
            # print('weights0', weights)
            if torch.any(torch.isnan(weights)):
                print(epsilon_0, self._weights, smc_samples)
                raise ValueError('0')
            # self.update_history(smc_samples=smc_samples, weights=weights)
            if self.ESS(self._weights) < self.min_ess * self.n_particle:
                # print('Resampled')
                smc_samples, weights = self.resample()
            self.set_weights(weights)
            # if self.adapt_alg == 'pretune':
            #     precondition_matrix, L, L_max_pretune, step_size, step_size_max_pretune = self.pretune(smc_samples)
            if self.adapt_alg == 'pretune':
                precondition_matrix = estimate_diag_precondition(particles=smc_samples, weights=weights)
                precondition_matrix, L, L_max_pretune, step_size, step_size_max_pretune = self.pretune(smc_samples, step_size_max_pretune=step_size_max_pretune, L_max_pretune=L_max_pretune)
                # print('pretune results', L, step_size)
                corr_stat = np.ones(self.params_dim)
                print('MCMC...')
                for m in range(training_steps_with_burnin):
                    prev_smc_samples = smc_samples.clone()
                    for j in range(self.n_particle):
                        posterior_samples, accept_probs, mcmc, kernel, logdensities, proposed_logdensities, = MCMC_update(Model=self.SMC_Model, posterior_samples=[smc_samples[j]], env=env, training_steps_with_burnin=1, training_steps=1, stepsize=step_size[j], num_steps=L[j], precondition_matrix=precondition_matrix, USE_AUTOGRAD=USE_AUTOGRAD, WARMUP_RATIO=WARMUP_RATIO, MCMC_SHOW_DISABLE=MCMC_SHOW_DISABLE, ADAPT_STEP_SIZE=ADAPT_STEP_SIZE, ADAPT_MASS_MATRIX=ADAPT_MASS_MATRIX, kernel='HMC')
                        smc_samples[j] = posterior_samples[-1]
                    test_dec, corr_stat = test_mcmc_stop(prev_smc_samples, smc_samples, corr_stat)
                    if test_dec:
                        print('MCMC stopped at', m, 'moves')
                        break
            self.update_history(smc_samples=smc_samples, weights=weights)
        else:
            epsilon_0 = self.SMC_Model.abclikelihood.epsilon
        pre_epsilon_0 = epsilon_0
        bellman_err = [self.bellman_error(smc_samples)]
        epsilon_l = [epsilon_0]
        # while epsilon_0 > epsilon:
        counter = 0
        max_iter = 100
        if not new_data_flag:
            self.epsilon_epi.append(epsilon_0)
            self.loop = 3
        else:
            self.loop = 2
        bellman_err_improve = []
        while counter < max_iter:
            print('Tuning down Epsilon with new data flag =', new_data_flag)
            # epsilon_0 = max(epsilon, self.find_epsilon_0(alpha=alpha, smc_samples=smc_samples, generate_weights_fn=generate_weights, epsilon_0=epsilon_0, a=epsilon_0*0.002, b=pre_epsilon_0))
            epsilon_0, a, b = self.find_epsilon_0(alpha=alpha, smc_samples=smc_samples, generate_weights_fn=generate_weights, epsilon_0=epsilon_0, a=epsilon_0*0.2, b=pre_epsilon_0)
            epsilon_0 = max(epsilon_0, epsilon) if new_data_flag else epsilon_0
            # epsilon_0 *= 0.9
            # if not new_data_flag:
            self.epsilon_epi.append(epsilon_0)
            print('epsilon_1 ==========', '\n', epsilon_0)
            weights = generate_weights(epsilon=epsilon_0, weights=self._weights, Model=Model, smc_samples=smc_samples, epsilon_0=pre_epsilon_0)
            # print('weights1', weights)
            if torch.any(torch.isnan(weights)):
                print(epsilon_0, self._weights, smc_samples, pre_epsilon_0)
                raise ValueError(epsilon_0)
            self.set_weights(weights)
            # self.update_history(smc_samples = smc_samples, weights=weights)
            self.SMC_Model.new_epsilon = epsilon_0
            # self.SMC_Model.epsilon = epsilon_0
            if self.ESS(self._weights) < self.min_ess * self.n_particle:
                # print('Resampled')
                smc_samples, weights = self.resample()
                # self.update_history(smc_samples=smc_samples, weights=weights)
            if self.adapt_alg == 'pretune':
                precondition_matrix = estimate_diag_precondition(particles=smc_samples, weights=weights)
                precondition_matrix, L, L_max_pretune, step_size, step_size_max_pretune = self.pretune(smc_samples, step_size_max_pretune=step_size_max_pretune, L_max_pretune=L_max_pretune)
                # print('pretune results', L, step_size)
                corr_stat = np.ones(self.params_dim)
                print('MCMC...')
                for m in range(training_steps_with_burnin):
                    prev_smc_samples = smc_samples.clone()
                    for j in range(self.n_particle):
                        posterior_samples, accept_probs, mcmc, kernel, logdensities, proposed_logdensities, = MCMC_update(Model=self.SMC_Model, posterior_samples=[smc_samples[j]], env=env, training_steps_with_burnin=1, training_steps=1, stepsize=step_size[j], num_steps=L[j], precondition_matrix=precondition_matrix, USE_AUTOGRAD=USE_AUTOGRAD, WARMUP_RATIO=WARMUP_RATIO, MCMC_SHOW_DISABLE=MCMC_SHOW_DISABLE, ADAPT_STEP_SIZE=ADAPT_STEP_SIZE, ADAPT_MASS_MATRIX=ADAPT_MASS_MATRIX, kernel='HMC')
                        smc_samples[j] = posterior_samples[-1]
                    test_dec, corr_stat = test_mcmc_stop(prev_smc_samples, smc_samples, corr_stat)
                    if test_dec:
                        print('MCMC stopped at', m, 'moves')
                        break
            if self.adapt_alg == 'NUTS':
                for j in range(self.n_particle):
                    posterior_samples, accept_probs, mcmc, kernel, logdensities, proposed_logdensities, = MCMC_update(Model=self.SMC_Model, posterior_samples=[smc_samples[j]], env=env, training_steps_with_burnin=training_steps_with_burnin, training_steps=training_steps, USE_AUTOGRAD=USE_AUTOGRAD, WARMUP_RATIO=WARMUP_RATIO, MCMC_SHOW_DISABLE=MCMC_SHOW_DISABLE, ADAPT_STEP_SIZE=ADAPT_STEP_SIZE, ADAPT_MASS_MATRIX=ADAPT_MASS_MATRIX)
                    # print(posterior_samples.shape)
                    smc_samples[j] = posterior_samples[-1]
            self.update_history(smc_samples=smc_samples, weights=weights)
            pre_epsilon_0 = epsilon_0
            if self.loop == 3:
                e1 = min(bellman_err)
                current_bellman_error = self.bellman_error(smc_samples)
                bellman_err.append(current_bellman_error)
                bellman_err_improve.append(((e1 - current_bellman_error) / e1) > error_perc)
                print('bellman error', bellman_err, -(current_bellman_error - e1) / e1)
            # bellman_err_improve = np.array([(e1 - e2) / e1 if e1 != 0 else 0 for e1, e2 in zip(bellman_err[:-1], bellman_err[1:])])
            epsilon_l.append(epsilon_0)
            if new_data_flag:
                print('New data', epsilon_0, epsilon)
                if epsilon_0 <= epsilon:
                    break
            elif len(bellman_err_improve) >= error_lag and np.sum(bellman_err_improve[-error_lag:]) == 0:
                epsilon_0 = epsilon_l[- error_lag - 1]
                print('stopped at epsilon=', epsilon_0)
                print('epsilon_l', epsilon_l[:- error_lag ])
                self.remove_history(error_lag)
                break
            counter += 1
        if not new_data_flag:
            self.bellman_err_l.append(bellman_err)
        return epsilon_0, smc_samples
        
    @staticmethod
    def ESS(lgweights):
        max_weight = torch.max(lgweights)
        return (torch.exp(lgweights - max_weight)).sum() ** 2 / (torch.exp(2 * (lgweights - max_weight))).sum()
        return 1 / torch.sum(torch.exp(2 * lgweights))
        # return 1 / ( 1 + torch.var(self._weights) )

    def resample(self):
        idx = random.choices(range(self.n_particle), torch.exp(self._weights), k=self.n_particle)
        paras = self.model.get_learnable_parameter()[idx]
        
        weights = torch.log(torch.ones(self.n_particle) / self.n_particle)
        self.set_weights(weights)
        return paras, weights
        
    def reset_stat(self):
        self.episode = -1
        self.samples = torch.tensor([])
        self._weights_history = torch.tensor([])
        self.ess_history = torch.tensor([])
        self.epsilon_history = []
        self.bellman_err_l = []
        
    def update_history(self, smc_samples=None, weights=None):
        if smc_samples is not None:
            self.model.set_learnable_parameter(torch.tensor(smc_samples))
            self.samples = torch.cat((self.samples, self.model.get_parameter().unsqueeze(0)))
        if weights is not None:
            self._weights_history = torch.cat((self._weights_history, torch.exp(weights).unsqueeze(0)))
            self.ess_history = torch.cat((self.ess_history, self.ESS(weights).unsqueeze(0)))
        # print(self.samples.shape)

    def remove_history(self, index):
        self.samples  = self.samples[: - (index - 1)]
        self._weights_history = self._weights_history[:- (index - 1)]
        self.ess_history  = self.ess_history[: - (index - 1)]
        self.epsilon_epi = self.epsilon_epi[: - index]
        self.set_weights(self._weights_history[- 1])
        self.model.set_parameter(self.samples[- 1])

    def plot_ess_fn(self, epsilon_0, smc_samples, generate_weights_fn, a, b, new_epsilon):
        # print(smc_samples[0, 0], self._weights, epsilon_0)
        ess_partial= partial(ess_, Model=self.SMC_Model, weights=self._weights, smc_samples=smc_samples, generate_weights_fn=generate_weights_fn, epsilon_0=epsilon_0, ess_fn=self.ESS)
        e_l = np.linspace(a, b*1.5, 100)
        ess_l = [ess_partial(e) for e in e_l]
        plot_ess(e_l, ess_l, epsilon_0, self.ESS(self._weights), alpha, new_epsilon, save=save, loop_num=self.loop, episode=self.episode, repeat=self.repeat, figure_path=dir)

    def find_epsilon_0(self, alpha, smc_samples, generate_weights_fn, epsilon_0, a, b, lower_side=True, method='bisect', tol=1e-7, max_epsilon=50):
        # print(a, b)
        b = min(b, max_epsilon)
        if b > 1.5 * max_epsilon or (lower_side and a < 1e-7):
            print('illegal range', a, max_epsilon)
            # raise ValueError(a, b, alpha, self.SMC_Model, self._weights, smc_samples, generate_weights_fn, epsilon_0)
            return epsilon_0, a, b
        # Use optimization to find epsilon_0 that satisfies the ESS condition
        if method == 'bisect':
            ess_simple = partial(ESS_Matching, alpha=alpha, Model=self.SMC_Model, weights=self._weights, smc_samples=smc_samples, generate_weights_fn=generate_weights_fn, epsilon_0=epsilon_0)
            ess_partial= partial(ess_, Model=self.SMC_Model, weights=self._weights, smc_samples=smc_samples, generate_weights_fn=generate_weights_fn, epsilon_0=epsilon_0, ess_fn=self.ESS)
            # e_l = np.linspace(a*0.5, b*1.5, 100)
            # ess_l = [ess_partial(e) for e in e_l]
            # plot_ess(e_l, ess_l, epsilon_0, self.ESS(self._weights), alpha)
            try:
                result = bisect(ess_simple, a=a, b=b, xtol=tol, maxiter=2000)
                if result > max_epsilon:
                    self.plot_ess_fn(epsilon_0, smc_samples, generate_weights_fn, a, b, new_epsilon=epsilon_0)
                    print('original epsilon')
                    return max_epsilon, a, b
            except ValueError as ve:
                print(ve)
                if lower_side and abs(ess_partial(b) - max(1, alpha*self.ESS(model._weights))) <= tol:
                    print('b side')
                    self.plot_ess_fn(epsilon_0, smc_samples, generate_weights_fn, a, b, new_epsilon=b)
                    return b, a, b
                if abs(ess_partial(a) - max(1, alpha*self.ESS(model._weights))) <= tol:
                    print('a side')
                    self.plot_ess_fn(epsilon_0, smc_samples, generate_weights_fn, a, b, new_epsilon=a)
                    return a, a, b
                # raise ValueError(a, b, alpha, self.SMC_Model, self._weights, smc_samples, generate_weights_fn, epsilon_0)
                b = b if lower_side else b * 2
                a = a * 0.8 if lower_side else a
                print('Bisect with new alpha', alpha, a, b)
                new_epsilon, a, b = self.find_epsilon_0(alpha, smc_samples, generate_weights_fn, epsilon_0, a, b, lower_side=lower_side, method=method)
                # self.plot_ess_fn(epsilon_0, smc_samples, generate_weights_fn, a, b, new_epsilon=new_epsilon)
                return new_epsilon, a, b
            # result = bisect(ess_simple, a=a, b=b, xtol=tol, maxiter=2000)
            self.plot_ess_fn(epsilon_0, smc_samples, generate_weights_fn, a, b, new_epsilon=result)
            # if epsilon_0 < 0.03 and (not lower_side):
            #     dsd
            return result, a, b
        elif method == 'root':
            try:
                result = root(ESS_Matching, x0=b, tol=1e-3, args=(alpha, smc.SMC_Model, smc._weights, smc_samples, generate_weights_fn, epsilon_0), method='lm')
                if not result.success:
                    b = b if lower_side else b * 2
                    print('Bisect with new alpha', min(alpha*1.2, 1), a*0.8, b)
                    return self.find_epsilon_0(min(alpha*1.2, 1), smc_samples, generate_weights_fn, epsilon_0, a*0.8, b, lower_side=lower_side, method=method)
                if max(result.x) > 1e3:
                    return epsilon_0
            except:
                # raise ValueError(a, b, alpha, self.SMC_Model, self._weights, smc_samples, generate_weights_fn, epsilon_0)
                b = b if lower_side else b * 2
                print('Bisect with new alpha', min(alpha*1.2, 1), a*0.8, b)
                return self.find_epsilon_0(min(alpha*1.2, 1), smc_samples, generate_weights_fn, epsilon_0, a*0.8, b, lower_side=lower_side, method=method)
            return max(result.x) 
        else:
            raise NotImplementedError('No method implemented for method', method)
    
    def pretune(self, smc_samples, step_size_max_pretune=0.1, L_max_pretune=99):
        precondition_matrix = torch.tensor(np.eye(self.params_dim)).float()
        H_change = torch.zeros(self.n_particle)
        step_size_pretune = torch.rand(self.n_particle) * step_size_max_pretune
        L_pretune = torch.randint(1, L_max_pretune + 1, size=(self.n_particle, ))
        # pretune_samples = []
        proposed_samples = []
        for j in range(self.n_particle):
            posterior_samples, accept_probs, mcmc, kernel, logdensities, proposed_logdensities, = MCMC_update(Model=self.SMC_Model, posterior_samples=[smc_samples[j]], env=env, training_steps_with_burnin=1, training_steps=1, stepsize=step_size_pretune[j], num_steps=L_pretune[j], precondition_matrix=precondition_matrix, USE_AUTOGRAD=USE_AUTOGRAD, WARMUP_RATIO=WARMUP_RATIO, MCMC_SHOW_DISABLE=MCMC_SHOW_DISABLE, ADAPT_STEP_SIZE=ADAPT_STEP_SIZE, ADAPT_MASS_MATRIX=ADAPT_MASS_MATRIX, kernel='HMC')
            H_change[j] = mcmc.kernel.H_change[-1]
            # pretune_samples.append(posterior_samples[-1])
            proposed_samples.append(mcmc.proposed_samples[-1])
        ESJD_pretune = compute_ESJD(particles_ls=smc_samples, proposed_ls=proposed_samples, H_change_ls=[H_change], precondition_matrix=precondition_matrix, L=L_pretune)
        L, step_size = resample_L_stepsize(ESJD=ESJD_pretune, L=L_pretune, step_size=step_size_pretune)
        L_max_pretune, step_size_max_pretune = Pretune_adaptation(H_change=H_change, L_origin=L_pretune, L_resampled=L, step_size_origin=step_size_pretune, L_max=L_max_pretune)
        return precondition_matrix, L, L_max_pretune, step_size, step_size_max_pretune
        
    def bellman_error(self, parameters):
        return -np.mean([self.SMC_Model.llh_new(parameter=p, epsilon=1)[0] for p in parameters])
    
def particles_stat(particles):
    return particles + particles ** 2        

def test_mcmc_stop(prev_smc_samples, smc_samples, prev_corr_array):
    dim = len(prev_smc_samples[0])
    prev_smc_samples_stat = particles_stat(prev_smc_samples)
    smc_samples_stat = particles_stat(smc_samples)
    corr_array = torch.tensor([np.corrcoef(prev_smc_samples_stat[:, i], smc_samples_stat[:,i])[1, 0] for i in range(dim)])
    corr_array = corr_array * prev_corr_array
    decision = True if torch.mean((corr_array > 0.1).float()) <= 0.1 else False
    return decision, corr_array

def estimate_diag_precondition(particles, weights):
    dim = len(particles[0])
    mat = np.diag(np.array([np.dot(weights, particles[:,i].numpy() ** 2) - np.dot(weights, particles[:, i].numpy()) ** 2 for i in range(dim)]))
    return torch.tensor(mat).float()

def norm(particles, matrix):
    return np.sqrt(np.sum((particles @ matrix.T) * particles, axis=-1))

def compute_ESJD(particles_ls, proposed_ls, H_change_ls, precondition_matrix, L):
    particles_ls = np.stack(particles_ls)
    proposed_ls = np.stack(proposed_ls)
    H_change_ls = np.stack(H_change_ls)
    mat = np.diag(1 / np.diag(np.array(precondition_matrix)))
    jd_norm = (norm(particles_ls-proposed_ls, matrix=mat) / L * np.minimum(np.exp(H_change_ls), 1)).numpy()
    ESJD =  np.mean(jd_norm, axis=0)

    ESJD[np.isnan(ESJD)] = 0
    return ESJD

def resample_L_stepsize(ESJD, L, step_size):
    n_particles = len(L)
    if np.all(ESJD == 0):
        Lambda = np.ones(len(ESJD)) / len(ESJD)
    else:
        Lambda = ESJD / np.sum(ESJD)
    h_indices = np.random.choice(range(n_particles), size=n_particles, replace=True, p=Lambda)
    L = L[h_indices]
    step_size = step_size[h_indices]
    return L, step_size

def quantile_regression(target, H_change, step_size):

    abslogtarget = np.abs(np.log(target))

    if np.isinf(H_change).any() or np.isnan(H_change).any():
        indices = np.isfinite(H_change)
        H_change = H_change[indices]
        step_size = step_size[indices]
    try:
        step_size_max_simple = torch.max(step_size[np.abs(H_change)<abslogtarget])
    except:
        step_size_max_simple = 0

    H_change = np.clip(np.abs(H_change), 0, 1e6)
    reg = QuantReg(H_change.numpy(), step_size.numpy()**2)
    quant_param = reg.fit(0.5).params
    step_size_max_quant = np.sqrt(abslogtarget / quant_param)

    step_size_max = np.maximum(step_size_max_quant, step_size_max_simple)

    return step_size_max

def Pretune_adaptation(H_change, L_origin, L_resampled, step_size_origin, L_max):
    
    #L
    L80 = np.percentile(L_origin, 80)
    L20 = np.percentile(L_origin, 20)

    if (L_resampled > L80).float().mean() > 0.5:
        L_max = L_max + 5
    elif (L_resampled < L20).float().mean() > 0.5:
        if L_max > 5:
            L_max = L_max - 5

    #stepsize
    step_size_max = quantile_regression(target=0.9, H_change=H_change, step_size=step_size_origin)

    return L_max, step_size_max

def generate_weights_0(epsilon, weights, Model, smc_samples, epsilon_0, raw_weights=False):
    if hasattr(epsilon, "__len__"):
        epsilon = epsilon[0]
    llh=[Model.llh_new(parameter=p, epsilon=epsilon)[0] for p in smc_samples]
    lg_new_weights = torch.tensor([w + l for w, l in zip(weights, llh)])
    # print(lg_new_weights)
    return lg_new_weights if raw_weights else lg_new_weights - torch.logsumexp(lg_new_weights, dim=-1)

def generate_weights(epsilon, weights, Model, smc_samples, epsilon_0, raw_weights=False):
    if hasattr(epsilon, "__len__"):
        epsilon = epsilon[0]
    lg_new_weights = torch.tensor([w + (Model.llh_new(parameter=p, epsilon=epsilon)[0] - Model.llh_new(parameter=p, epsilon=epsilon_0)[0]) for w, p in zip(weights, smc_samples)])
    # print(lg_new_weights)
    return lg_new_weights if raw_weights else lg_new_weights - torch.logsumexp(lg_new_weights, dim=-1) 

def ESS_Matching(epsilon, alpha, Model, weights, smc_samples, generate_weights_fn, epsilon_0):
    # print(epsilon, alpha, Model, weights, smc_samples, generate_weights_fn, epsilon_0)
    lg_new_weights = generate_weights_fn(epsilon=epsilon, weights=weights, Model=Model, smc_samples=smc_samples, epsilon_0=epsilon_0, raw_weights=True)
    ess = SMC.ESS(lg_new_weights)
    target_ess = max(1, alpha * SMC.ESS(weights))
    # print(epsilon, epsilon_0, weights)
    # print(ess, target_ess)
    # print('is nan', torch.isnan(ess))
    # raise ValueError('nan')
    return ess - target_ess

def get_SMC_Model(obs, model, env, epsilon):
    prior = IsotropicGaussianPrior(sd=PRIOR_SIGMA)
    abclikelihood = GaussianABCLikelihood(epsilon=epsilon)
    old_data = torch.tensor(obs._buffers["rewards"], dtype=torch.float32)
    new_data = torch.tensor(obs._new_data_buffers["rewards"], dtype=torch.float32)
    r_hat_old = partial(generate_samples, model=model, obs=obs._buffers)
    r_hat_new = partial(generate_samples, model=model, obs=obs._new_data_buffers)
    llh_transform_grad_fn_old = lambda parameter:  tabular_indicator_deterministic(para=parameter.reshape(env.n_cell + (env.action_space.n, )), model=model, obs=obs._buffers)#standard form
    llh_transform_grad_fn_new = lambda parameter:  tabular_indicator_deterministic(para=parameter.reshape(env.n_cell + (env.action_space.n, )), model=model, obs=obs._new_data_buffers)
    Model = DeterministicSRModelSMC(prior=prior, abclikelihood=abclikelihood, data=obs, old_data=old_data, new_data=new_data, llh_transform_fn_old=r_hat_old, llh_transform_fn_new=r_hat_new, llh_transform_grad_fn_old=llh_transform_grad_fn_old, llh_transform_grad_fn_new=llh_transform_grad_fn_new)
    return Model

def display_smc_results(smc, n=0, figure_path=None, save=False, episode='', repeat=''):
    figure_name = f'SMCSamplesE{episode}R{repeat}'
    len_sample = len(smc.samples)
    samples = smc.samples[n:]
    dim = samples.shape
    fig,ax = plt.subplots(env.n_cell[0]-1, env.n_cell[1]-1, figsize=(dim[2]*6, dim[3]*6))
    for i in range(dim[2] - 1):
        for k in range(i+1):
            for j in range(dim[1]):
                ax[i,k].scatter(range(n, len_sample), samples[:, j, i, k, 0], alpha=smc._weights_history[n:, j])
                ax[i,k].plot(range(n, len_sample), samples[:, j, i, k, 0], linestyle='-', alpha=0.6)
                # ax[i,k].scatter(range(dim[0]), samples[:, j, i, k, 1], label=f"left {j}", alpha=smc._weights_history[:,j])
                # ax[i,k].plot(range(dim[0]), samples[:, j, i, k, 1], linestyle='-')
                # ax[i,k].hlines(Q_star[i, k, 0], xmin=0, xmax=len(samples), linestyle="--", label="true right",color="green")
            ax[i, k].hlines(Q_star[i, k, 0], xmin=n, xmax=len_sample, linestyle="--", label="true right", color="purple")
            ax[i, k].set_title([i, k, 0])
    handles, labels = ax[i,k].get_legend_handles_labels()
    ax[i, k].legend(handles, labels, bbox_to_anchor=(0.7, 1.5), loc='right')
    if save:
        plt.savefig(f'{figure_path+figure_name}0.png', bbox_inches='tight')
    plt.show()
    plt.close()

    fig2,ax2 = plt.subplots(env.n_cell[0]-1, env.n_cell[1]-1, figsize=(dim[2]*6, dim[3]*6))
    for i in range(dim[2] - 1):
        for k in range(i+1):
            for j in range(dim[1]):
                # ax2[i,k].scatter(range(dim[0]), samples[:, j, i, k, 0], label=f"right {j}", alpha=smc._weights_history[:,j])
                # ax2[i,k].plot(range(dim[0]), samples[:, j, i, k, 0], linestyle='--')
                ax2[i,k].scatter(range(n, len_sample), samples[:, j, i, k, 1], alpha=smc._weights_history[n:, j])
                ax2[i,k].plot(range(n, len_sample), samples[:, j, i, k, 1], linestyle='-', alpha=0.6)
                # ax2[i,k].hlines(Q_star[i, k, 0], xmin=0, xmax=len(samples), linestyle="--", label="true right",color="green")
            ax2[i,k].hlines(Q_star[i, k, 1], xmin=n, xmax=len_sample, linestyle="--", label="true left", color="purple")
            ax2[i,k].set_title([i, k, 1])
    handles, labels = ax2[i,k].get_legend_handles_labels()
    ax2[i, k].legend(handles, labels, bbox_to_anchor=(0.7, 1.5), loc='right')
    if save:
        plt.savefig(f'{figure_path+figure_name}1.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    # from tqdm import tqdm
    import json
    from mcmcplot import mcmcplot as mcp
    from arviz import ess, plot_autocorr, plot_trace
    from sklearn.metrics import mean_squared_error
    import datetime
    import argparse
    '''module import'''
    from Environment.GridWorld import *
    from Environment.Maze import *
    from Environment.DeepSea import *
    from model import *
    from QLearning import *
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--training_step', default=MCMC_T, type=int)
    parser.add_argument('-t', '--time', default=datetime.datetime.now().strftime("%f"))
    parser.add_argument('-s', '--save', default=SAVE)
    parser.add_argument('-p', '--show', default=SHOW)
    parser.add_argument('-e', '--epsilon', default=EPSILON, type=float)
    parser.add_argument('-n', '--stepsize', default=STEPSIZE, type=float)
    parser.add_argument('--traj_len', default=TRAJECTORY_LENGTH, type=float)
    parser.add_argument('--mass', default=MASS, type=float)
    parser.add_argument('--seed', default=SEED, type=int)
    parser.add_argument('--MCMC', default=True, action='store_false', help='Bool type')
    parser.add_argument('-g', '--Greedy', default=GREEDY, action='store_true', help='Bool type')
    parser.add_argument('--Env', default=ENV_NAME)
    parser.add_argument('--sto', default=False)
    parser.add_argument('--online', default=False, action='store_true', help='Bool type')
    parser.add_argument('--auto', default=False, action='store_true', help='Bool type')
    parser.add_argument('--warmup', default=WARMUP_RATIO, type=float)
    parser.add_argument('--transform', default=TRANSFORM)
    parser.add_argument('--precondition', default=False, action='store_true', help='Bool type')
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
    TRAJECTORY_LENGTH = args.traj_len
    MASS = args.mass
    GREEDY = args.Greedy
    env_name = args.Env
    STOCHASTIC = args.sto
    ONLINE_LEARNING = args.online
    USE_AUTOGRAD = args.auto
    WARMUP_RATIO = args.warmup
    warmup_steps = int(training_steps * WARMUP_RATIO)
    TRANSFORM = args.transform
    use_precondition = args.precondition
    
    ADAPT_STEP_SIZE = True if WARMUP_RATIO > 0 else False
    ADAPT_MASS_MATRIX = True if WARMUP_RATIO > 0 else False
    KERNEL_NAME = 'NUTS'
    EPISODES = 100

    N_PARTICLE = 20
    training_steps_with_burnin = training_steps#int(training_steps * (1 + BURN_IN))
    random.seed(seed)
    pyro.set_rng_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if save:
        dir = f'../SMC/{time}/'
        if not os.path.exists(dir):
            os.makedirs(dir)
            with open('../parameter.py', 'r') as para_file:
                parameters = para_file.read()
            with open(f'{dir}para.txt', 'w') as para_txt:
                para_txt.write(parameters)
                for key, value in vars(args).items():
                    para_txt.write(f"{key}: {value}\n")
    
    if env_name == 'GridWorld':
        env = GridWorld((1,2), obstacles=False, stochastic=STOCHASTIC)
        # env.plot_env()
    if env_name == 'Maze':
        env = Maze()
    if env_name == 'DeepSea':
        env = DeepSea(depth=15)
        EPISODES = 2000#env.n_cell[0] * 100
    
    S = []
    if len(env.n_cell) == 1:
        S = [(i, ) for i in range(env.n_cell[0])]
    else:
        if env_name == 'DeepSea':
            for i in range(env.n_cell[0] - 1):
                for j in range(env.n_cell[1]):
                    S.append((i, j)) 
        else:
            for i in range(env.n_cell[0]):
                for j in range(env.n_cell[1]):
                    S.append((i, j)) 
    Q = np.zeros(shape=(env.n_cell + (env.action_space.n, ))) / env.observation_space.n / env.action_space.n
    A = range(env.action_space.n)
    # pi_star, Q_star, V_star = OfflineQLearning(Q, A, S, env, gamma=GAMMA, show=show, thresh=1e-3, alpha=1)
    pi_star, Q_star, V_star = DynamicProgramming(Q, A, S, env, gamma=GAMMA, show=show)
    dim = env.observation_space.n * env.action_space.n
    # model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal', mean=PRIOR_MEAN, std=PRIOR_SIGMA, gamma=GAMMA, initial_tables=Q_star, idx=FROZEN_IDX)
    # model.plot_policy(paras=np.array([Q_star]), title=f'True Values/Policy by Dynamic Programming', additional_info = V_star, show=show)
    
    #SMC initialization
    alpha = ESS_ALPHA
    
    env.reset()
    results = []
    r_all_repeat = []
    samples_all_repeat = []
    smc_all_repeat = []
    for repeat in range(REPEAT_EXPERIMENT):
        epsilon = abc_epsilon
        STEPSIZE = INITIAL_STEPSIZE
        # model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal', gamma=GAMMA)
        model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal', mean=PRIOR_MEAN, std=PRIOR_SIGMA, gamma=GAMMA, initial_tables=Q_star, idx=FROZEN_IDX)#For frozen all but one dimensions
        smc = SMC(model=model, initial_params=model.get_learnable_parameter())
        smc.update_history(model.get_learnable_parameter(), torch.log(model._weights))
        r_all_epi = [] 
        samples_all_ep = []
        obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
        s0, _ = env.reset()
        for e in range(EPISODES):
            s0, _ = env.reset()
            para = model.sample_para()
            plot_qtable(para, title=f'Sampled Q table for episode {e} repeat {repeat}', save=save, figure_path=dir)
            print(f'Episode {e} in repeat {repeat} with epsilon={epsilon}')
            plt.show()
            R = 0
            # h = 0
            # while True: #Turn on h += 1
            new_data_flag = False
            for h in tqdm(range(HORIZON)):
                
                action = model.act(s0, para, greedy=GREEDY)
                s1, r, done, *info = env.step(action)
                # if STOCHASTIC:
                #     s1_augmented = []
                #     for _ in range(M_Z):
                #         s1_augmented.append(env.step(action, state=s0)[0])
                    # print(s1_augmented)
                R += r
                new_data_flag = (obs.insert({'state0': s0, 'state1': s1, 'action': int(action), 'rewards': r, 'done': done}, unique=UNIQUE_OBS, update_new_data=True) or new_data_flag)
                s0 = s1
                if done or ( h + 1)  % FROZEN_T == 0:
                    # model.set_learnable_idx(obs)
                    smc_samples = torch.tensor(model.get_learnable_parameter())
                    #MCMC
                    # posterior_samples, accept_probs = MCMC_update(posterior_samples=smc_samples, obs=obs, model=model, env=env)[:2]
                    # mode_idx = torch.argmax(mcmc.logdensities) if 'mcmc' in vars() else None
                    pre_sample_size = len(smc.samples)
                    if TRANSFORM:
                        smc_samples = TruncatedGaussianABCLikelihood.log_neg_transform(smc_samples)
                    if new_data_flag and e > 0:
                        Model = get_SMC_Model(obs=obs, model=model, env=env, epsilon=epsilon)
                        smc.SMC_Model = Model
                        _, smc_samples = smc.update(alpha, smc_samples, episode=e, repeat=repeat)
                        # plt.plot(smc.bellman_err_l[-1])
                        # if save:
                        #     plt.savefig(f'{dir}bellmanErrE{e}R{repeat}NewData.png', bbox_inches='tight')
                        # plt.show()
                    # model.sample_random_tables(set=True)
                    obs.init_new_data_buffer()
                    plot_obs(obs, env, env_name=ENV_NAME, title=f'Eploration path till ep {e} repeat {repeat}', additional_info=V_star, figure_path=dir, save=save)
                    Model = get_SMC_Model(obs=obs, model=model, env=env, epsilon=epsilon)
                    smc.SMC_Model = Model
                    # epsilon *= 0.9 + env.n_cell[0] * 0.003
                    # _, smc_samples = smc.update(alpha, smc_samples, epsilon=epsilon, episode=e, repeat=repeat)
                    epsilon, smc_samples = smc.update(alpha, smc_samples, epsilon=epsilon, episode=e, repeat=repeat, error_lag=ERROR_LAG, error_perc=ERROR_PERCENTAGE)
                    plot_save(smc.bellman_err_l[-1], figure_path=dir, episode=e, repeat=repeat, save=save, title='bellmanErr')
                    if TRANSFORM:
                        smc_samples = TruncatedGaussianABCLikelihood.neg_exp_transform(smc_samples)
                    display_smc_results(smc, n=pre_sample_size, episode=e, repeat=repeat, save=save, figure_path=dir)
                if done:
                    print("done with", h + 1, 'steps')
                    print('Return', R)
                    break
                # h += 1
            r_all_epi.append(R)
            samples_all_ep.append(smc.samples)
            explore_pct = [model.get_parameter().numpy()[:, i, i, 0] > model.get_parameter().numpy()[:, i, i, 1] for i in range(env.n_cell[0] - 1)]
            explore_pct_all = np.sum([np.logical_and.reduce(explore_pct[:i + 1], axis=0) for i in range(len(explore_pct))], axis=1) / len(smc_samples)
            plot_save(smc.epsilon_epi, figure_path=dir, episode=e, repeat=repeat, save=save, title='Epsilon')
            plot_save(smc.ess_history[:], figure_path=dir, repeat=repeat, save=save, title='ESS')
            print('explore percentage', explore_pct_all)
            if save:
                save_results(results=r_all_epi, folder='Returns', dir=dir, stochastic=STOCHASTIC, episode=e, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat, episodic=False)
                save_results(results=samples_all_ep, folder='Samples', dir=dir, stochastic=STOCHASTIC, episode=e, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat, episodic=False)
                save_results(results=obs, folder='Obs', dir=dir, stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat)
            if len(obs._buffers['state0']) >= smc.params_dim:
                print('============================', '\n', f'Finished with {e} Episodes')
                break
        r_all_repeat.append(r_all_epi)
        smc_all_repeat.append(smc)
        samples_all_repeat.append(samples_all_ep)
        if save:
            save_results(results=r_all_repeat, folder='Returns', dir=dir, stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat)        
            save_results(results=samples_all_repeat, folder='Samples', dir=dir, stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat)
            save_results(results=obs, folder='Obs', dir=dir, stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat)
        plot_return_vs_episodes(r_all_epi, repeat=repeat, save=save, figure_path=dir)
        display_smc_results(smc, save=save, figure_path=dir)
    # plot_return_vs_episodes_repeat(r_all_repeat, save=save, figure_path=dir)
    if show:
        plt.show()
