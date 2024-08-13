import numpy as np
from scipy.optimize import bisect
from scipy.optimize import root
import sys
import os
import gc
from memory_profiler import profile
import objgraph
from weightedcorrs import weightedcorrs
# import matplotlib
# matplotlib.use('MacOSX')
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
        self.adapt_alg = adapt_alg
        self.reset_stat()
        # self.kernel = 'HMC' if adapt_alg == 'pretune' else 'NUTS'

    def set_weights(self, weights):
        self._weights = weights
        self.model.set_weights(weights)
    
    # @profile
    def adaptvie_mcmc_move(self, epsilon_0, smc_samples, precondition_matrix, early_stop=EARLY_STOP):
        L, L_max_pretune, step_size, step_size_max_pretune = self.pretune(smc_samples, step_size_max_pretune=self.step_size_max_pretune, L_max_pretune=self.L_max_pretune, precondition_matrix=precondition_matrix)
        corr_stat = np.ones(self.params_dim)
        if early_stop:
            corr_stat_all = torch.tensor(corr_stat).unsqueeze(0).clone()
            gelman_rubin_all = torch.tensor([])
        accept_probs_all = torch.tensor([])
        mcmc_samples = smc_samples.unsqueeze(0).clone()
        for m in range(training_steps_with_burnin):
            accept_probs_particle = torch.zeros(self.n_particle)
            for j in range(self.n_particle):
                posterior_samples, accept_probs, mcmc, kernel, logdensities, proposed_logdensities, = MCMC_update(Model=self.SMC_Model, posterior_samples=[smc_samples[j]], env=env, training_steps_with_burnin=1, training_steps=1, stepsize=step_size[j], num_steps=L[j], precondition_matrix=precondition_matrix, USE_AUTOGRAD=USE_AUTOGRAD, WARMUP_RATIO=WARMUP_RATIO, MCMC_SHOW_DISABLE=MCMC_SHOW_DISABLE, ADAPT_STEP_SIZE=ADAPT_STEP_SIZE, ADAPT_MASS_MATRIX=ADAPT_MASS_MATRIX, kernel='HMC')
                smc_samples[j] = posterior_samples[-1]
                accept_probs_particle[j] = accept_probs[-1]
            accept_probs_all = torch.cat((accept_probs_all, accept_probs_particle.unsqueeze(0)))
            mcmc_samples = torch.cat((mcmc_samples, smc_samples.unsqueeze(0)))
            # try:
            #     step_size_max_pretune = step_size_max_pretune.numpy()
            # except:
            #     pass
            if early_stop:
                # test_dec, corr_stat = mcmc_stop(mcmc_samples, smc_samples, weights=self._weights, save=save, show=show, figure_path=dircty, figure_name=f'E{self.episode}R{repeat}loop{self.loop}Epsln{epsilon_0}Steps{m}', corr_stat_all=corr_stat_all)
                test_dec, gb = gelman_rubin(mcmc_samples, smc_samples, corr_stat, weights=self._weights, thresh=GELMAN_RUBIN, save=save, show=show, figure_path=dircty, figure_name=f'E{self.episode}R{repeat}loop{self.loop}Epsln{epsilon_0}StpMax{step_size_max_pretune}Steps{m}', corr_stat_all=corr_stat_all, gelman_rubin_all=gelman_rubin_all)
                # test_dec, corr_stat = test_mcmc_stop(mcmc_samples, smc_samples, corr_stat, weights=self._weights, save=save, show=show, figure_path=dircty, figure_name=f'E{self.episode}R{repeat}loop{self.loop}Epsln{epsilon_0}Steps{m}', corr_stat_all=corr_stat_all)
                # corr_stat_all = torch.cat((corr_stat_all, torch.tensor(corr_stat).unsqueeze(0)))
                # gelman_rubin_all = torch.cat((gelman_rubin_all, torch.tensor(gb).unsqueeze(0)))
                if test_dec:
                    print('MCMC stopped at', m, 'moves')
                    break
        print('averaged accept prob', torch.mean(accept_probs_all, dim=0)[:5])
        print('pretune results, stepsize_max, L_max=', step_size_max_pretune, L_max_pretune, step_size[:5])
        return smc_samples, mcmc_samples, corr_stat, m
    
    def check_and_resample(self, smc_samples, weights):
        if self.ESS(weights) < self.min_ess * self.n_particle:
            # print('Resampled')
            smc_samples, weights = self.resample()
        return smc_samples, weights
    
    # @profile
    def update(self, alpha, smc_samples, epsilon=None, error_lag=2, error_perc=3e-4, new_data_flag=False, episode='', repeat=''):
        if self.episode != episode:
            if episode != 0:
                self.epsilon_history.append(self.epsilon_epi)
                # self.mcmc_steps.append(self.mcmc_steps_epi)
            self.epsilon_epi = []
            self.mcmc_steps_epi = []
            self.episode = episode
            self.repeat = repeat
        
        if epsilon is None or episode == 0:
            new_data_flag = True
            self.loop = 1
            epsilon = self.SMC_Model.abclikelihood.epsilon
            print('epsilon_0 ==========')
            epsilon_0 = epsilon
            # epsilon_0, a, b = self.find_epsilon_0(alpha=alpha, smc_samples=smc_samples, generate_weights_fn=generate_weights_0, epsilon_0=epsilon, a=epsilon, b=epsilon*10, lower_side=False, show=show)
            print(epsilon_0)
            self.SMC_Model.new_epsilon = epsilon_0
            self.epsilon_epi.append(epsilon_0)
            # epsilon_0 = epsilon * 2
            weights = generate_weights_0(epsilon=epsilon_0, weights=self._weights, Model=Model, smc_samples=smc_samples, epsilon_0=epsilon_0)
            # print('weights0', weights)
            if torch.any(torch.isnan(weights)):
                print(epsilon_0, self._weights, smc_samples)
                raise ValueError('0')
            self.set_weights(weights)
            # if self.adapt_alg == 'pretune':
            #     precondition_matrix, L, L_max_pretune, step_size, step_size_max_pretune = self.pretune(smc_samples)
            if self.adapt_alg == 'pretune':
                precondition_matrix = torch.tensor(np.eye(self.params_dim)).float()
                smc_samples, mcmc_samples, corr_stat, m = self.adaptvie_mcmc_move(epsilon_0, smc_samples, precondition_matrix)
            smc_samples, weights = self.check_and_resample(smc_samples, weights)
            self.update_history(smc_samples=smc_samples, weights=weights)
            if episode == 0:
                epsilon = self.SMC_Model.abclikelihood.epsilon = epsilon_0
                new_data_flag = False
        else:
            if STOPPING_CRITERIA == 'fixed_reduce':
                print('target epsilon for loop 3', epsilon)
            epsilon_0 = self.SMC_Model.abclikelihood.epsilon
        pre_epsilon_0 = epsilon_0
        bellman_err = [self.bellman_error(smc_samples)]
        epsilon_l = [epsilon_0]
        counter = 0
        max_iter = 100
        if not new_data_flag:
            self.epsilon_epi.append(epsilon_0)
            self.epsilon_all_history.append(epsilon_0)
            self.loop = 3
        else:
            self.loop = 2
        bellman_err_improve = []
        print('Tuning down Epsilon with new data flag =', new_data_flag, 'target', epsilon)
        while True:
            # epsilon_0 > epsilon:
        # while counter < max_iter:
            # epsilon_0 = max(epsilon, self.find_epsilon_0(alpha=alpha, smc_samples=smc_samples, generate_weights_fn=generate_weights, epsilon_0=epsilon_0, a=epsilon_0*0.002, b=pre_epsilon_0))
            # epsilon_0, a, b = self.find_epsilon_0(alpha=alpha, smc_samples=smc_samples, generate_weights_fn=generate_weights, epsilon_0=epsilon_0, a=epsilon_0*0.2, b=pre_epsilon_0, show=show)
            # epsilon_0 = max(epsilon_0, epsilon) if (new_data_flag or STOPPING_CRITERIA=='fixed_reduce') else epsilon_0
            epsilon_0 = epsilon
            # epsilon_0 *= 0.9
            self.epsilon_epi.append(epsilon_0)
            if not new_data_flag:
                self.epsilon_all_history.append(epsilon_0)
            print('epsilon_1', epsilon_0)
            sys.stdout.flush()
            weights = generate_weights(epsilon=epsilon_0, weights=self._weights, Model=Model, smc_samples=smc_samples, epsilon_0=pre_epsilon_0)
            # print('weights1', weights)
            if torch.any(torch.isnan(weights)):
                print(epsilon_0, self._weights, smc_samples, pre_epsilon_0)
                raise ValueError(epsilon_0)
            self.set_weights(weights)
            self.SMC_Model.new_epsilon = epsilon_0
            # self.SMC_Model.epsilon = epsilon_0
                # self.update_history(smc_samples=smc_samples, weights=weights)
            # smc_samples, weights = self.check_and_resample(smc_samples, weights)
            if self.adapt_alg == 'pretune':
                precondition_matrix = estimate_diag_precondition(particles=smc_samples, weights=weights)
                smc_samples, mcmc_samples, corr_stat, m = self.adaptvie_mcmc_move(epsilon_0, smc_samples, precondition_matrix)
                # sys.stdout.flush()
                if not new_data_flag:
                    self.mcmc_steps_epi.append(m)
                    if m == training_steps_with_burnin - 1 and self.adapt_alg == 'pretune' and STOPPING_CRITERIA == 'natural_reduce':
                        maximum_epsilon = epsilon_0 * 10
                        self.loop = 4
                        while not gelman_rubin(mcmc_samples, smc_samples, thresh=GELMAN_RUBIN*1.5, prev_corr_array=corr_stat, weights=self._weights)[0] and epsilon_0 < maximum_epsilon:
                        # while not test_mcmc_stop(mcmc_samples, smc_samples, corr_stat, weights=self._weights, thresh=3*CORR_THRESHOLD_PRODUCT)[0] and epsilon_0 < maximum_epsilon:
                        # while not mcmc_stop(mcmc_samples, smc_samples, weights=self._weights, thresh=1.2*CORR_THRESHOLD)[0] and epsilon_0 < maximum_epsilon:
                            print('epsilon too small', epsilon_0)
                            epsilon_0 = self.find_epsilon_0(alpha=compromise_alpha, smc_samples=smc_samples, generate_weights_fn=generate_weights, epsilon_0=epsilon_0, a=epsilon_0, b=epsilon_0*5, max_epsilon=2*epsilon_0, lower_side=False, show=show)[0]
                            self.epsilon_epi.append(epsilon_0)
                            self.epsilon_all_history.append(epsilon_0)
                            print('increased epsilon', epsilon_0)
                            weights = generate_weights(epsilon=epsilon_0, weights=self._weights, Model=Model, smc_samples=smc_samples, epsilon_0=pre_epsilon_0)
                            self.set_weights(weights)
                            precondition_matrix = estimate_diag_precondition(particles=smc_samples, weights=weights)
                            smc_samples, mcmc_samples, corr_stat, m = self.adaptvie_mcmc_move(epsilon_0, smc_samples, precondition_matrix)
                            smc_samples, weights = self.check_and_resample(smc_samples, weights)
                            self.update_history(smc_samples=smc_samples, weights=weights)
                            if not gelman_rubin(mcmc_samples, smc_samples, thresh=GELMAN_RUBIN*1.5, prev_corr_array=corr_stat, weights=self._weights)[0]:
                                print('MCMC not working, retry with epsilon', epsilon_0)
                                self.epsilon_epi.append(epsilon_0)
                                self.epsilon_all_history.append(epsilon_0)
                                weights = generate_weights(epsilon=epsilon_0, weights=self._weights, Model=Model, smc_samples=smc_samples, epsilon_0=pre_epsilon_0)
                                self.set_weights(weights)
                                precondition_matrix = estimate_diag_precondition(particles=smc_samples, weights=weights)
                                smc_samples, mcmc_samples, corr_stat, m = self.adaptvie_mcmc_move(epsilon_0, smc_samples, precondition_matrix)
                                smc_samples, weights = self.check_and_resample(smc_samples, weights)
                                self.update_history(smc_samples=smc_samples, weights=weights)
                        break
            if self.adapt_alg == 'NUTS':
                for j in range(self.n_particle):
                    posterior_samples, accept_probs, mcmc, kernel, logdensities, proposed_logdensities, = MCMC_update(Model=self.SMC_Model, posterior_samples=[smc_samples[j]], env=env, training_steps_with_burnin=training_steps_with_burnin, training_steps=training_steps, USE_AUTOGRAD=USE_AUTOGRAD, WARMUP_RATIO=WARMUP_RATIO, MCMC_SHOW_DISABLE=MCMC_SHOW_DISABLE, ADAPT_STEP_SIZE=ADAPT_STEP_SIZE, ADAPT_MASS_MATRIX=ADAPT_MASS_MATRIX)
                    # print(posterior_samples.shape)
                    smc_samples[j] = posterior_samples[-1]
            smc_samples, weights = self.check_and_resample(smc_samples, weights)
            self.update_history(smc_samples=smc_samples, weights=weights)
            pre_epsilon_0 = epsilon_0
            if self.loop == 3 and STOPPING_CRITERIA == 'natural_reduce':
                e1 = min(bellman_err)
                current_bellman_error = self.bellman_error(smc_samples)
                bellman_err.append(current_bellman_error)
                self.bellman_err_l.append(current_bellman_error)
                bellman_err_improve.append(((e1 - current_bellman_error) / e1) > error_perc)
                print('bellman error', bellman_err, - (current_bellman_error - e1) / e1)
            # bellman_err_improve = np.array([(e1 - e2) / e1 if e1 != 0 else 0 for e1, e2 in zip(bellman_err[:-1], bellman_err[1:])])
            epsilon_l.append(epsilon_0)
            if new_data_flag or STOPPING_CRITERIA == 'fixed_reduce':
                if epsilon_0 <= epsilon:
                    break
            elif STOPPING_CRITERIA == 'natural_reduce' and len(bellman_err_improve) >= error_lag and np.sum(bellman_err_improve[-error_lag:]) == 0:
                # epsilon_0 = epsilon_l[- error_lag - 1]
                print('stopped at epsilon=', epsilon_0)
                print('epsilon_l', epsilon_l)
                # self.remove_history(error_lag)
                break
            counter += 1
            if counter >= max_iter:
                break
        # if STOPPING_CRITERIA == 'natural_reduce' and (not new_data_flag):
            # self.bellman_err_l.append(bellman_err)
        return epsilon_0, self.model.get_learnable_parameter()
        
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
        self.epsilon_all_history = []
        self.bellman_err_l = []
        self.mcmc_steps_epi = []
        if self.adapt_alg == 'pretune':
            self.L_max_pretune, self.step_size_max_pretune = 100, 0.1
        
    def update_history(self, smc_samples=None, weights=None):
        if smc_samples is not None:
            self.model.set_learnable_parameter(torch.tensor(smc_samples))
            self.samples = torch.cat((self.samples, self.model.get_parameter().unsqueeze(0)))
        if weights is not None:
            self._weights_history = torch.cat((self._weights_history, torch.exp(weights).unsqueeze(0)))
            self.ess_history = torch.cat((self.ess_history, self.ESS(weights).unsqueeze(0)))

    def remove_history(self, index):
        self.samples  = self.samples[: - (index - 1)]
        self._weights_history = self._weights_history[:- (index - 1)]
        self.ess_history  = self.ess_history[: - (index - 1)]
        self.epsilon_all_history = self.epsilon_all_history[: - index]
        self.set_weights(torch.log(self._weights_history[- 1]))
        self.model.set_parameter(self.samples[- 1])

    def plot_ess_fn(self, epsilon_0, smc_samples, generate_weights_fn, a, b, new_epsilon, alpha, show=False):
        # print(smc_samples[0, 0], self._weights, epsilon_0)
        ess_partial= partial(ess_, Model=self.SMC_Model, weights=self._weights, smc_samples=smc_samples, generate_weights_fn=generate_weights_fn, epsilon_0=epsilon_0, ess_fn=self.ESS)
        e_l = np.linspace(a, b*1.5, 100)
        ess_l = [ess_partial(e) for e in e_l]
        plot_ess(e_l, ess_l, epsilon_0, self.ESS(self._weights), alpha, new_epsilon, save=save, loop_num=self.loop, episode=self.episode, repeat=self.repeat, figure_path=dircty, show=show)

    def find_epsilon_0(self, alpha, smc_samples, generate_weights_fn, epsilon_0, a, b, lower_side=True, method='bisect', tol=1e-7, max_epsilon=PRIOR_SIGMA, min_epsilon=1e-3, show=False, max_alpha=5):
        if show:
            print(a, b, alpha)
            sys.stdout.flush()
        if b > 2 * max_epsilon:
            print(a, b, alpha)
            # self.plot_ess_fn(epsilon_0, smc_samples, generate_weights_fn, a, b, new_epsilon=epsilon_0, show=show, alpha=alpha)
            print('max epsilon')
            return max_epsilon, a, b
        if a < min_epsilon/2:
            return epsilon_0, a, b
            # print('illegal range', a, max_epsilon, alpha, a, max_epsilon*0.9)
            # if alpha >= max_alpha:
            #     return max_epsilon, a, b
            # new_epsilon, a, b = self.find_epsilon_0(alpha*1.05, smc_samples, generate_weights_fn, epsilon_0, a, max_epsilon*0.9, lower_side=lower_side, method=method, show=show)
            # # raise ValueError(a, b, alpha, self.SMC_Model, self._weights, smc_samples, generate_weights_fn, epsilon_0)
            # return max_epsilon, a, b
        # Use optimization to find epsilon_0 that satisfies the ESS condition
        if method == 'bisect':
            ess_simple = partial(ESS_Matching, alpha=alpha, Model=self.SMC_Model, weights=self._weights, smc_samples=smc_samples, generate_weights_fn=generate_weights_fn, epsilon_0=epsilon_0)
            ess_partial= partial(ess_, Model=self.SMC_Model, weights=self._weights, smc_samples=smc_samples, generate_weights_fn=generate_weights_fn, epsilon_0=epsilon_0, ess_fn=self.ESS)
            # e_l = np.linspace(a*0.5, b*1.5, 100)
            # ess_l = [ess_partial(e) for e in e_l]
            # plot_ess(e_l, ess_l, epsilon_0, self.ESS(self._weights), alpha)
            try:
                result = bisect(ess_simple, a=a, b=b, xtol=tol, maxiter=20, disp=False)
                # print('try bisect', result, a, b, alpha)
                if result > max_epsilon:
                    # self.plot_ess_fn(epsilon_0, smc_samples, generate_weights_fn, a, b, new_epsilon=epsilon_0, show=show, alpha=alpha)
                    print('original epsilon')
                    return max_epsilon, a, b
            except ValueError as ve:
                # print(ve)
                if lower_side and abs(ess_partial(b) - max(1, alpha*self.ESS(self._weights))) <= tol:
                    # print('b side')
                    # self.plot_ess_fn(epsilon_0, smc_samples, generate_weights_fn, a, b, new_epsilon=b, show=show, alpha=alpha)
                    return b, a, b
                if abs(ess_partial(a) - max(1, alpha*self.ESS(self._weights))) <= tol:
                    # print('a side')
                    # self.plot_ess_fn(epsilon_0, smc_samples, generate_weights_fn, a, b, new_epsilon=a, show=show, alpha=alpha)
                    return a, a, b
                # raise ValueError(a, b, alpha, self.SMC_Model, self._weights, smc_samples, generate_weights_fn, epsilon_0)
                b = b if lower_side else b * 2
                a = a * 0.8 if lower_side else a
                # print('Bisect with new alpha', alpha, a, b)
                new_epsilon, a, b = self.find_epsilon_0(alpha, smc_samples, generate_weights_fn, epsilon_0, a, b, lower_side=lower_side, method=method, show=show)
                # self.plot_ess_fn(epsilon_0, smc_samples, generate_weights_fn, a, b, new_epsilon=new_epsilon, show=show, alpha=alpha)
                return new_epsilon, a, b
            except RecursionError as re:
                print(re ,a, b, alpha, self.SMC_Model, self._weights, smc_samples, generate_weights_fn, epsilon_0)
                return epsilon_0, a, b
            # result = bisect(ess_simple, a=a, b=b, xtol=tol, maxiter=2000)
            # self.plot_ess_fn(epsilon_0, smc_samples, generate_weights_fn, a, b, new_epsilon=result, show=show, alpha=alpha)
            # if epsilon_0 < 0.03 and (not lower_side):
            #     dsd
            result = max(result, min_epsilon)
            return result, a, b
        elif method == 'root':
            try:
                result = root(ESS_Matching, x0=b, tol=1e-3, args=(alpha, smc.SMC_Model, smc._weights, smc_samples, generate_weights_fn, epsilon_0), method='lm')
                if not result.success:
                    b = b if lower_side else b * 2
                    print('Bisect with new alpha', min(alpha*1.2, 1), a*0.8, b)
                    return self.find_epsilon_0(min(alpha*1.2, 1), smc_samples, generate_weights_fn, epsilon_0, a*0.8, b, lower_side=lower_side, method=method, show=show)
                if max(result.x) > 1e3:
                    return epsilon_0
            except:
                # raise ValueError(a, b, alpha, self.SMC_Model, self._weights, smc_samples, generate_weights_fn, epsilon_0)
                b = b if lower_side else b * 2
                print('Bisect with new alpha', min(alpha*1.2, 1), a*0.8, b)
                return self.find_epsilon_0(min(alpha*1.2, 1), smc_samples, generate_weights_fn, epsilon_0, a*0.8, b, lower_side=lower_side, method=method, show=show)
            return max(result.x) 
        else:
            raise NotImplementedError('No method implemented for method', method)
    
    def pretune(self, smc_samples, precondition_matrix, step_size_max_pretune=0.1, L_max_pretune=99):
        H_change = torch.zeros(self.n_particle)
        step_size_pretune = torch.rand(self.n_particle) * step_size_max_pretune
        # print('pretune, max_step_size', step_size_max_pretune, step_size_pretune)
        L_pretune = torch.randint(1, L_max_pretune + 1, size=(self.n_particle, ))
        proposed_samples = []
        for j in range(self.n_particle):
            posterior_samples, accept_probs, mcmc, kernel, logdensities, proposed_logdensities, = MCMC_update(Model=self.SMC_Model, posterior_samples=[smc_samples[j]], env=env, training_steps_with_burnin=1, training_steps=1, stepsize=step_size_pretune[j], num_steps=L_pretune[j], precondition_matrix=precondition_matrix, USE_AUTOGRAD=USE_AUTOGRAD, WARMUP_RATIO=WARMUP_RATIO, MCMC_SHOW_DISABLE=MCMC_SHOW_DISABLE, ADAPT_STEP_SIZE=ADAPT_STEP_SIZE, ADAPT_MASS_MATRIX=ADAPT_MASS_MATRIX, kernel='HMC')
            H_change[j] = mcmc.kernel.H_change[-1]
            # pretune_samples.append(posterior_samples[-1])
            proposed_samples.append(mcmc.proposed_samples[-1])
        ESJD_pretune = compute_ESJD(particles_ls=smc_samples, proposed_ls=proposed_samples, H_change_ls=[H_change], precondition_matrix=precondition_matrix, L=L_pretune)
        L, step_size = resample_L_stepsize(ESJD=ESJD_pretune, L=L_pretune, step_size=step_size_pretune)
        self.L_max_pretune, self.step_size_max_pretune = Pretune_adaptation(H_change=H_change, L_origin=L_pretune, L_resampled=L, step_size_origin=step_size_pretune, L_max=L_max_pretune)
        return L, self.L_max_pretune, step_size, self.step_size_max_pretune
        
    def bellman_error(self, parameters):
        # print('self weights', self._weights)
        return -np.sum([self.SMC_Model.llh_new(parameter=p, epsilon=1)[0] for p in parameters] * np.exp(self._weights).numpy())
    
def particles_stat(particles):
    return particles + particles ** 2        

def test_mcmc_stop(mcmc_samples, smc_samples, prev_corr_array, weights=None, thresh=CORR_THRESHOLD_PRODUCT, save=False, show=False, figure_path='', figure_name='', corr_stat_all=None):
    sample_dim = mcmc_samples.shape
    prev_smc_samples = mcmc_samples[-2]
    prev_smc_samples_stat = particles_stat(prev_smc_samples)
    smc_samples_stat = particles_stat(smc_samples)
    corr_array = torch.tensor([weightedcorrs(np.array([prev_smc_samples_stat[:, i].numpy(), smc_samples_stat[:, i].numpy()]).T, np.exp(weights).numpy())['R'][0, 1] for i in range(sample_dim[2])])
    # corr_array = torch.tensor([np.corrcoef(prev_smc_samples_stat[:, i], smc_samples_stat[:, i])[1, 0] for i in range(dim)])
    corr_array = corr_array * prev_corr_array
    decision = True if torch.mean((corr_array > thresh).float()) <= 0.1 else False
    if len(mcmc_samples) == training_steps_with_burnin:
        figure_name = f'Fail{figure_name}'
    n, m = 2, 3
    if corr_stat_all is not None and decision or len(mcmc_samples) == training_steps_with_burnin:
        corr_stat_all = torch.abs(torch.cat((corr_stat_all, torch.tensor(corr_array).unsqueeze(0))))
        # corr_array_all = torch.tensor([[np.corrcoef(initial_stats[:, i], particles_stat(mcmc_samples[t])[:, i])[1, 0] for t in range(sample_dim[0])] for i in range(sample_dim[2])]) #dxT
        fig, ax = plt.subplots(n, m, figsize=(20, 20))
        for i in range(n):
            for j in range(m):
                ax[i, j].plot(mcmc_samples[:, :, m*i+j].numpy())
                ax[i, j].set_title(env.names[m*i+j])
        if show:
            plt.show()
        if save:
            plt.savefig(f'{figure_path}{figure_name}traj.png', bbox_inches='tight')
        plt.close()
        fig, ax = plt.subplots(n, m, figsize=(20, 20))
        for i in range(n):
            for j in range(m):
                # ax[i, j].plot(mcmc_samples[:, :, m*i+j].numpy())
                ax[i, j].plot(corr_stat_all[:, m*i+j].numpy())
                ax[i, j].axhline(thresh, 0, sample_dim[0], color='r', linestyle='-.')
                ax[i, j].set_title(env.names[m*i+j])
        if show:
            plt.show()
        if save:
            plt.savefig(f'{figure_path}{figure_name}statsstop.png', bbox_inches='tight')
        plt.close()
        mcmc_samples_stat = particles_stat(mcmc_samples[0])
        corr_array_all = torch.abs(torch.tensor([[np.corrcoef(mcmc_samples_stat[:, i], particles_stat(mcmc_samples[t])[:, i])[1, 0] for t in range(sample_dim[0])] for i in range(sample_dim[2])])) #dxT
        fig, ax = plt.subplots(n, m, figsize=(20, 20))
        for i in range(n):
            for j in range(m):
                # ax[i, j].plot(mcmc_samples[:, :, m*i+j].numpy())
                ax[i, j].plot(corr_array_all[m*i+j].numpy())
                ax[i, j].axhline(CORR_THRESHOLD, 0, sample_dim[0], color='r', linestyle='-.')
                ax[i, j].set_title(env.names[m*i+j])
        if show:
            plt.show()
        if save:
            plt.savefig(f'{figure_path}{figure_name}corrstop.png', bbox_inches='tight')
        plt.close()
    return decision, corr_array

def mcmc_stop(mcmc_samples, smc_samples, weights=None, thresh=CORR_THRESHOLD, save=False, show=False, figure_path='', figure_name='', corr_stat_all=None):
    # mcmc_samples: torch.tensor of shape TxNxd, where N is the number of particles and d is the dimensions of the parameters space, smc_samples: torch.tensor of shape Nxd
    sample_dim = mcmc_samples.shape
    mcmc_samples_stat = particles_stat(mcmc_samples[0])
    smc_samples_stat = particles_stat(smc_samples)
    corr_array = torch.tensor([weightedcorrs(np.array([mcmc_samples_stat[:, i].numpy(), smc_samples_stat[:, i].numpy()]).T, np.exp(weights).numpy())['R'][0, 1] for i in range(sample_dim[2])])
    # corr_array = torch.tensor([np.corrcoef(mcmc_samples_stat[:, i], smc_samples_stat[:, i])[1, 0] for i in range(sample_dim[2])]) 
    decision = True if torch.mean((corr_array > thresh).float()) <= 0.1 else False
    # n, m = 2, 3
    # if len(mcmc_samples) == training_steps_with_burnin:
    #     figure_name = f'Fail{figure_name}'
    # if decision or len(mcmc_samples) == training_steps_with_burnin:
    #     corr_array_all = torch.abs(torch.tensor([[np.corrcoef(mcmc_samples_stat[:, i], particles_stat(mcmc_samples[t])[:, i])[1, 0] for t in range(sample_dim[0])] for i in range(sample_dim[2])])) #dxT
    #     fig, ax = plt.subplots(n, m, figsize=(20, 20))
    #     for i in range(n):
    #         for j in range(m):
    #             # ax[i, j].plot(mcmc_samples[:, :, m*i+j].numpy())
    #             ax[i, j].plot(corr_array_all[m*i+j].numpy())
    #             ax[i, j].axhline(thresh, 0, sample_dim[0], color='r', linestyle='-.')
    #             ax[i, j].set_title(env.names[m*i+j])
    #     if show:
    #         plt.show()
    #     if save:
    #         plt.savefig(f'{figure_path}corrstop{figure_name}.png', bbox_inches='tight')
    #     plt.close()
    return decision, corr_array

def gelman_rubin(mcmc_samples, smc_samples, prev_corr_array, weights=None, thresh=GELMAN_RUBIN, save=False, show=False, figure_path='', figure_name='', corr_stat_all=None, gelman_rubin_all=None, plot=False):
    """
    Computes the Gelman-Rubin convergence diagnostic for MCMC chains.

    Parameters:
    - mcmc_samples: A 3D numpy array of shape (n_samples, n_chains, n_parameters).
             Samples are stored along the first dimension, each chain is
             stored along the second dimension, and parameters are stored
             along the third dimension.

    Returns:
    - gelman_rubin: A 1D numpy array containing the Gelman-Rubin statistic
                    for each parameter.
    """
    sample_dim = mcmc_samples.shape
    #for test_mcmc_stop
    prev_smc_samples = mcmc_samples[-2]
    prev_smc_samples_stat = particles_stat(prev_smc_samples)
    smc_samples_stat = particles_stat(smc_samples)
    
    chain = mcmc_samples.numpy()

    # Calculate the mean of each chain
    chain_means = np.mean(chain, axis=0)

    # Calculate the between-chain variance
    between_chain_variance = sample_dim[0] * np.var(chain_means, axis=0, ddof=1)

    # Calculate the within-chain variance
    within_chain_variance = np.mean(np.var(chain, axis=0, ddof=1), axis=0)

    # Calculate the estimated variance
    var_estimate = (1 - 1 / sample_dim[0]) * within_chain_variance + between_chain_variance / sample_dim[0]

    # Calculate the potential scale reduction factor (R-hat)
    gelman_rubin = np.sqrt(var_estimate / within_chain_variance)
    decision = True if np.mean(gelman_rubin > thresh) <= 0.4 and sample_dim[0]>=0.1 * training_steps_with_burnin else False
    if plot and len(mcmc_samples) == training_steps_with_burnin and not decision:
        # print(gelman_rubin, thresh, np.mean(gelman_rubin > thresh),  sample_dim[0], 0.1 * training_steps_with_burnin)
        figure_name = f'Fail{figure_name}'
        # corr_array = torch.tensor([weightedcorrs(np.array([prev_smc_samples_stat[:, i].numpy(), smc_samples_stat[:, i].numpy()]).T, np.exp(weights).numpy())['R'][0, 1] for i in range(sample_dim[2])])
        # corr_array = corr_array * prev_corr_array
        n = 3
        idx = np.where(gelman_rubin > thresh)[0][-9:]
        m = max(1, len(idx) // n)
        # if corr_stat_all is not None and decision or len(mcmc_samples) == training_steps_with_burnin and gelman_rubin_all is not None:
        if False:
            return
        else:
            # corr_stat_all = torch.abs(torch.cat((corr_stat_all, torch.tensor(corr_array).unsqueeze(0))))
            fig, ax = plt.subplots(n, m, figsize=(20, 20))
            for i in range(n):
                for j in range(m):
                    ax[i, j].plot(mcmc_samples[:, :, idx[m*i+j]].numpy())
                    ax[i, j].set_title(env.names[idx[m*i+j]])
            if show:
                plt.show()
            if save:
                plt.savefig(f'{figure_path}{figure_name}traj.png', bbox_inches='tight')
            plt.close()
            # fig, ax = plt.subplots(n, m, figsize=(20, 20))
            # for i in range(n):
            #     for j in range(m):
            #         # ax[i, j].plot(mcmc_samples[:, :, m*i+j].numpy())
            #         ax[i, j].plot(corr_stat_all[:, m*i+j].numpy())
            #         ax[i, j].axhline(CORR_THRESHOLD_PRODUCT, 0, sample_dim[0], color='r', linestyle='-.')
            #         ax[i, j].set_title(env.names[m*i+j])
            # if show:
            #     plt.show()
            # if save:
            #     plt.savefig(f'{figure_path}{figure_name}statsstop.png', bbox_inches='tight')
            # plt.close()
            # mcmc_samples_stat = particles_stat(mcmc_samples[0])
            # corr_array_all = torch.abs(torch.tensor([[np.corrcoef(mcmc_samples_stat[:, i], particles_stat(mcmc_samples[t])[:, i])[1, 0] for t in range(sample_dim[0])] for i in range(sample_dim[2])])) #dxT
            # fig, ax = plt.subplots(n, m, figsize=(20, 20))
            # for i in range(n):
            #     for j in range(m):
            #         # ax[i, j].plot(mcmc_samples[:, :, m*i+j].numpy())
            #         ax[i, j].plot(corr_array_all[m*i+j].numpy())
            #         ax[i, j].axhline(CORR_THRESHOLD, 0, sample_dim[0], color='r', linestyle='-.')
            #         ax[i, j].set_title(env.names[m*i+j])
            # if show:
            #     plt.show()
            # if save:
            #     plt.savefig(f'{figure_path}{figure_name}corrstop.png', bbox_inches='tight')
            # plt.close()
            # return decision, corr_array, gelman_rubin
            gelman_rubin_all = torch.cat((gelman_rubin_all, torch.tensor(gelman_rubin).unsqueeze(0)))
            fig, ax = plt.subplots(n, m, figsize=(20, 20))
            for i in range(n):
                for j in range(m):
                    ax[i, j].plot(gelman_rubin_all.T[idx[m*i+j]].numpy(), marker='o')
                    # ax[i, j].axhline(np.mean(gelman_rubin), 0, sample_dim[2], color='b', linestyle='--', label='Mean')
                    ax[i, j].axhline(thresh, 0, sample_dim[2], color='r', linestyle='-.', label='Threshold of ' + r'$\hat{R}$')
                    ax[i, j].set_title(env.names[idx[m*i+j]])
            ax[i, j].legend()
            if show:
                plt.show()
            if save:
                plt.savefig(f'{figure_path}{figure_name}gelmanstop.png', bbox_inches='tight')
            plt.close()
    return decision, gelman_rubin

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
        L_max = max(0, L_max - 5)

    #stepsize
    step_size_max = quantile_regression(target=0.5, H_change=H_change, step_size=step_size_origin)

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

@profile
def display_smc_results(smc, n=0, figure_path=None, save=False, episode='', repeat='', show=False):
    figure_name = f'E{episode}R{repeat}SMCSamples'
    len_sample = len(smc.samples)
    subsample_int = 2
    samples = smc.samples[n::subsample_int]
    dim = samples.shape
    means = torch.mean(samples, dim=1)
    stds = torch.std(samples, dim=1)
    upbd = means + stds
    lwbd = means - stds
    for a in range(env.action_size):
        fig, ax = plt.subplots(env.n_cell[0]-1, env.n_cell[1]-1, figsize=(dim[2]*4, dim[3]*4))
        for i in range(dim[2] - 1):
            for k in range(i+1):
                ax[i, k].set_ylim(-PRIOR_SIGMA, PRIOR_SIGMA)
                ax[i, k].plot(range(n, len_sample)[::subsample_int], means[:, i, k, a], label='mean')
                ax[i, k].fill_between(range(n, len_sample)[::subsample_int], lwbd[:, i, k, a], upbd[:, i, k, a], alpha=0.5, label='std')
                # for j in range(dim[1]):
                #     ax[i,k].scatter(range(n, len_sample)[::subsample_int], samples[:, j, i, k, a], alpha=smc._weights_history[n::subsample_int, j])
                #     ax[i,k].plot(range(n, len_sample)[::subsample_int], samples[:, j, i, k, a], linestyle='-', alpha=0.6)
                    # ax[i,k].scatter(range(dim[0]), samples[:, j, i, k, 1], label=f"left {j}", alpha=smc._weights_history[:,j])
                    # ax[i,k].plot(range(dim[0]), samples[:, j, i, k, 1], linestyle='-')
                    # ax[i,k].hlines(Q_star[i, k, 0], xmin=0, xmax=len(samples), linestyle="--", label="true right",color="green")
                ax[i, k].hlines(Q_star[i, k, a], xmin=n, xmax=len_sample-2, linestyle="--", label="true right", color="purple")
                ax[i, k].set_title([i, k, a])
        handles, labels = ax[i,k].get_legend_handles_labels()
        ax[i, k].legend(handles, labels, bbox_to_anchor=(0.7, 1.5), loc='right')
        if save:
            plt.savefig(f'{figure_path+figure_name}{a}.png', bbox_inches='tight')
        if show:
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
    parser.add_argument('-t', '--time', default=datetime.datetime.now().strftime("%Y%m%d_%H%M"))
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
    parser.add_argument('--load', default=False, action='store_true', help='Bool type')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--sample_path', type=str, default='')
    args = parser.parse_args()
    print(args)
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
    load = args.load
    load_path = args.load_path
    dircty=''
    
    ADAPT_STEP_SIZE = True if WARMUP_RATIO > 0 else False
    ADAPT_MASS_MATRIX = True if WARMUP_RATIO > 0 else False
    KERNEL_NAME = 'NUTS'
    EPISODES = 100

    N_PARTICLE = 100
    training_steps_with_burnin = training_steps#int(training_steps * (1 + BURN_IN))
    random.seed(seed)
    pyro.set_rng_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if env_name == 'GridWorld':
        env = GridWorld((1,2), obstacles=False, stochastic=STOCHASTIC)
        # env.plot_env()
    if env_name == 'Maze':
        env = Maze()
    if env_name == 'DeepSea':
        env = DeepSea(depth=30)
        EPISODES = 200000#env.n_cell[0] * 100
    
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
    compromise_alpha = COMPROMISE_ALPHA
    
    env.reset()
    results = []
    r_all_repeat = []
    samples_all_repeat = []
    smc_all_repeat = []
    dircty = ''
    
    time = load_path.split('/')[-1] if load else args.time
    real_time = args.time
    print('time', time)
    if save:
        dircty_top = f'../SMC/{env.n_cell[0]}/{time}/'
        if not os.path.exists(dircty_top):
            os.makedirs(dircty_top)
            if not load:
                with open('../parameter.py', 'r') as para_file:
                    parameters = para_file.read()
                with open(f'{dircty_top}para.txt', 'w') as para_txt:
                    para_txt.write(parameters)
                    for key, value in vars(args).items():
                        para_txt.write(f"{key}: {value}\n")
    
    for repeat in range(REPEAT_EXPERIMENT):
        if load:
            load_postfix = f'T{training_steps}_StoFalse_M{N_PARTICLE}_GdyFalse_Sigma{PRIOR_SIGMA}.pt'
            load_samples = torch.load(f'../SMC/{load_path}/R{repeat}/Samples{args.sample_path}_{load_postfix}')[-1:]
            load_weight = torch.load(f'../SMC/{load_path}/R{repeat}/Weights{args.sample_path}_{load_postfix}')[-1:]
            load_epsilon = torch.load(f'../SMC/{load_path}/R{repeat}/Epsilon_{load_postfix}')
            initial_obs = torch.load(f'../SMC/{load_path}/R{repeat}/Obs_{load_postfix}')
            initial_tables = load_samples[-1]
            initial_weights = load_weight[-1]
            initial_epsilon = load_epsilon[-1]
            
        else:
            initial_tables = None
            initial_weights = torch.ones(N_PARTICLE) / N_PARTICLE
            initial_obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
            initial_episode = 0
            initial_epsilon = abc_epsilon
        if save:
            dircty = f'{dircty_top}R{repeat}/'
            if not os.path.exists(dircty):
                os.makedirs(dircty)
        epsilon = initial_epsilon
        STEPSIZE = INITIAL_STEPSIZE
        # model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal', gamma=GAMMA)
        model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal', mean=PRIOR_MEAN, std=PRIOR_SIGMA, gamma=GAMMA, initial_tables=initial_tables, initial_weights=initial_weights, idx=FROZEN_IDX)#For frozen all but one dimensions
        smc = SMC(model=model, initial_params=model.get_learnable_parameter())
        if load:
            # smc = torch.load(f'../SMC/{load_path}/R{repeat}/SMC_T100_StoFalse_M0_GdyFalse_Sigma4.pt')
            smc._weights = load_weight[-1]
            smc._weights_history = load_weight
            smc.epsilon_epi = [initial_epsilon]
            smc.epsilon_all_history = load_epsilon
            r_all_epi = torch.load(f'../SMC/{load_path}/R{repeat}/Returns_{load_postfix}')
            initial_episode = len(r_all_epi)
            print('initial episode', initial_episode)
            smc.episode = initial_episode
            gc.collect()
            load = False
        else:
            # smc = SMC(model=model, initial_params=model.get_learnable_parameter())
            smc.update_history(model.get_learnable_parameter(), torch.log(model._weights))
            if repeat == 0:
                r_all_epi = [] 
        obs = initial_obs
        s0, _ = env.reset()
        for e in range(initial_episode, EPISODES):
            # objgraph.show_most_common_types()
            s0, _ = env.reset()
            para = model.sample_para()
            # plot_qtable(para, title=f'Sampled Q table for episode {e} repeat {repeat}', save=save, figure_path=dircty)
            print(f'Episode {e} in repeat {repeat} with epsilon={epsilon}')
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
                    if new_data_flag:
                        plot_obs(obs, env, env_name=ENV_NAME, title=f'ExplorationE{e}', additional_info=V_star, figure_path=dircty, save=save, show=show)
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
                        #     plt.savefig(f'{dircty}bellmanErrE{e}NewData.png', bbox_inches='tight')
                        # plt.show()
                    # model.sample_random_tables(set=True)
                    obs.init_new_data_buffer()
                    Model = get_SMC_Model(obs=obs, model=model, env=env, epsilon=epsilon)
                    smc.SMC_Model = Model
                    
                    #fixed decreasing
                    if STOPPING_CRITERIA == 'fixed_reduce':
                        epsilon *= min(1, (0.35 + e * 0.01)**0.1)
                        _, smc_samples = smc.update(alpha, smc_samples, epsilon=epsilon, episode=e, repeat=repeat)
                    
                    #Natural decreasing
                    elif STOPPING_CRITERIA == 'natural_reduce':
                        epsilon, smc_samples = smc.update(alpha, smc_samples, epsilon=epsilon, episode=e, repeat=repeat, error_lag=ERROR_LAG, error_perc=ERROR_PERCENTAGE)
                        plot_save(smc.bellman_err_l, figure_path=dircty, repeat=repeat, save=save, title='bellmanErr', show=show)
                    
                    if TRANSFORM:
                        smc_samples = TruncatedGaussianABCLikelihood.neg_exp_transform(smc_samples)
                    if len(smc.samples) - pre_sample_size <=2:
                        pre_sample_size -= min(pre_sample_size, 10)
                    # display_smc_results(smc, n=pre_sample_size, episode=e, repeat=repeat, save=save, figure_path=dircty, show=show)
                if done:
                    print("done with", h + 1, 'steps')
                    print('Return', R)
                    r_all_epi.append(R)
                    break
            explore_pct = [model.get_parameter().numpy()[:, i, i, 0] > model.get_parameter().numpy()[:, i, i, 1] for i in range(env.n_cell[0] - 1)]
            explore_pct_all = np.sum([np.logical_and.reduce(explore_pct[:i + 1], axis=0) for i in range(len(explore_pct))], axis=1) / len(smc_samples)
            # plot_save(smc.epsilon_epi, figure_path=dircty, episode=e, repeat=repeat, save=save, title=f'Epsilon{smc.epsilon_all_history[-1]}', show=show)
            plot_save(smc.epsilon_all_history[:], figure_path=dircty, repeat=repeat, save=save, title=f'Epsilon', show=show)
            # plot_save(smc.mcmc_steps_epi[:], figure_path=dircty, episode=e, repeat=repeat, save=save, title=f'MCMCSteps', show=show)
            # plot_save(smc.ess_history[:], figure_path=dircty, repeat=repeat, save=save, title='ESS')
            print('explore percentage', explore_pct_all)
            if save:
                save_results(results=r_all_epi, folder='Returns', dir=dircty, stochastic=STOCHASTIC, episode=e, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=N_PARTICLE, repeat=repeat, episodic=False)
                save_results(results=smc.samples[-100:], folder=f'Samples{real_time}', dir=dircty, stochastic=STOCHASTIC, episode=e, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=N_PARTICLE, repeat=repeat, episodic=False)
                save_results(results=smc._weights_history, folder=f'Weights{real_time}', dir=dircty, stochastic=STOCHASTIC, episode=e, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=N_PARTICLE, repeat=repeat, episodic=False)
                save_results(results=obs, folder='Obs', dir=dircty, stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=N_PARTICLE, repeat=repeat)
                save_results(results=smc.epsilon_all_history, folder='Epsilon', dir=dircty, stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=N_PARTICLE, repeat=repeat)
                # save_results(results=smc, folder='SMC', dir=dircty, stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=N_PARTICLE, repeat=repeat)
            if len(obs._buffers['state0']) == smc.params_dim and new_data_flag:
                print('============================', '\n', f'Finished exploration with {e} Episodes')
                # break
            gc.collect()
        r_all_repeat.append(r_all_epi)
        if save:
            save_results(results=r_all_repeat, folder='Returns', dir=dircty_top, stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=N_PARTICLE, episodic=False)       
            # save_results(results=samples_all_repeat, folder='Samples', dir=dircty, stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat)
            # save_results(results=obs, folder='Obs', dir=dircty, stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat)
        plot_return_vs_episodes(r_all_epi, repeat=repeat, save=save, figure_path=dircty, show=show)
        # display_smc_results(smc, save=save, figure_path=dircty, show=show)
    plot_return_vs_episodes_repeat(r_all_repeat, save=save, figure_path=dircty)
    if show:
        plt.show()
