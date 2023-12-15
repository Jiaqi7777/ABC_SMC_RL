import numpy as np
from scipy.optimize import bisect
import sys
import os
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
    def __init__(self, model, Model=None, min_ess=0.5, num_samples=None, initial_params=None, params_dim=None):
        assert initial_params is not None or params_dim is not None, "Should either specify initial_params or params_dim"
        if initial_params is not None:
            self.initial_params = initial_params
            self.n_particle = len(initial_params)
        else:
            self.initial_params = model.get_learnable_parameter()
            self.n_particle = params_dim
        self.min_ess = min_ess 
        self.model = model
        self._weights = model._weights
        self.num_samples = num_samples
        self.SMC_Model = Model
        self.reset_stat()

    def set_weights(self, weights):
        self._weights = weights
        self.model.set_weights(weights)

    def update(self, alpha, smc_samples):
        epsilon = self.SMC_Model.abclikelihood.epsilon
        epsilon_0 = self.find_epsilon_0(alpha=alpha, smc_samples=smc_samples, generate_new_weights_fn=generate_new_weights_0, epsilon_0=epsilon, a=epsilon, b=epsilon*10)
        self.SMC_Model.new_epsilon = epsilon_0
        self.epsilon_history.append(epsilon_0)
        # epsilon_0 = epsilon * 2
        print('epsilon_0 ==========', '\n', epsilon_0)
        weights = generate_new_weights_0(epsilon=epsilon_0, weights=self._weights, Model=Model, smc_samples=smc_samples, epsilon_0=epsilon_0)
        # print('weights 0==========', '\n', weights)
        self.set_weights(weights)
        self.update_history(smc_samples=model.get_parameter(), weights=weights)
        if self.ESS(self._weights) < self.min_ess * self.n_particle:
            print('Resampled')
            self.SMC_Model.new_epsilon = epsilon_0
            smc_samples, weights = self.resample()
            self.update_history(smc_samples=model.get_parameter(), weights=weights)
        pre_epsilon_0 = epsilon_0
        while epsilon_0 > epsilon:
            print('Solving...')
            epsilon_0 = max(epsilon, self.find_epsilon_0(alpha=alpha, smc_samples=smc_samples, generate_new_weights_fn=generate_new_weights, epsilon_0=epsilon_0, a=epsilon_0*0.002, b=pre_epsilon_0))
            # epsilon_0 *= 0.9
            self.epsilon_history.append(epsilon_0)
            weights = generate_new_weights(epsilon=epsilon_0, weights=self._weights, Model=Model, smc_samples=smc_samples, epsilon_0=pre_epsilon_0)
            print('epsilon_0 ==========', '\n', epsilon_0)
            self.set_weights(weights)
            self.update_history(smc_samples=model.get_parameter(), weights=weights)
            if self.ESS(self._weights) < self.min_ess * self.n_particle:
                print('Resampled')
                self.SMC_Model.new_epsilon = epsilon_0
                smc_samples, weights = self.resample()
                self.update_history(smc_samples=model.get_parameter(), weights=weights)
            for j in range(self.n_particle):
                posterior_samples, accept_probs, mcmc, kernel, logdensities, proposed_logdensities, = MCMC_update(Model=self.SMC_Model, posterior_samples=[smc_samples[j]], env=env, training_steps_with_burnin=training_steps_with_burnin, training_steps=training_steps, USE_AUTOGRAD=USE_AUTOGRAD, WARMUP_RATIO=WARMUP_RATIO, MCMC_SHOW_DISABLE=MCMC_SHOW_DISABLE, ADAPT_STEP_SIZE=ADAPT_STEP_SIZE, ADAPT_MASS_MATRIX=ADAPT_MASS_MATRIX)
                # print(posterior_samples.shape)
                smc_samples[j] = posterior_samples[-1]
            self.model.set_learnable_parameter(torch.tensor(smc_samples))
            pre_epsilon_0 = epsilon_0
            
            
        # for j in range(self.n_particle):
        #     para = self.model.get_learnable_parameter()[j]
        #     lld = self.Model.llh(parameter=para, llh_info_dict=dict())
        #     # lld = stats.multivariate_normal.logpdf(obs['rewards'][-update_frequency:], samples_l[-update_frequency:,j])
        #     self._weights[j] *= lld
            
        # self._weights /= sum(self._weights)
        # self.model.set_weights(self._weights)
        
    @staticmethod
    def ESS(weights):
        return 1 / torch.sum(weights**2)
        # return 1 / ( 1 + torch.var(self._weights) )

    def resample(self):
        idx = random.choices(range(self.n_particle), self._weights, k=self.n_particle)
        paras = self.model.get_learnable_parameter()[idx]
        
        weights = torch.ones(self.n_particle) / self.n_particle
        self.set_weights(weights)
        return paras, weights
        
    def reset_stat(self):
        self.samples = torch.tensor([])
        self._weights_history = torch.tensor([])
        self.ess_history = torch.tensor([])
        self.epsilon_history = []
        
    def update_history(self, smc_samples=None, weights=None):
        print('Update samples')
        if smc_samples is not None:
            self.samples = torch.cat((self.samples, smc_samples.unsqueeze(0)))
        if weights is not None:
            self._weights_history = torch.cat((self._weights_history, weights.unsqueeze(0)))
            self.ess_history = torch.cat((self.ess_history, self.ESS(weights).unsqueeze(0)))
        # print(self.samples.shape)

    def find_epsilon_0(self, alpha, smc_samples, generate_new_weights_fn, epsilon_0, a, b):
        # Use optimization to find epsilon_0 that satisfies the ESS condition
        try:
            result = bisect(ESS_Matching, a=a, b=b, xtol=1e-3, maxiter=2000, args=(alpha, self.SMC_Model, self._weights, smc_samples, generate_new_weights_fn, epsilon_0))
        except:
            return self.find_epsilon_0(alpha, smc_samples, generate_new_weights_fn, epsilon_0, a, b*2)
        return result
        # if result.success:
        #     return result.x[0]
        # else:
        #     raise ValueError("Optimization did not converge. Check input parameters.")

def generate_new_weights_0(epsilon, weights, Model, smc_samples, epsilon_0):
    llh=[Model.llh_new(parameter=p, epsilon=epsilon)[0] for p in smc_samples]
    new_weights = torch.tensor([torch.log(w) + l for w, l in zip(weights, llh)])
    new_weights = torch.exp(new_weights)
    return new_weights / torch.sum(new_weights)

def generate_new_weights(epsilon, weights, Model, smc_samples, epsilon_0):
    new_weights = torch.tensor([w * torch.exp(Model.llh_new(parameter=p, epsilon=epsilon)[0] - Model.llh_new(parameter=p, epsilon=epsilon_0)[0]) for w, p in zip(weights, smc_samples)])
    return new_weights / torch.sum(new_weights)

def ESS_Matching(epsilon, alpha, Model, weights, smc_samples, generate_new_weights_fn, epsilon_0):
    new_weights = generate_new_weights_fn(epsilon=epsilon, weights=weights, Model=Model, smc_samples=smc_samples, epsilon_0=epsilon_0)
    ess = SMC.ESS(new_weights)
    target_ess = alpha * SMC.ESS(weights)
    # print(ess, target_ess)
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

# def SMC_update(smc, Model, obs, model, alpha, smc_samples):
#     smc.update(obs, alpha, smc_samples)
    # epsilon_0 = smc.find_epsilon_0(Model=Model, alpha=alpha, smc_samples=smc_samples, generate_new_weights_fn=generate_new_weights_0)
    # print('epsilon 0==========', '\n', epsilon_0)
    # weights = generate_new_weights_0(epsilon_0=epsilon_0, weights=smc._weights, Model=Model, smc_samples=smc_samples)
    
    # smc.set_weights(weights)
    # while epsilon_0 > epsilon:
    #     epsilon_0 = max(epsilon, smc.find_epsilon_0(Model=Model, alpha=alpha, smc_samples=smc_samples, generate_new_weights_fn=generate_new_weights))
    #     print('epsilon 0==========', '\n', epsilon_0)
    #     weights = generate_new_weights(epsilon_0=epsilon_0, weights=smc._weights, Model=Model, smc_samples=smc_samples)
    #     smc.set_weights(weights)
    # return smc

def display_smc_results(smc):
    dim = smc.samples.shape
    samples = smc.samples
    fig,ax = plt.subplots(5, 5,figsize=(20,20))
    for i in range(dim[2] - 1):
        for k in range(i+1):
            for j in range(dim[1]):
                ax[i,k].scatter(range(dim[0]), samples[:, j, i, k, 0], label=f"right {j}", alpha=smc._weights_history[:,j])
                ax[i,k].plot(range(dim[0]), samples[:, j, i, k, 0], linestyle='--', alpha=0.6)
                # ax[i,k].scatter(range(dim[0]), samples[:, j, i, k, 1], label=f"left {j}", alpha=smc._weights_history[:,j])
                # ax[i,k].plot(range(dim[0]), samples[:, j, i, k, 1], linestyle='-')
                # ax[i,k].hlines(Q_star[i, k, 0], xmin=0, xmax=len(samples), linestyle="--", label="true right",color="green")
                ax[i,k].hlines(Q_star[i, k, 1], xmin=0, xmax=len(samples), linestyle="--", label="true left", color="purple")
                ax[i,k].set_title([i, k])
            # ax[i,k].legend()
    plt.show()

    fig2,ax2 = plt.subplots(5, 5,figsize=(20,20))
    for i in range(dim[2] - 1):
        for k in range(i+1):
            for j in range(dim[1]):
                # ax2[i,k].scatter(range(dim[0]), samples[:, j, i, k, 0], label=f"right {j}", alpha=smc._weights_history[:,j])
                # ax2[i,k].plot(range(dim[0]), samples[:, j, i, k, 0], linestyle='--')
                ax2[i,k].scatter(range(dim[0]), samples[:, j, i, k, 1], label=f"left {j}", alpha=smc._weights_history[:,j])
                ax2[i,k].plot(range(dim[0]), samples[:, j, i, k, 1], linestyle='-', alpha=0.6)
                # ax2[i,k].hlines(Q_star[i, k, 0], xmin=0, xmax=len(samples), linestyle="--", label="true right",color="green")
                ax2[i,k].hlines(Q_star[i, k, 1], xmin=0, xmax=len(samples), linestyle="--", label="true left", color="purple")
                ax2[i,k].set_title([i, k])
            # ax2[i,k].legend()
    plt.show()
    
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

    N_PARTICLE = 10
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
        env = DeepSea(depth=6)
    
    S = []
    if len(env.n_cell) == 1:
        S = [(i, ) for i in range(env.n_cell[0])]
    else:
        if env_name == 'DeepSea':
            for i in range(env.n_cell[0] - 1):
                for j in range(env.n_cell[1]):
                    S.append((i,j)) 
        else:
            for i in range(env.n_cell[0]):
                for j in range(env.n_cell[1]):
                    S.append((i,j)) 
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
    for repeat in range(REPEAT_EXPERIMENT):
        epsilon = abc_epsilon
        STEPSIZE = INITIAL_STEPSIZE
        # model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal', gamma=GAMMA)
        model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal', mean=PRIOR_MEAN, std=PRIOR_SIGMA, gamma=GAMMA, initial_tables=Q_star, idx=FROZEN_IDX)#For frozen all but one dimensions
        smc = SMC(model=model, initial_params=model.get_parameter())
        smc.update_history(model.get_parameter(), model._weights)
        r_all_epi = [] 
        samples_all_ep = []
        obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
        s0, _ = env.reset()
        for e in range(EPISODES):
            s0, _ = env.reset()
            para = model.sample_para()
            plot_qtable(para, title=f'Sampled Q table for episode {e}')
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
                    # model.set_learnable_idx(obs)
                    smc_samples = torch.tensor(model.get_learnable_parameter())
                    plot_obs(obs, env, env_name=ENV_NAME, title=f'Eploration path till ep {e}', additional_info=V_star)
                    plt.show()
                    #MCMC
                    if TRANSFORM:
                        smc_samples = TruncatedGaussianABCLikelihood.log_neg_transform(smc_samples)
                    # posterior_samples, accept_probs = MCMC_update(posterior_samples=smc_samples, obs=obs, model=model, env=env)[:2]
                    # mode_idx = torch.argmax(mcmc.logdensities) if 'mcmc' in vars() else None
                    if new_data_flag and e > 0:
                        Model = get_SMC_Model(obs=obs, model=model, env=env, epsilon=epsilon)
                        smc.SMC_Model = Model
                        smc.update(alpha, smc_samples)
                    if TRANSFORM:
                        smc_samples = TruncatedGaussianABCLikelihood.neg_exp_transform(smc_samples)
                    # model.sample_random_tables(set=True)
                    obs.init_new_data_buffer()
                if done:
                    print("done with", h + 1, 'steps')
                    print('Return', R)
                    break
                # h += 1
            epsilon *= 0.95
            r_all_epi.append(R)
            samples_all_ep.append(model.get_parameter().clone().detach())
            explore_pct = [model.get_parameter().numpy()[:, i, i, 0] > model.get_parameter().numpy()[:, i, i, 1] for i in range(env.n_cell[0] - 1)]
            explore_pct_all = np.sum([explore_pct[0], explore_pct[0] & explore_pct[1],  explore_pct[0] & explore_pct[1] & explore_pct[2],   explore_pct[0] & explore_pct[1] & explore_pct[2] & explore_pct[3],  explore_pct[0] & explore_pct[1] & explore_pct[2] & explore_pct[3] & explore_pct[4]], axis=-1)/len(smc_samples) 
            print('explore percentage', explore_pct_all)
            if save:
                save_results(results=r_all_epi, folder='Returns', stochastic=STOCHASTIC, episode=e, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat, episodic=False)
                save_results(results=samples_all_ep, folder='Samples', stochastic=STOCHASTIC, episode=e, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat, episodic=False)
                save_results(results=obs, folder='Obs', stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat)
        r_all_repeat.append(r_all_epi)
        samples_all_repeat.append(samples_all_ep)
        if save:
            save_results(results=r_all_repeat, folder='Returns', stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat)        
            save_results(results=samples_all_repeat, folder='Samples', stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat)
            save_results(results=obs, folder='Obs', stochastic=STOCHASTIC, training_steps=training_steps, greedy=GREEDY, epsilon=epsilon, time=time, m_z=M_Z, repeat=repeat)
        plot_return_vs_episodes(r_all_epi, repeat=repeat)
    plot_return_vs_episodes_repeat(r_all_repeat)
    display_smc_results(smc)
    if show:
        plt.show()
