import numpy as np
import scipy.stats as stats
import torch
from copy import deepcopy
from functools import partial
import pyro
import pyro.distributions as dist
from tqdm.notebook import tqdm
import math
import sys
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.append('/scratch/Rabbit/work/ABC_SMC_RL/')
sys.path.append('/Users/guojiaqi/work/ABC_SMC_RL/')
print('system path', sys.path)
'''module import'''
from MCMC_Algorithms.Kernels import *
from MCMC_Algorithms.Models import *
from parameter import *

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
    
class MCMC_Gibbs(MCMC):
    """the class to run Gibbs sampler for z"""
    def __init__(self, variables, kernel_functions, block_size=5, num_samples=MCMC_SAMPLE, initial_params=None, params_dim=None, warmup_steps=MCMC_T//5, disable_progbar=MCMC_SHOW_DISABLE, warmup_settings=dict(target_prob=0.7, auto_init_stepsize=True), **kwargs):
        """
        kernel_functions: dictionary {'para': para_kernel, 'z': z_kernel}
            - dictionary of a parameter kernel which conditioned on u, z, r, and a z kernel which could generate a block of z given u, and return the likelihood function
        block_size: block size of z that are being updated together
        """
        self.num_samples = num_samples
        self.kernel = kernel_functions
        assert initial_params is not None or params_dim is not None, "Should either specify initial_params or params_dim"
        if initial_params is not None:
            self.initial_params = initial_params
            self.params_dim = {var: torch.tensor(initial_param).shape for var, initial_param in zip(variables, initial_params)}
        else:
            self.initial_params = torch.zeros(params_dim)
            self.params_dim = params_dim

        assert warmup_steps is None or isinstance(warmup_steps,int) or isinstance(warmup_steps, np.integer), "warmup_steps must be None or integer"
        self.warmup_steps = 0 if warmup_steps is None else warmup_steps
        self.warmup_settings = warmup_settings  
        
        self.variables = variables      
        self.block_size = block_size
        
        self.reset_stat()
        
    def run(self):
        
        self.reset_stat()
        
        current_para, current_z = self.initial_params
        current_z_llh, current_z_llh_info_dict  = self.kernel['z'].model.llh(parameter=[current_para, current_z], llh_info_dict=dict())
        current_z_info_dict = {"logdensities":current_z_llh, "llh_info_dict":current_z_llh_info_dict}
        self.samples['para'][0] = current_para
        self.samples['z'][0] = torch.tensor(current_z)
        self.logdensities['z'][0] = current_z_llh
        self.proposed_logdensities['z'][0] = current_z_llh

        pbar = tqdm(range(self.num_samples))
        t = len(self.kernel['para'].model.data)
        for i in pbar: 
            current_para = self.kernel['para'].propose_accept(current_para=[current_para, current_z])
            for j in range(math.ceil(t // self.block_size)):
                '''proposed_block_z: block_size x M_Z x 2'''
                indices = [j * self.block_size, min((j + 1) * self.block_size, t)]
                z_accept_prob, proposed_z, proposed_z_info_dict = self.kernel['z'].propose_accept(current_para=[current_para, current_z], 
                                                            indices=range(indices[0], indices[1]), current_z_info_dict=current_z_info_dict)
                    

                if np.random.uniform(0,1) < z_accept_prob:
                    current_z = proposed_z
                    current_z_info_dict = proposed_z_info_dict
                    
                    self.accepted['z'] += 1

            pbar.set_description("Acceptance probability {}".format(np.round(self.accepted['z']/(i+1)/self.block_size, 2)))

            self.samples['para'][i+1] = current_para
            self.samples['z'][i+1] = torch.tensor(current_z)
            self.logdensities['z'][i+1] = current_z_info_dict["logdensities"]
            self.proposed_logdensities['z'][i+1] = proposed_z_info_dict["logdensities"]
            self.accept_prob['z'][i+1] = z_accept_prob

        return self.samples['para']
    
    def warmup(self, para_key, init_para, init_para_info_dict=dict()):
        current_para, current_para_info_dict, info = self.kernel[para_key].warmup(init_para=init_para, 
                                                                        init_para_info_dict=init_para_info_dict,
                                                                        iterations=self.warmup_steps, 
                                                                        set_stepsize=True,
                                                                        **self.warmup_settings)

        return current_para, current_para_info_dict, info
    
    def reset_stat(self):
        self.samples = {var: torch.zeros(((self.num_samples+1, ) + tuple(dim))) for var, dim in self.params_dim.items()}
        self.logdensities = {var: torch.zeros(self.num_samples+1) for var in self.params_dim.keys()}
        self.proposed_logdensities = {var: torch.zeros(self.num_samples+1) for var in self.params_dim.keys()}
        self.accepted = {var: 0 for var in self.params_dim.keys()}
        self.accept_prob = {var: torch.zeros(self.num_samples+1) for var in self.params_dim.keys()}   

    
#MCMC
def MCMC_update(obs, posterior_samples, model, env):
    global STEPSIZE
    if BATCH_TRAINING:
        batch_indices = random.sample(range(min(len(obs._buffers['state0']), BUFFER_SIZE)), k=min(BATCH_SIZE, len(obs._buffers['state0']))) #TODO: what is this?
    else:
        batch_indices = slice(None)
    
    prior = IsotropicGaussianPrior(sd=PRIOR_SIGMA)
    abclikelihood = GaussianABCLikelihood(epsilon=EPSILON)
    data = torch.tensor(obs._buffers["rewards"])[-BUFFER_SIZE:][batch_indices]
    if STOCHASTIC:
        '''Stochastic'''
        z_sample = generate_z(env=env, obs=obs._buffers)
        r_hat = partial(generate_samples_with_z, model=model, obs=obs._buffers, batch_indices=batch_indices)
        z_transform_func = partial(generate_z, env=env, obs=obs._buffers)
        llh_transform_grad_fn = lambda parameter:  tabular_indicator_stochastic(para=[parameter[0].reshape(env.n_cell + (env.action_space.n, )), z_sample], model=model, obs=obs._buffers, env=env)#stochastic
        Model = StochasticSModel(prior=prior, abclikelihood=abclikelihood, data=data, z_transform_fn=z_transform_func, llh_transform_fn=r_hat, llh_transform_grad_fn=llh_transform_grad_fn)

    else:
        '''Determinisitc'''
        r_hat = partial(generate_samples, model=model, obs=obs._buffers,  batch_indices=batch_indices)
        llh_transform_grad_fn = lambda parameter:  tabular_indicator_deterministic(para=parameter.reshape(env.n_cell + (env.action_space.n, )), model=model, obs=obs._buffers)#standard form
        Model = DeterministicSRModel(prior=prior, abclikelihood=abclikelihood, data=data, llh_transform_fn=r_hat, llh_transform_grad_fn=llh_transform_grad_fn)

    def fn(parameter):
        current_logtarget_density, _ = Model.logtarget_density(parameter=parameter, llh_info_dict=dict())
        return current_logtarget_density
    if STOCHASTIC:
        kernel = HMC_Z(model=Model, stepsize=STEPSIZE, use_autograd=False)
        z_kernel = Z(model=Model)
    else:
        # hessian = torch.autograd.functional.hessian(fn, posterior_samples[0].reshape(-1)) + 1e-6 * torch.eye(len(posterior_samples[0]).reshape(-1))
        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE)
        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE, covariance_matrix=-torch.linalg.inv(hessian))
        #kernel = pCN(model=Model, stepsize=STEPSIZE)
        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=None)
        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=-torch.linalg.inv(hessian))
        #kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=False)
        #kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=False, precondition_matrix=-torch.linalg.inv(hessian))
        kernel = HMC_pyro(model=Model, stepsize=STEPSIZE, full_mass=FULL_MASS, adapt_step_size=ADAPT_STEP_SIZE, adapt_mass_matrix=ADAPT_MASS_MATRIX, target_accept_prob=TARGET_ACCEPT_PROB, num_steps=NUM_STEPS)
        # kernel = HMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=False)
        # kernel = HMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=False, precondition_matrix=-torch.linalg.inv(hessian), traj_len=None)
        # kernel = HMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=True, precondition_matrix=-torch.linalg.inv(hessian), traj_len=None)
        #kernel = mMALA(model=Model, stepsize=STEPSIZE, use_autograd=False, use_autohess=False)
        #kernel = mMALA(model=Model, stepsize=STEPSIZE, use_autograd=True, use_autohess=True)
        #kernel = mHMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=False, use_autohess=False, traj_len=None, fp_iterations=50)
    accept_probs = None
    STEPSIZE *= DECREASING_FACTOR

    if kernel.original is True:
        if STOCHASTIC:
            mcmc = MCMC_Gibbs(variables=['para', 'z'], kernel_functions={'para': kernel, 'z': z_kernel}, num_samples=training_steps, initial_params=[posterior_samples[-1].reshape(-1), z_sample], warmup_steps=np.int64(np.floor(training_steps*WARMUP_RATIO)), warup_settings=dict(target_prob=0.7, auto_init_stepsize=True))
        else:
            mcmc = MCMC(num_samples=training_steps, kernel=kernel, initial_params=posterior_samples[-1].reshape(-1), warmup_steps=np.int64(np.floor(training_steps*WARMUP_RATIO)), warup_settings=dict(target_prob=0.7, auto_init_stepsize=True))
        posterior_samples = mcmc.run().reshape((-1, ) + env.n_cell + (env.action_space.n, ))
        logdensities = mcmc.get_logdensities()
        proposed_logdensities = mcmc.get_proposed_logdensities()
        accept_probs = mcmc.get_accept_prob()
        return posterior_samples, accept_probs, logdensities, proposed_logdensities

    else:
        if STOCHASTIC:
            raise NotImplementedError('pyro model for stochastic hasn\'t been implemented' )
        else:
            mcmc = MCMC_pyro(num_samples=training_steps, kernel=kernel, initial_params=posterior_samples[-1].reshape(-1), warmup_steps=np.int64(np.floor(training_steps*WARMUP_RATIO)), disable_progbar=MCMC_SHOW_DISABLE)
        posterior_samples = mcmc.run().reshape((-1, ) + env.n_cell + (env.action_space.n, ))
        return posterior_samples, accept_probs
    
def display_results(posterior_samples, model, accept_probs):
    model.plot_policy(paras=posterior_samples.numpy(), title=f'policy_T{training_steps}_{time}', additional_info = env.R, save=save, show=show)
    # model.plot_value(paras=posterior_samples.numpy(), title=f'value_T{training_steps}_{time}')
    plt.imshow(torch.round(torch.max(torch.mean(posterior_samples, 0), -1).values, decimals=2))
    if show:
        plt.show()
    data_plot = posterior_samples.numpy().reshape(posterior_samples.shape[0], -1)[:, OBSERVE_DATA_START:OBSERVE_DATA_END]#Change the indices of names and Q_star below as well
    f = mcp.plot_chain_panel(chains=data_plot, names=env.names[OBSERVE_DATA_START:OBSERVE_DATA_END],
                                                    settings=dict(add_pm2std=True, fig=dict(figsize=(10,10), dpi=250),
                                                    mean=dict(color='y', label='mean'),
                                                    plot=dict(color='k', label='trace')))
    ax = f.get_axes()
    for i, ai in enumerate(ax):
        ai.axhline(y = Q_star.flatten()[OBSERVE_DATA_START:OBSERVE_DATA_END][i], linestyle=':', linewidth=5, color = 'g',  label = 'true q')
        q = np.percentile(data_plot[:, i], [PLOT_THRESHOLD, 100 - PLOT_THRESHOLD])
        ai.set_ylim(q)   
    f.tight_layout()
    handles, labels = ai.get_legend_handles_labels()
    ai.legend(handles, labels, bbox_to_anchor=(2, 0.2), loc='right')
    if show:
        plt.show()

    # if accept_probs is not None:
    #     fig, ax = plt.subplots(1,1,sharex=True)

    #     ax.plot(proposed_logdensities.numpy(), label="proposed samples")
    #     ax.plot(logdensities.numpy(), label="accepted samples")
    #     ax.set_xlabel("samples")
    #     ax.set_ylabel("log density")

    #     ax2 = ax.twinx()
    #     ax2.plot(accept_probs.numpy(), label="log acceptance probability", c="tab:green")
    #     ax2.set_ylabel("log acceptance probability")

    #     fig.legend()
    #     fig.tight_layout()

    #     if show:
    #         plt.show()


if __name__ == '__main__':
    from tqdm import tqdm
    import json
    from mcmcplot import mcmcplot as mcp
    from arviz import ess, plot_autocorr, plot_trace
    from sklearn.metrics import mean_squared_error
    import datetime
    import argparse
    '''module import'''
    from Environment.GridWorld import *
    from Environment.Maze import *
    from model import *
    from QLearning import *
    
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
    parser.add_argument('--Env', default=ENV_NAME)
    parser.add_argument('--sto', default=False)
    parser.add_argument('--online', default=False)
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
    env_name = args.Env
    STOCHASTIC = args.sto
    ONLINE_LEARNING = args.online
    

    N_PARTICLE = 10
    random.seed(seed)
    pyro.set_rng_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if env_name == 'GridWorld':
        env = GridWorld((3,4), obstacles=True, stochastic=STOCHASTIC)
        # env.plot_env()
    if env_name == 'Maze':
        env = Maze()
    dim = env.observation_space.n * env.action_space.n
    model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal')
    
    S = []
    if len(env.n_cell) == 1:
        S = [(i, ) for i in range(env.n_cell[0])]
    else:
        for i in range(env.n_cell[0]):
            for j in range(env.n_cell[1]):
                S.append((i,j)) 
    Q = np.ones(shape=(env.n_cell + (env.action_space.n, ))) / env.observation_space.n / env.action_space.n
    A = range(env.action_space.n)
    pi_star, Q_star, V_star = DynamicProgramming(Q, A, S, env, gamma=GAMMA, show=show)
    
    env.reset()
    results = []
    if ONLINE_LEARNING:
        r_all_iter = []
        for repeat in range(REPEAT_EXPERIMENT):
            STEPSIZE = INITIAL_STEPSIZE
            model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal')
            r_all_epi = []
            obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
            s0, _ = env.reset()
            posterior_samples = torch.tensor(model.get_parameter())
            for e in range(EPISODES):
                s0, _ = env.reset()
                para = model.sample_para()
                print(f'Episode {e} in repeat {repeat}')
                R = 0
                R_star = 0
                h = 0
                # while True: #Turn on h += 1
                for h in tqdm(range(HORIZON)):
                    action = model.act(s0, para, greedy=GREEDY)
                    s1, r, done, *info = env.step(action)
                    # if STOCHASTIC:
                    #     s1_augmented = []
                    #     for _ in range(M_Z):
                    #         s1_augmented.append(env.step(action, state=s0)[0])
                        # print(s1_augmented)
                    R += r
                    obs.insert({'state0': s0, 'state1': s1, 'action': int(action), 'rewards': r, 'done': done}, unique=UNIQUE_OBS)
                    s0 = s1
                    if done or ( h + 1)  % FROZEN_T == 0:
                        #MCMC
                        posterior_samples, accept_probs = MCMC_update(posterior_samples=posterior_samples, obs=obs, model=model, env=env)[:2]
                        model.set_parameter(posterior_samples)
                        display_results(posterior_samples=posterior_samples, model=model, accept_probs=accept_probs)

                    if done:
                        print("done with", h + 1, 'steps')
                        print('Return', R)
                        break
                    # h += 1
                r_all_epi.append(R)
                if save:
                    save_results(results=r_all_epi, folder='Returns', stochastic=STOCHASTIC, episode=e, training_steps=training_steps, greedy=GREEDY, epsilon=EPSILON, initial_stepsize=INITIAL_STEPSIZE, decreasing_factor=DECREASING_FACTOR, time=time, m_z=M_Z, repeat=repeat, episodic=True)
            r_all_iter.append(r_all_epi)
            if save:
                save_results(results=r_all_iter, folder='Returns', stochastic=STOCHASTIC, episode=e, training_steps=training_steps, greedy=GREEDY, epsilon=EPSILON, initial_stepsize=INITIAL_STEPSIZE, decreasing_factor=DECREASING_FACTOR, time=time, m_z=M_Z, repeat=repeat)        
        plt.plot(R)
        if show:
            plt.show()
            
    else:
        file_path = f"../Results/E1/T{training_steps}_Sto{STOCHASTIC}_{time}.json"
        error_all = []
        epsilon_lst = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4]
        data_percentage_lst = np.linspace(0.1, 1, 10)
        for EPSILON in epsilon_lst:
            error_all_percentage = []
            for data_percentage in data_percentage_lst:
                env.uniform_policy(data_percentage=data_percentage)
                obs = env.uniform_obs
                posterior_samples = model.get_parameter()
                posterior_samples, accept_probs = MCMC_update(posterior_samples=posterior_samples, obs=obs, model=model, env=env)[:2]
                model.set_parameter(posterior_samples)
                display_results(posterior_samples=posterior_samples, model=model, accept_probs=accept_probs)
                error_all_percentage.append(mean_squared_error(Q_star.flatten(), posterior_samples[-1].flatten()))
                print(error_all_percentage)
            error_all.append(error_all_percentage)
            experiment_info = {
                'args': vars(args), 
                'epsilon_list': epsilon_lst, 
                'data_percentage': data_percentage_lst.tolist(),
                'errors': error_all
            }
            json_data = json.dumps(experiment_info)
            with open(file_path, "w") as file:
                file.write(json_data)
            print(f"Dictionary saved to {file_path}")