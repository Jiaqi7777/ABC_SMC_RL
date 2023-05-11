import numpy as np
import scipy.stats as stats
import torch
from copy import deepcopy
from functools import partial
import pyro
import pyro.distributions as dist
from tqdm.notebook import tqdm
import sys
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.append('/scratch/Rabbit/work/ABC_SMC_RL/')
print(sys.path)
'''module import'''
from MCMC_Algorithms.Kernels import *
from MCMC_Algorithms.Models import *
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
    from tqdm import tqdm
    from mcmcplot import mcmcplot as mcp
    from arviz import ess, plot_autocorr, plot_trace
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

    N_PARTICLE = 10
    random.seed(seed)
    pyro.set_rng_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if env_name == 'GridWorld':
        env = GridWorld((3,4), obstacles=True)
        env.plot_env()
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
    Q = np.ones(shape=(env.n_cell + (env.action_space.n, )))/env.observation_space.n / env.action_space.n
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
                    R += r
                    obs.insert({'state0': s0, 'state1': s1, 'action': int(action), 'rewards': r, 'done': done}, unique=UNIQUE_OBS)
                    s0 = s1
                    if done or ( h + 1)  % FROZEN_T == 0:
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
                        hessian = torch.autograd.functional.hessian(fn, posterior_samples[0].reshape(-1))
                        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE)
                        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE, covariance_matrix=-torch.linalg.inv(hessian))
                        #kernel = pCN(model=Model, stepsize=STEPSIZE)
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=None)
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=-torch.linalg.inv(hessian))
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=False)
                        #kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=False, precondition_matrix=-torch.linalg.inv(hessian))
                        #kernel = HMC_pyro(model=Model, stepsize=STEPSIZE, full_mass=FULL_MASS, adapt_step_size=ADAPT_STEP_SIZE, adapt_mass_matrix=ADAPT_MASS_MATRIX, target_accept_prob=TARGET_ACCEPT_PROB, num_steps=NUM_STEPS)
                        #kernel = HMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=False)
                        kernel = HMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=False, precondition_matrix=-torch.linalg.inv(hessian), traj_len=None)
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
                        print('Return', R)
                        break
                    # h += 1
                r_all_epi.append(R)
                with open(f'Returns/MCMC/Episode{e}T{training_steps}_Gdy{GREEDY}_Ep{EPSILON}_Stp{INITIAL_STEPSIZE}_Dcrs{DECREASING_FACTOR}_{time}.npy', 'wb') as f:
                    np.save(f, r_all_epi)
                    print(f'EPISODES return for repeat {repeat} saved at', f'Returns/MCMC/Episode{e}T{training_steps}_Gdy{GREEDY}_Ep{EPSILON}_Stp{INITIAL_STEPSIZE}_Dcrs{DECREASING_FACTOR}_{time}.npy')
            r_all_iter.append(r_all_epi)
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