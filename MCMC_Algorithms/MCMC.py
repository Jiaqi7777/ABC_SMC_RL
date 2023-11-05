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
from utils import *

import torch.multiprocessing as mp
import queue
import time

class MCMC_Process:
    def __init__(self, kernel, iterable_fn, event, chain_num, sample_queue, initial_params, warmup_steps, warmup_settings, num_samples, comm_interval=1):
        self.kernel = kernel
        self.iterable_fn = iterable_fn
        self.event = event
        self.chain_num = chain_num
        self.sample_queue = sample_queue
        self.initial_params = initial_params
        self.warmup_steps = warmup_steps
        self.warmup_settings = warmup_settings
        self.num_samples = num_samples
        self.comm_interval = comm_interval if comm_interval is not None else 1

    def run(self):
        counter = 0
        temp_list = []
        try: 
            for sample in self.iterable_fn():

                counter += 1
                temp_list.append((self.chain_num, sample))
                if counter == self.comm_interval:
                    self.sample_queue.put_nowait(temp_list)
                    self.event.wait()
                    self.event.clear()

                    temp_list = []
                    counter = 0

            if len(temp_list) > 0:
                self.sample_queue.put_nowait(temp_list)
                self.event.wait()
                self.event.clear()

        except Exception as e:
            self.sample_queue.put_nowait([(self.chain_num, e)])


def warmup(kernel, warmup_steps, warmup_settings, init_para, init_para_info_dict=dict()):
    current_para, current_para_info_dict, info = kernel.warmup(init_para=init_para, 
                                                                    init_para_info_dict=init_para_info_dict,
                                                                    iterations=warmup_steps, 
                                                                    set_stepsize=True,
                                                                    **warmup_settings)

    return current_para, current_para_info_dict, info

def generate_samples_iterable(kernel, initial_params, warmup_steps, warmup_settings, num_samples, idx=None, use_progress_bar=True):

    accepted = 0
    current_para = initial_params
    current_logtarget_density, current_para_llh_info_dict  = kernel.model.logtarget_density(parameter=current_para, llh_info_dict=dict())
    current_para_info_dict = {"logdensities":current_logtarget_density, "llh_info_dict":current_para_llh_info_dict}
    if warmup_steps > 0:
        try: 
            current_para, current_para_info_dict, _ = warmup(
                kernel=kernel,
                warmup_steps=warmup_steps,
                warmup_settings=warmup_settings,
                init_para=current_para,
                init_para_info_dict=current_para_info_dict)
    
        except NotImplementedError:
            print("Warmup is not implemented for the current kernel. Skip to sampling...")

    pbar = tqdm(range(num_samples)) if use_progress_bar is True else range(num_samples)
    for i in pbar:

        accept_prob, proposed_para, proposed_para_info_dict  = kernel.propose_accept(current_para=current_para,
                                                                        current_para_info_dict=current_para_info_dict, indices=idx)

        ifaccept = 0
        if np.random.uniform(0, 1) < accept_prob:
            current_para = proposed_para
            current_para_info_dict = proposed_para_info_dict
            accepted += 1
            ifaccept = 1

        if use_progress_bar:
            pbar.set_description("Acceptance probability {}".format(np.round(accepted/(i+1), 2)))

        yield current_para, current_para_info_dict["logdensities"], proposed_para_info_dict["logdensities"], accept_prob, ifaccept
    
    yield None #indicate finish

def generate_samples_gibbs_iterable(kernel, block_size, data_length, variables, initial_params, warmup_steps, warmup_settings, num_samples, idx=None, use_progress_bar=True):
    
    current_para = initial_params
    current_para_info_dict = dict()

    for var in variables:
        kernel[var].model.set_var(var)
        kernel[var].model.set_samples(current_para)
        current_logtarget_density, current_para_llh_info_dict  = kernel[var].model.logtarget_density(parameter=current_para[var], llh_info_dict=dict())
        current_para_info_dict = {"logdensities":current_logtarget_density, "llh_info_dict":current_para_llh_info_dict}

        if warmup_steps > 0:
            try:
                current_para[var], current_para_info_dict, _ = warmup(kernel=kernel[var],
                                                                           warmup_steps=warmup_steps,
                                                                           warmup_settings=warmup_settings,
                                                                           init_para=current_para[var],
                                                                           init_para_info_dict=current_para_info_dict)
            except NotImplementedError:
                print("Warmup is not implemented for the current kernel. Skip to sampling...")

    pbar = tqdm(range(num_samples), position=0, leave=True) if use_progress_bar is True else range(num_samples)
    accepted = {var: 0 for var in variables}

    for i in pbar:

        ifaccept = {var: np.zeros([math.ceil(data_length[var] / block_size[var])]) for var in variables}
        accept_probs = {var: np.zeros(math.ceil(data_length[var] / block_size[var])) for var in variables}
        current_logdensities = {var: 0. for var in variables}
        proposed_logdensities = {var: 0. for var in variables}

        for var in variables:
            kernel[var].model.set_var(var)
            current_logtarget_density, current_para_llh_info_dict  = kernel[var].model.logtarget_density(parameter=current_para[var], llh_info_dict=dict())
            current_para_info_dict = {"logdensities":current_logtarget_density, "llh_info_dict":current_para_llh_info_dict}
            for j in range(math.ceil(data_length[var] / block_size[var])):
                '''proposed_block_z: block_size x M_Z x 2'''
                selected_indices = idx[var] if idx[var] is not None else slice(None)
                indices = range(j * block_size[var], min((j + 1) * block_size[var], data_length[var]))[selected_indices]
                accept_prob, proposed_para, proposed_para_info_dict = kernel[var].propose_accept(current_para=current_para[var], 
                                                            indices=indices, current_para_info_dict=current_para_info_dict)
                # if j==4 and var=='z':
                #     print(current_para[var], proposed_para)
                    # print('accept prob', j, accept_prob)

                if np.random.uniform(0, 1) < accept_prob:
                    current_para[var] = proposed_para
                    current_para_info_dict = proposed_para_info_dict
                    ifaccept[var][j] = 1
                    accepted[var] += 1
                
                accept_probs[var][j] = accept_prob
                kernel[var].model.set_samples(current_para)

            current_logdensities[var] = current_para_info_dict["logdensities"]
            proposed_logdensities[var] = proposed_para_info_dict["logdensities"]
            

        if use_progress_bar:
            pbar.set_description("Acceptance probability {}".format({var: np.round(accepted[var]/((i+1)*math.ceil(data_length[var] / block_size[var])), 2) for var in variables}))

        yield current_para, current_logdensities, proposed_logdensities, accept_probs, ifaccept
        
    yield None # indicate finish

class MCMC:
    """the class to run MCMC
    parallel computing follows a similar structure to https://docs.pyro.ai/en/stable/_modules/pyro/infer/mcmc/api.html#MCMC
    """
    def __init__(self, kernel, warmup_steps=0, num_samples=MCMC_SAMPLE, initial_params=None, num_chains=1, params_dim=None, warmup_settings=dict(target_prob=0.7, auto_init_stepsize=True), mp_settings=dict(comm_interval=1, mp_context="spawn"), **kwargs):
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
        self.num_chains = num_chains
        self.mp_settings = mp_settings

        #check cpu numbers
        if num_chains > 1:

            num_cpus = max(mp.cpu_count()-1, 1)
            if num_chains <= num_cpus:
                self.parallel = True
            else:
                raise AssertionError("num_chains > num of cpus - 1, aborted")

        else:
            self.parallel = False
        self.check_initial_params_and_dims(initial_params=initial_params, params_dim=params_dim)

        assert warmup_steps is None or isinstance(warmup_steps,int) or isinstance(warmup_steps, np.integer), "warmup_steps must be None or integer"
        self.warmup_steps = 0 if warmup_steps is None else warmup_steps
        self.warmup_settings = warmup_settings

        self.reset_stat()

    def check_initial_params_and_dims(self, initial_params, params_dim):
        assert initial_params is not None or params_dim is not None, "Should either specify initial_params or params_dim"
        if initial_params is not None:
            if self.parallel:
                assert initial_params.shape[0] == self.num_chains, "Leading dimension of initial_params should match num_chains for parallel computing"
                self.initial_params = initial_params
                self.params_dim = len(initial_params[0])
            else:
                assert (initial_params.shape[0] == 1 and len(initial_params.shape) == 2) or len(initial_params.shape) == 1, "Leading dimension of initial_params must be either 1 or params_dim for sequential computing"
                self.initial_params = initial_params.reshape(-1)
                self.params_dim = len(self.initial_params)
        else:
            if self.parallel:
                self.initial_params = torch.zeros([num_chains,params_dim])
            else:
                self.initial_params = torch.zeros(params_dim)
            self.params_dim = params_dim


    def run(self, idx=None):
        self.reset_stat()

        if self.parallel:
            samples = self._run_parallel(idx=idx)
        else:
            samples = self._run_sequential(idx=idx)
        return samples

    def _run_sequential(self, idx):
        iterable = self.get_iterable_fn(idx=idx, initial_params=self.initial_params, use_progress_bar=True)()
        for i, sample in enumerate(iterable):
            if sample is not None:
                self.unpack_sample_sequential(sample=sample, pos=i)

        return self.samples

    def unpack_sample_sequential(self, sample, pos):
        current_parameter, current_logtarget_density, proposed_logtarget_density, accept_prob, ifaccept = sample
        self.samples[pos] = current_parameter
        self.logdensities[pos] = current_logtarget_density
        self.proposed_logdensities[pos] = proposed_logtarget_density
        self.accept_prob[pos] = accept_prob
        self.ifaccept[pos] = ifaccept

    def _run_parallel(self, idx):
        self.init_multiprocesses(idx=idx)
        active_processes = self.num_chains

        chain_counts = [0 for _ in range(self.num_chains)]

        try:
            for p in self.processes:
                p.start()
            progress_bars = [tqdm(total=self.num_samples, position=0, leave=True) for _ in range(self.num_chains)]

            while active_processes > 0:
                try: 
                    items = self.sample_queue.get(timeout=5)
                except queue.Empty:
                    continue

                for (chain_num, sample) in items:
                    if isinstance(sample, Exception):
                        raise sample

                    if sample is not None:
                        self.events[chain_num].set()

                        pos = chain_counts[chain_num]
                        self.unpack_sample_parallel(sample=sample, chain_num=chain_num, pos=pos)

                        chain_counts[chain_num] += 1
                        progress_bars[chain_num].update(1)
                        # progress_bars[chain_num].set_description("Chain ID: {}, Acceptance probability {}".format(chain_num, np.round(self.ifaccept[chain_num].sum()/(chain_counts[chain_num]+1), 2)))
                        progress_bars[chain_num].set_description("Chain ID: {}, Acceptance probability {}".format(chain_num, self.get_acceptance_prob_stat(num_iter=chain_counts[chain_num], chain_num=chain_num)))

                    else:
                        active_processes -= 1

                del items
        finally:
            self.terminate_multiprocesses()

        return self.samples
    
    def unpack_sample_parallel(self, sample, chain_num, pos):
        current_parameter, current_logtarget_density, proposed_logtarget_density, accept_prob, ifaccept = sample
        self.samples[chain_num, pos] = current_parameter.clone()
        self.logdensities[chain_num, pos] = current_logtarget_density.clone()
        self.proposed_logdensities[chain_num, pos] = proposed_logtarget_density.clone()
        self.accept_prob[chain_num, pos] = accept_prob
        self.ifaccept[chain_num, pos] = ifaccept

    def get_acceptance_prob_stat(self, num_iter=None, chain_num=0):
        num_iter = self.num_samples if num_iter is None else num_iter
        if self.parallel:
            return np.round(self.ifaccept[chain_num].sum()/num_iter, 2)
        else:
            return np.round(self.ifaccept.sum()/num_iter, 2)

    def init_multiprocesses(self, idx):
        self.context = mp.get_context(self.mp_settings.get("mp_context"))
        self.events = [self.context.Event() for _ in range(self.num_chains)]
        self.sample_queue = self.context.Queue()

        self.processes = []
        for chain_num in range(self.num_chains):
            initial_params = self.initial_params[chain_num]

            iterable_fn = self.get_iterable_fn(idx=idx, initial_params=initial_params, use_progress_bar=False)
            process = MCMC_Process(kernel=self.kernel, iterable_fn=iterable_fn, event=self.events[chain_num], chain_num=chain_num, sample_queue=self.sample_queue, initial_params=initial_params, warmup_steps=self.warmup_steps, warmup_settings=self.warmup_settings, num_samples=self.num_samples, comm_interval=self.mp_settings.get("comm_interval"))

            self.processes.append(
                self.context.Process(name=str(chain_num), target=process.run))

    def terminate_multiprocesses(self):
        for p in self.processes:
            if p.is_alive():
                p.terminate()

    def get_iterable_fn(self, idx, initial_params, use_progress_bar):
        iterable = partial(generate_samples_iterable,kernel=self.kernel,
                                         initial_params=initial_params,
                                         warmup_steps=self.warmup_steps,
                                         warmup_settings=self.warmup_settings,
                                         num_samples=self.num_samples,
                                         idx=idx,
                                         use_progress_bar=use_progress_bar)
        return iterable

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
        if self.parallel:
            self.samples = torch.zeros((self.num_chains, self.num_samples, self.params_dim))
            self.logdensities = torch.zeros((self.num_chains, self.num_samples))
            self.proposed_logdensities = torch.zeros((self.num_chains, self.num_samples))
            self.accept_prob = np.zeros((self.num_chains, self.num_samples))
            self.ifaccept = np.zeros((self.num_chains, self.num_samples))
        else:
            self.samples = torch.zeros((self.num_samples, self.params_dim))
            self.logdensities = torch.zeros(self.num_samples)
            self.proposed_logdensities = torch.zeros(self.num_samples)
            self.accept_prob = np.zeros(self.num_samples)
            self.ifaccept = np.zeros(self.num_samples)


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
    def __init__(self, variables, kernel_functions, block_size={'para': 10000, 'z':1}, num_samples=MCMC_SAMPLE, initial_params=None, num_chains=1, params_dim=None, warmup_steps=MCMC_T//5, warmup_settings=dict(target_prob=0.7, auto_init_stepsize=True), mp_settings=dict(comm_interval=1, mp_context="spawn"), **kwargs):
        """
        kernel_functions: dictionary {'para': para_kernel, 'z': z_kernel}
            - dictionary of a parameter kernel which conditioned on u, z, r, and a z kernel which could generate a block of z given u, and return the likelihood function
        block_size: block size of z that are being updated together
        """
        self.variables = variables      
        self.block_size = block_size
        self.data_length = {var: len(kernel_functions[var].model.data) for var in self.variables}

        super(MCMC_Gibbs, self).__init__(kernel=kernel_functions, warmup_steps=warmup_steps, num_samples=num_samples, initial_params=initial_params, num_chains=num_chains, params_dim=params_dim, warmup_settings=warmup_settings, mp_settings=mp_settings, **kwargs) 

    def check_initial_params_and_dims(self, initial_params, params_dim): #TODO: not ideal

        if initial_params is not None:
            if self.parallel:
                assert len(initial_params) == self.num_chains, "len(initial_params) should match num_chains for parallel computing"
                self.initial_params = initial_params
                self.params_dim = {var: initial_params[0][var].shape for var in self.variables}
            else:
                self.initial_params = initial_params[0] if (isinstance(initial_params,list) and len(initial_params) == 1) else initial_params
                self.params_dim = {var: self.initial_params[var].shape for var in self.variables}
        else:
            raise NotImplementedError
    
    def unpack_sample_sequential(self, sample, pos):
        current_parameter, current_logtarget_density, proposed_logtarget_density, accept_prob, ifaccept = sample
        for var in self.variables:
            self.samples[var][pos] = torch.tensor(current_parameter[var]) if type(current_parameter[var]) == np.ndarray else current_parameter[var]
            self.logdensities[var][pos] = current_logtarget_density[var]
            self.proposed_logdensities[var][pos] = proposed_logtarget_density[var]
            self.accept_prob[var][pos] = accept_prob[var]
            self.ifaccept[var][pos] = ifaccept[var]
        
    def unpack_sample_parallel(self, sample, chain_num, pos):
        current_parameter, current_logtarget_density, proposed_logtarget_density, accept_prob, ifaccept = sample
        for var in self.variables:
            self.samples[var][chain_num, pos] = torch.tensor(current_parameter[var]) if type(current_parameter[var]) == np.ndarray else current_parameter[var].clone()
            self.logdensities[var][chain_num, pos] = current_logtarget_density[var].clone()
            self.proposed_logdensities[var][chain_num, pos] = proposed_logtarget_density[var].clone()
            self.accept_prob[var][chain_num, pos] = accept_prob[var]
            self.ifaccept[var][chain_num, pos] = ifaccept[var]
    
    def get_acceptance_prob_stat(self, num_iter=None, chain_num=0):
        num_iter = self.num_samples if num_iter is None else num_iter
        if self.parallel:
            return {var: np.round(self.ifaccept[var][chain_num].sum()/(self.ifaccept[var].shape[-1]*num_iter), 2) for var in self.variables}
        else:
            return {var: np.round(self.ifaccept[var].sum()/(self.ifaccept[var].shape[-1]*num_iter), 2) for var in self.variables}

    def get_iterable_fn(self, idx, initial_params, use_progress_bar):
        iterable = partial(generate_samples_gibbs_iterable, kernel=self.kernel,
                                               block_size=self.block_size,
                                               data_length=self.data_length,
                                               variables=self.variables,
                                               initial_params=initial_params,
                                               warmup_steps=self.warmup_steps,
                                               warmup_settings=self.warmup_settings,
                                               num_samples=self.num_samples,
                                               idx=idx,
                                               use_progress_bar=use_progress_bar)
        return iterable
    
    def reset_stat(self):
        if self.parallel:
            self.samples = {var: torch.zeros(((self.num_chains, self.num_samples, ) + tuple(dim))) for var, dim in self.params_dim.items()}
            self.logdensities = {var: torch.zeros((self.num_chains, self.num_samples)) for var in self.params_dim.keys()}
            self.proposed_logdensities = {var: torch.zeros((self.num_chains, self.num_samples)) for var in self.params_dim.keys()}
            self.accept_prob = {var: np.zeros([self.num_chains, self.num_samples, math.ceil(self.data_length[var] / self.block_size[var])]) for var in self.params_dim.keys()}
            self.ifaccept = {var: np.zeros([self.num_chains, self.num_samples, math.ceil(self.data_length[var] / self.block_size[var])]) for var in self.params_dim.keys()}   
        else:
            self.samples = {var: torch.zeros(((self.num_samples,) + tuple(dim))) for var, dim in self.params_dim.items()}
            # self.samples = {var: torch.zeros((self.num_samples + 1, math.ceil(self.data_length[var] / self.block_size[var]), ) + tuple(dim)) for var, dim in self.params_dim.items()}  
            self.logdensities = {var: torch.zeros(self.num_samples) for var in self.params_dim.keys()}
            self.proposed_logdensities = {var: torch.zeros(self.num_samples) for var in self.params_dim.keys()}
            self.accept_prob = {var: np.zeros([self.num_samples, math.ceil(self.data_length[var] / self.block_size[var])]) for var in self.params_dim.keys()}
            self.ifaccept = {var: np.zeros([self.num_samples, math.ceil(self.data_length[var] / self.block_size[var])]) for var in self.params_dim.keys()}   
            # self.proposed_samples = {var: torch.zeros((self.num_samples, math.ceil(self.data_length[var] / self.block_size[var]), ) + tuple(dim)) for var, dim in self.params_dim.items()}   

    

# class MCMC_Gibbs2(MCMC):
#     """the class to run Gibbs sampler for z"""
#     def __init__(self, variables, kernel_functions, block_size={'para': 10000, 'z':1}, num_samples=MCMC_SAMPLE, initial_params=None, params_dim=None, warmup_steps=MCMC_T//5, disable_progbar=MCMC_SHOW_DISABLE, warmup_settings=dict(target_prob=0.7, auto_init_stepsize=True), **kwargs):
#         """
#         kernel_functions: dictionary {'para': para_kernel, 'z': z_kernel}
#             - dictionary of a parameter kernel which conditioned on u, z, r, and a z kernel which could generate a block of z given u, and return the likelihood function
#         block_size: block size of z that are being updated together
#         """
#         self.num_samples = num_samples
#         self.kernel = kernel_functions
#         assert initial_params is not None or params_dim is not None, "Should either specify initial_params or params_dim"
#         if initial_params is not None:
#             self.initial_params = initial_params
#             self.params_dim = {var: torch.tensor(initial_params[var]).shape for var in variables}
#         else:
#             self.initial_params = torch.zeros(params_dim)
#             self.params_dim = params_dim

#         assert warmup_steps is None or isinstance(warmup_steps,int) or isinstance(warmup_steps, np.integer), "warmup_steps must be None or integer"
#         self.warmup_steps = 0 if warmup_steps is None else warmup_steps
#         self.warmup_settings = warmup_settings  
        
#         self.variables = variables      
#         self.block_size = block_size
        
#         self.reset_stat()
        
#     def run(self, idx=None):
        
#         self.reset_stat()
        
#         pbar = tqdm(range(self.num_samples), position=0, leave=True)
#         current_para = self.initial_params
#         current_para_info_dict = dict()
        
#         for var in self.variables:
#             self.kernel[var].model.set_var(var)
#             self.kernel[var].model.set_samples(current_para)
#             current_logtarget_density, current_para_llh_info_dict  = self.kernel[var].model.logtarget_density(parameter=current_para[var], llh_info_dict=dict())
#             current_para_info_dict[var] = {"logdensities":current_logtarget_density, "llh_info_dict":current_para_llh_info_dict}
#             self.samples[var][0] = torch.tensor(current_para[var])
#             self.logdensities[var][0] = current_logtarget_density
#             self.proposed_logdensities[var][0] = current_logtarget_density
            
#             if self.warmup_steps > 0:
#                 try: 
#                     current_para[var], current_para_info_dict[var], _ = self.warmup(var=var, init_para=current_para[var], init_para_info_dict=current_para_info_dict)
#                 except NotImplementedError:
#                     print("Warmup is not implemented for the current kernel. Skip to sampling...")
        
#         for i in pbar:
#             for var in self.variables:
#                 self.kernel[var].model.set_var(var)
#                 for j in range(math.ceil(self.data_length[var] / self.block_size[var])):
#                     '''proposed_block_z: block_size x M_Z x 2'''
#                     selected_indices = idx[var] if idx[var] is not None else slice(None)
#                     indices = range(j * self.block_size[var], min((j + 1) * self.block_size[var], self.data_length[var]))[selected_indices]
#                     accept_prob, proposed_para, proposed_para_info_dict = self.kernel[var].propose_accept(current_para=current_para[var], 
#                                                                 indices=indices, current_para_info_dict=current_para_info_dict[var])
#                     # if j==4 and var=='z':
#                     #     print(current_para[var], proposed_para)
#                         # print('accept prob', j, accept_prob)
#                     if np.random.uniform(0, 1) < accept_prob:
#                         current_para[var] = proposed_para
#                         current_para_info_dict[var] = proposed_para_info_dict
#                         self.accepted[var] += 1
                        
#                     self.accept_prob[var][i][j] = accept_prob
#                     # self.proposed_samples[var][i][j] = torch.tensor(proposed_para)
#                     # self.samples[var][i + 1][j] = torch.tensor(current_para[var])
#                 self.samples[var][i + 1] = torch.tensor(current_para[var])
#                 self.logdensities[var][i + 1] = current_para_info_dict[var]["logdensities"]
#                 self.proposed_logdensities['z'][i + 1] = proposed_para_info_dict["logdensities"]
#                 self.kernel[var].model.set_samples(current_para)
            
#             pbar.set_description("Acceptance probability {}".format(np.round(self.accepted['para']/(i+1), 2)))
#         return self.samples['para']
    
#     def warmup(self, var, init_para, init_para_info_dict=dict()):
#         current_para, current_para_info_dict, info = self.kernel[var].warmup(init_para=init_para, 
#                                                                         init_para_info_dict=init_para_info_dict,
#                                                                         iterations=self.warmup_steps, 
#                                                                         set_stepsize=True,
#                                                                         **self.warmup_settings)

#         return current_para, current_para_info_dict, info
    
#     def reset_stat(self):
#         self.data_length = {var: len(self.kernel[var].model.data) for var in self.variables}
#         self.samples = {var: torch.zeros(((self.num_samples + 1, ) + tuple(dim))) for var, dim in self.params_dim.items()}
#         # self.samples = {var: torch.zeros((self.num_samples + 1, math.ceil(self.data_length[var] / self.block_size[var]), ) + tuple(dim)) for var, dim in self.params_dim.items()}  
#         self.logdensities = {var: torch.zeros(self.num_samples+1) for var in self.params_dim.keys()}
#         self.proposed_logdensities = {var: torch.zeros(self.num_samples+1) for var in self.params_dim.keys()}
#         self.accepted = {var: 0 for var in self.params_dim.keys()}
#         self.accept_prob = {var: torch.zeros(self.num_samples, math.ceil(self.data_length[var] / self.block_size[var])) for var in self.params_dim.keys()}   
#         # self.proposed_samples = {var: torch.zeros((self.num_samples, math.ceil(self.data_length[var] / self.block_size[var]), ) + tuple(dim)) for var, dim in self.params_dim.items()}   

    
#MCMC
def MCMC_update(obs, posterior_samples, model, env, num_chains):
    global STEPSIZE, Model
    if BATCH_TRAINING:
        batch_indices = random.sample(range(min(len(obs._buffers['state0']), BUFFER_SIZE)), k=min(BATCH_SIZE, len(obs._buffers['state0']))) #TODO: what is this?
    else:
        batch_indices = slice(None)
        
    if TRANSFORM:
        prior = TruncatedGaussianPrior(sd=PRIOR_SIGMA)
    else:
        prior = IsotropicGaussianPrior(sd=PRIOR_SIGMA)
    data = torch.tensor(obs._buffers["rewards"], dtype=torch.float32)[-BUFFER_SIZE:][batch_indices]
    
    if STOCHASTIC:
        '''Stochastic'''
        abclikelihood = PartialGaussianABCLikelihood(epsilon=EPSILON)
        z_sample = generate_z(env=env, obs=obs._buffers)
        r_hat = partial(generate_samples_with_z, model=model, obs=obs._buffers, batch_indices=batch_indices)
        z_transform_func = partial(generate_z, env=env, obs=obs._buffers)
        #llh_transform_grad_fn = lambda parameter:  tabular_indicator_stochastic(para={'para': parameter['para'].reshape(env.n_cell + (env.action_space.n, )), 'z': parameter['z']}, model=model, obs=obs._buffers)#stochastic
        llh_transform_grad_fn = partial(tabular_indicator_stochastic,model=model, obs=obs._buffers)#stochastic
        Model = StochasticSModel(prior=prior, abclikelihood=abclikelihood, data=data, z_transform_fn=z_transform_func, llh_transform_fn=r_hat, llh_transform_grad_fn=llh_transform_grad_fn)

    else:
        '''Determinisitc'''
        if TRANSFORM:
            abclikelihood = TruncatedGaussianABCLikelihood(epsilon=EPSILON)
        else:
            abclikelihood = GaussianABCLikelihood(epsilon=EPSILON)
        r_hat = partial(generate_samples, model=model, obs=obs._buffers,  batch_indices=batch_indices)
        #llh_transform_grad_fn = lambda parameter:  tabular_indicator_deterministic(para=parameter, model=model, obs=obs._buffers)#standard form
        llh_transform_grad_fn = partial(tabular_indicator_deterministic, model=model, obs=obs._buffers)
        Model = DeterministicSRModel(prior=prior, abclikelihood=abclikelihood, data=data, llh_transform_fn=r_hat, llh_transform_grad_fn=llh_transform_grad_fn)

    def fn(parameter):
        current_logtarget_density, _ = Model.logtarget_density(parameter=parameter, llh_info_dict=dict())
        return current_logtarget_density
    if STOCHASTIC:
        # kernel = HMC(model=Model, stepsize=STEPSIZE, use_autograd=USE_AUTOGRAD, traj_len=0.1)
        kernel = RandomWalk(model=Model, stepsize=STEPSIZE)
        z_kernel = Z(model=Model)
    else:
        # hessian = torch.autograd.functional.hessian(fn, posterior_samples[0].reshape(-1)) + 1e-6 * torch.eye(len(posterior_samples[0]).reshape(-1))
        hessian = torch.autograd.functional.hessian(fn, torch.tensor(Q_star, dtype=torch.float32).reshape(-1)) - 1e-6 * torch.eye(len(posterior_samples[0].reshape(-1)))
        # kernel = RandomWalk(model=Model, stepsize=STEPSIZE)
        #kernel = RandomWalk(model=Model, stepsize=STEPSIZE, covariance_matrix=-torch.linalg.inv(hessian))
        #kernel = pCN(model=Model, stepsize=STEPSIZE)
        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=None)
        #kernel = MALA(model=Model, stepsize=STEPSIZE, precondition_matrix=-torch.linalg.inv(hessian))
        #kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=USE_AUTOGRAD)
        #kernel = MALA(model=Model, stepsize=STEPSIZE, use_autograd=USE_AUTOGRAD, precondition_matrix=-torch.linalg.inv(hessian))
        # kernel = HMC_pyro(model=Model, stepsize=STEPSIZE, full_mass=FULL_MASS, adapt_step_size=ADAPT_STEP_SIZE, adapt_mass_matrix=ADAPT_MASS_MATRIX, target_accept_prob=TARGET_ACCEPT_PROB, num_steps=NUM_STEPS)
        # kernel = HMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=USE_AUTOGRAD)
        # kernel = AM(model=Model,  stepsize=STEPSIZE)
        # kernel = HMC(model=Model, stepsize=STEPSIZE, use_autograd=USE_AUTOGRAD, mass=MASS, precondition_matrix=-torch.linalg.inv(hessian))
        kernel = HMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=USE_AUTOGRAD, mass=MASS, precondition_matrix=-torch.linalg.inv(hessian), traj_len=None)
        #kernel = mMALA(model=Model, stepsize=STEPSIZE, use_autograd=USE_AUTOGRAD, use_autohess=False)
        #kernel = mMALA(model=Model, stepsize=STEPSIZE, use_autograd=USE_AUTOGRAD, use_autohess=True)
        #kernel = mHMC(model=Model, stepsize=STEPSIZE, num_steps=NUM_STEPS, use_autograd=USE_AUTOGRAD, use_autohess=False, traj_len=None, fp_iterations=50)
    accept_probs = None
    STEPSIZE *= DECREASING_FACTOR

    if kernel.original is True:
        if STOCHASTIC:
            initial_params = [{'para': posterior_samples[-1].reshape(-1), 'z': z_sample}]*num_chains if num_chains > 1 else {'para': posterior_samples[-1].reshape(-1), 'z': z_sample} #TODO temporary
            mcmc = MCMC_Gibbs(variables=['para', 'z'], kernel_functions={'para': kernel, 'z': z_kernel}, num_samples=training_steps, initial_params=initial_params, warmup_steps=np.int64(np.floor(training_steps*WARMUP_RATIO)), warup_settings=dict(target_prob=0.7, auto_init_stepsize=True))
            posterior_samples = mcmc.run(idx={'para':FROZEN_NO, 'z': None})["para"].reshape((-1, ) + env.n_cell + (env.action_space.n, ))
        else:
            initial_params = posterior_samples[-1].reshape(-1).repeat([num_chains,1]) if num_chains > 1 else posterior_samples[-1].reshape(-1) #TODO temporary
            mcmc = MCMC(num_samples=training_steps, kernel=kernel, initial_params=initial_params, warmup_steps=np.int64(np.floor(training_steps*WARMUP_RATIO)), warup_settings=dict(target_prob=0.7, auto_init_stepsize=True), num_chains=num_chains, mp_settings=dict(comm_interval=COMM_INTERVAL, mp_context=MP_CONTEXT))
            posterior_samples = mcmc.run(idx=FROZEN_NO).reshape((-1, ) + env.n_cell + (env.action_space.n, ))
        logdensities = mcmc.get_logdensities()
        proposed_logdensities = mcmc.get_proposed_logdensities()
        accept_probs = mcmc.get_accept_prob()
        return posterior_samples, accept_probs, logdensities, proposed_logdensities, mcmc, kernel

    else:
        if STOCHASTIC:
            raise NotImplementedError('pyro model for stochastic hasn\'t been implemented' )
        else:
            mcmc = MCMC_pyro(num_samples=training_steps, kernel=kernel, initial_params=posterior_samples[-1].reshape(-1), warmup_steps=np.int64(np.floor(training_steps*WARMUP_RATIO)), disable_progbar=MCMC_SHOW_DISABLE)
        posterior_samples = mcmc.run().reshape((-1, ) + env.n_cell + (env.action_space.n, ))
        return posterior_samples, accept_probs, mcmc
    
def display_results(posterior_samples, model, accept_probs, logdensities, proposed_logdensities):
    display_indices = min(len(posterior_samples) // 10 + 1, 1000)
    model.plot_policy(paras=posterior_samples[-display_indices:].numpy(), title=f'policy_T{training_steps}_{time}', additional_info = env.R, save=save, show=show)
    # model.plot_value(paras=posterior_samples.numpy(), title=f'value_T{training_steps}_{time}')
    plt.imshow(torch.round(torch.max(torch.mean(posterior_samples[-display_indices:], 0), -1).values, decimals=2))
    plt.colorbar()
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
        
    #     fig, ax = plt.subplots(len(logdensities.keys()), 2, sharex=True)
    #     for i, var in enumerate(logdensities.keys()):

    #         ax[i, 0].plot(proposed_logdensities[var].numpy(), label="proposed samples")
    #         ax[i, 0].plot(logdensities[var].numpy(), label="accepted samples")
    #         ax[i, 0].set_xlabel(f"{var} samples")
    #         ax[i, 0].set_ylabel(f"log density of {var}")

    #         ax[i, 1].plot(accept_probs[var].numpy(), label="log acceptance probability", c="tab:green")
    #         ax[i, 1].set_ylabel("log acceptance probability")
    #     for i in range(len(logdensities.keys())):
    #         handles, labels = ax[-1, i].get_legend_handles_labels()
    #         ax[0, i].legend(handles, labels, bbox_to_anchor=(0.5, 1.5), loc='upper right')
    #     fig.tight_layout()

    #     if show:
    #         plt.show()


if __name__ == '__main__':
    from tqdm.notebook import tqdm
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
    
    #__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)" #https://stackoverflow.com/questions/45720153/python-multiprocessing-error-attributeerror-module-main-has-no-attribute
    __spec__ = None

    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--training_step', default=MCMC_T, type=int)
    parser.add_argument('-t', '--time', default=datetime.datetime.now().strftime("%f"))
    parser.add_argument('-s', '--save', default=SAVE)
    parser.add_argument('-p', '--show', default=SHOW)
    parser.add_argument('-e', '--epsilon', default=EPSILON, type=float)
    parser.add_argument('-n', '--stepsize', default=STEPSIZE, type=float)
    parser.add_argument('--mass', default=MASS, type=float)
    parser.add_argument('--seed', default=SEED, type=int)
    parser.add_argument('--num_chains', default=1, type=int)
    parser.add_argument('--MCMC', default=True, action='store_false', help='Bool type')
    parser.add_argument('-g', '--Greedy', default=GREEDY, action='store_true', help='Bool type')
    parser.add_argument('--Env', default=ENV_NAME)
    parser.add_argument('--sto', default=False)
    parser.add_argument('--online', default=False)
    parser.add_argument('--auto', default=False)
    parser.add_argument('--warmup', default=WARMUP_RATIO, type=float)
    parser.add_argument('--transform', default=TRANSFORM)
    args = parser.parse_args()
    print(args)
    #time = args.time
    #print('time:', time)
    training_steps = args.training_step
    save = args.save
    show = args.show
    abc_epsilon = args.epsilon
    seed = args.seed
    MCMC_SHOW_DISABLE=args.MCMC
    STEPSIZE = args.stepsize
    MASS = args.mass
    warmup_steps = int(training_steps * WARMUP_RATIO)
    GREEDY = args.Greedy
    env_name = args.Env
    STOCHASTIC = args.sto
    ONLINE_LEARNING = args.online
    USE_AUTOGRAD = args.auto
    WARMUP_RATIO = args.warmup
    TRANSFORM = args.transform
    
    num_chains = args.num_chains
    ADAPT_STEP_SIZE = True if WARMUP_RATIO > 0 else False
    ADAPT_MASS_MATRIX = False #True if WARMUP_RATIO > 0 else False

    N_PARTICLE = training_steps
    random.seed(seed)
    pyro.set_rng_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if env_name == 'GridWorld':
        env = GridWorld((3,3), obstacles=False, stochastic=STOCHASTIC)
        # env.plot_env()
    if env_name == 'Maze':
        env = Maze()
    
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
    dim = env.observation_space.n * env.action_space.n
    model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal', gamma=GAMMA, initial_tables=Q_star, idx=FROZEN_IDX)
    
    env.reset()
    results = []
    if ONLINE_LEARNING:
        r_all_iter = []
        for repeat in range(REPEAT_EXPERIMENT):
            STEPSIZE = INITIAL_STEPSIZE
            # model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal', gamma=GAMMA)
            model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal', gamma=GAMMA, initial_tables=Q_star, idx=FROZEN_IDX)#For frozen all but one dimensions
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
                        # posterior_samples, accept_probs = MCMC_update(posterior_samples=posterior_samples, obs=obs, model=model, env=env)[:2]
                        posterior_samples, accept_probs, logdensities, proposed_logdensities = MCMC_update(posterior_samples=posterior_samples, obs=obs, model=model, env=env, num_chains=num_chains)[:4]
                        model.set_parameter(posterior_samples, idx=FROZEN_IDX)
                        display_results(posterior_samples=posterior_samples, model=model, accept_probs=accept_probs, logdensities=logdensities, proposed_logdensities=proposed_logdensities)

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
        experiment_code = 1
        file_path = f"../Results/E{experiment_code}/T{training_steps}_Sto{STOCHASTIC}_{time}.json"
        error_all = []
        block_accept = []
        epsilon_lst = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4][2:3]
        data_percentage_lst = np.linspace(0.1, 1, 10)[-1:]
        s_l = [1e-4, 1e-3, 1e-2, 1e-1]
        for repeat in range(REPEAT_EXPERIMENT):
            print('Repeat no. ', repeat)
            error_all_epsilon = []
            samples_all_stepsize = []
            H_change_all_stepsize = []
            for EPSILON in epsilon_lst:
            # for STEPSIZE in s_l:
                print('MCMC Stepsize', STEPSIZE)
                error_all_percentage = []
                for data_percentage in data_percentage_lst:
                    print(f'Experiment with epsilon={EPSILON}, %={data_percentage}')
                    env.uniform_policy(data_percentage=data_percentage)
                    obs = env.uniform_obs
                    posterior_samples = model.get_parameter()
                    # posterior_samples, accept_probs = MCMC_update(posterior_samples=posterior_samples, obs=obs, model=model, env=env)[:2]
                    # print(posterior_samples)
                    if TRANSFORM:
                        posterior_samples = TruncatedGaussianABCLikelihood.log_neg_transform(posterior_samples)
                    # print(posterior_samples)
                    posterior_samples, accept_probs, logdensities, proposed_logdensities, mcmc, kernel = MCMC_update(posterior_samples=posterior_samples, obs=obs, model=model, env=env, num_chains=num_chains)
                    if TRANSFORM:
                        posterior_samples = TruncatedGaussianABCLikelihood.neg_exp_transform(posterior_samples)
                    model.set_parameter(posterior_samples, idx=FROZEN_IDX)
                    display_results(posterior_samples=posterior_samples, model=model, accept_probs=accept_probs, logdensities=logdensities, proposed_logdensities=proposed_logdensities)
                    average_over = min(100, len(posterior_samples))
                    error_all_percentage.append(mean_squared_error(Q_star.flatten(), torch.mean(posterior_samples[-average_over:].reshape(average_over, -1), axis=0)))
                    print('error_all_percentage', error_all_percentage)
                samples_all_stepsize.append(posterior_samples)
                H_change_all_stepsize.append(kernel.H_change)
                # error_all_epsilon.append(error_all_percentage)
            # error_all.append(error_all_epsilon)
            experiment_info = {
                'args': vars(args), 
                'epsilon_list': epsilon_lst, 
                'data_percentage': np.array(data_percentage_lst).tolist(),
                'errors': error_all
            }
            if save:
                json_data = json.dumps(experiment_info)
                with open(file_path, "w") as file:
                    file.write(json_data)
                print(f"Dictionary saved to {file_path}")
            if STOCHASTIC:
                block_accept.append(mcmc.accept_prob['z'].numpy())
        # plot_repeat(error_all, labels=epsilon_lst, label='Epsilon', x=data_percentage_lst, smooth=1, xlabel='data percentage', ylabel='MSE', title='MSE for offline learning vs DP')
        if STOCHASTIC:
            plot_block_accpt_prob(mcmc=mcmc, block_accept=np.array(block_accept))