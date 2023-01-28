'''

from abcpy.continuousmodels import MultivariateNormal as MultiGaussian
from abcpy.inferences import SMCABC 
from abcpy.statistics import Identity
statistics_calculator = Identity(degree=2, cross=False)
from abcpy.distances import Euclidean
distance_calculator = Euclidean(statistics_calculator, seed=42)

import numpy as np
from abcpy.perturbationkernel import DefaultKernel
kernel = DefaultKernel([mu, sigma])
from abcpy.backends import BackendDummy as Backend
backend = Backend()
mean = [0]*10#np.zeros(10)
std = np.eye(10)*0.1#np.ones(10)*0.1

theta = MultiGaussian([mean, std.tolist()], name = 'theta')
sampler = SMCABC([theta], [distance_calculator], backend, seed=1)
'''
from functools import reduce
import numpy as np
from operator import mul
from MCMC import *
from model import *

class SMC:
    def __init__(self, n_particle, model, proposal=None):
        self.n_particle = n_particle
        self._weights = np.ones(n_particle) / n_particle
        self.model = model
        self._parameter = model.get_parameter()
        if proposal == None:
            proposal = RandomWalk()
        self._mcmc = MCMC(self._parameter, proposal)
        self.parameter_history = [self._parameter]

    def update(self, obs, samples_l):
        for j in range(self.n_particle):
            lld = self.model.lld(obs, samples_l[-1][j])
            self._weights[j] *= lld#reduce(mul,lld, 1)
        self._weights /= sum(self._weights)
        new_parameter = self._mcmc.update(self.model.get_parameter(), obs, samples_l)
        self.parameter_history.append(new_parameter)
        self.model.set_parameter(new_parameter)


if __name__ == '__main__':
    n_particle = 5    
    r = []
    model = Tabular(dim_s=3, dim_a=2, n_particle=n_particle, prior='normal')
    sampler = SMC(n_particle, model)
    obs = {'rewards':[1,2,3]}
    samples = [[7,8,7,8,7],[4,5,4,5,4], [2,3,2,3,2]]
    sampler.update(obs, samples)
    print(sampler._weights)
    print(model.tables)

 