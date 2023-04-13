import numpy as np
from MCMC import *
from MountainCar import *
from model import *
from parameter import *


class SMC:
    def __init__(self, model, min_ess=0.5, kernel=None):
        self.n_particle = model.n_particle
        self.min_ess = min_ess 
        self._weights = model._weights
        self.model = model
        self._parameter = model.get_parameter()
        if kernel == None:
            kernel = RandomWalk(model=model, sigma=sigma)
        self._mcmc = MCMC(kernal=kernel)

    def update(self, obs, samples_l, update_frequency=5):
        samples_l = np.array(samples_l)
        for j in range(self.n_particle):
            lld = stats.multivariate_normal.logpdf(obs['rewards'][-update_frequency:], samples_l[-update_frequency:,j])
            self._weights[j] *= lld
        self._weights /= sum(self._weights)
        self.model.set_weights(self._weights)
        if self.ESS() < self.min_ess * self.n_particle:
            # print('Resampled')
            samples_l = self.resample(self.model, samples_l)
        new_parameter, samples_l = self._mcmc.update(self.model.get_parameter(), obs, samples_l)
        self.model.set_parameter(new_parameter)
        return samples_l

    def ESS(self):
        return 1 / np.sum(self._weights**2)
        return 1 / ( 1 + np.var(self._weights) )

    def resample(self, model, samples_l):
        idx = random.choices(range(self.n_particle), self._weights, k=self.n_particle)
        paras = model.get_parameter()[idx]
        self._weights = np.ones(self.n_particle) / self.n_particle
        self.model.set_parameter(paras)
        self.model.set_weights(self._weights)
        return samples_l[:, idx]
        


if __name__ == '__main__':
    N_PARTICLE = 5    
    r = []
    env = MountainCar()
    model = Tabular(env=env, n_particle=N_PARTICLE, prior='normal')
    sampler = SMC(model)
    obs = {'rewards':[1,2,3], 'state0':[(1,2),(1,2),(1,2)],'state1':[(7,5),(8,2),(1,8)], 'action':[-1,0,1]}
    samples = np.vstack(([7,8,7,8,7],[4,5,4,5,4], [2,3,2,3,2]))
    sampler.update(obs, samples, 2)
    print(sampler._weights)
    print(model.tables.shape)

 