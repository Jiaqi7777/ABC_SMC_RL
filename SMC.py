import numpy as np
from MCMC import *
from MountainCar import *
from model import *

class SMC:
    def __init__(self, model, min_ess=0.5, kernel=None):
        self.n_particle = model.n_particle
        self.min_ess = min_ess 
        self._weights = model._weights
        self.model = model
        self._parameter = model.get_parameter()
        if kernel == None:
            kernel = RandomWalk(model=model)
        self._mcmc = MCMC(kernal=kernel)
        #self.parameter_history = np.array([self._parameter])

    def update(self, obs, samples_l, update_frequency=5):
        for j in range(self.n_particle):
            lld = stats.multivariate_normal.logpdf(obs['rewards'][-update_frequency:], samples_l[-update_frequency:,j])
            self._weights[j] *= lld
        self._weights /= sum(self._weights)
        self.model.set_weights(self._weights)
        new_parameter = self._mcmc.update(self.model.get_parameter(), obs, samples_l)
        #self.parameter_history = np.vstack((self.parameter_history, new_parameter))
        self.model.set_parameter(new_parameter)
        if self.ESS() < self.min_ess:
            self.resample(self.model)

    def ESS(self):
        return 1 / ( 1 + np.var(self._weights) )

    def resample(self, model):
        paras = random.choices(model.get_parameter(), self._weights, self.n_particle)
        self._weights = np.ones(self.n_particle) / self.n_particle
        self.model.set_parameter(paras)
        self.model.set_weights(self._weights)
        


if __name__ == '__main__':
    n_particle = 5    
    r = []
    env = MountainCar()
    model = Tabular(env=env, n_particle=n_particle, prior='normal')
    sampler = SMC(model)
    obs = {'rewards':[1,2,3], 'state0':[(1,2),(1,2),(1,2)],'state1':[(7,5),(8,2),(1,8)], 'action':[-1,0,1]}
    samples = np.vstack(([7,8,7,8,7],[4,5,4,5,4], [2,3,2,3,2]))
    sampler.update(obs, samples, 2)
    print(sampler._weights)
    print(model.tables.shape)

 