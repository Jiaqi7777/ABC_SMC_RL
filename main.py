'''
import SMC
import random
import bandits
import numpy as np
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

n_arms = 10
backend = Backend()
mean = [0]*n_arms#np.zeros(10)
std = np.eye(n_arms)*0.1#np.ones(10)*0.1

theta = MultiGaussian([mean, std.tolist()], name = 'theta')
sampler = SMCABC([theta], [distance_calculator], backend, seed=1)

random.seed(1)
horizon = 10
obs_steps = 10
train_steps = 2
n_arms = 10
n_sample = 5
bandit = bandits.SimpleGaussianBandit(n_arms=n_arms) 
for t in range(horizon):
    r_l = []
    for i in range(obs_steps):
        action = bandit.select_arm(theta)
        r = bandit.pull_arm(action)
        r_l.append(r)
    journal = sampler.sample([r_l], train_steps, n_sample)
    theta = random.choice(journal.accepted_parameters)
print(theta)
'''
from SMC import *
from model import *
from MCMC import *
from MountainCar import *
import gym 
import numpy as np

#Tabular method
if __name__ == '__main__':

    #Environment
    #env = gym.make('MountainCar-v0')
    env = MountainCar()

    s0, _ = env.reset()

    horizon = 500

    

    #SMC
    n_particle = 2
    prior = 'normal'
    lld = 'normal'
    train_steps = 2
    model = Tabular(env, n_particle, prior)
    sampler = SMC(n_particle, model)

    #Discretize
    discrete = True
    if discrete:
        s0 = model.discrete_state(s0)

    obs = Buffer(['state', 'action', 'rewards', 'done'])
    samples_l = []
    for i in range(train_steps):
        para = model.sample_para()[0]
        s0, _ = env.reset()
        if discrete:
            s0 = model.discrete_state(s0)
        for t in range(horizon):
            action = model.act(s0, para)
            s1, r, done, *info = env.step(action)
            if done:
                break
            if discrete:
                s1 = model.discrete_state(s1)
            obs.insert({'state': s1, 'action': action, 'rewards': r, 'done': done})
            samples = model.r_hat(s0, s1, action)
            samples_l.append(samples)
            s0 = s1

        sampler.update(obs._buffers, samples_l)

