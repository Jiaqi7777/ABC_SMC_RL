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
    sampler = SMC(model)
    update_frequency = 5
    #Discretize
    discrete = True
    if discrete:
        s0 = model.discrete_state(s0)

    obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
    samples_l = []
    for i in range(train_steps):
        para = model.sample_para(sampler._weights)[0]
        s0, _ = env.reset()
        if discrete:
            s0 = model.discrete_state(s0)
        for t in range(horizon):
            action = model.act(s0, para)
            s1, r, done, *info = env.step(action)
            if discrete:
                s1 = model.discrete_state(s1)
            obs.insert({'state0': s0, 'state1': s1, 'action': action, 'rewards': r, 'done': done})
            samples = model.r_hat(s0, s1, action)
            samples_l.append(samples)
            s0 = s1
            if t % update_frequency == 0:
                sampler.update(obs._buffers, samples_l, update_frequency)
            if done:
                break

