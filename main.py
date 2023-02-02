from SMC import *
from model import *
from MCMC import *
from MountainCar import *
import gym 
import numpy as np
from parameter import *
from utils import *
import argparse


#Tabular method
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue_train', default=False)
    parser.add_argument('-o', '--output', default='')
    parser.add_argument('-f', '--file', default=f'Models/MountainCar_P{n_particle}_B{bins[0]}_E{last_episode}')
    args = parser.parse_args()
    continue_training = args.continue_train
    file_path = args.file
    np.random.seed(seed)

    #Environment
    #env = gym.make('MountainCar-v0')
    env = MountainCar()

    #SMC
    model = Tabular(env, n_particle, prior)
    if continue_training:
        model.set_parameter(np.load(f'{file_path}_tables.npy'))
        model.set_weights(np.load(f'{file_path}_weights.npy'))
        np.random.seed(last_episode)
    else:
        last_episode = 0
    sampler = SMC(model)

    s0, _ = env.reset()

    #Discretize
    discrete = True
    if discrete:
        s0 = model.discrete_state(s0)

    obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
    samples_l = []
    for e in range(episodes):
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
                print("Done!!")
                break
        model.plot_policy(title=f'MountainCar Policy for Episode={e + last_episode + 1}', xlabel='position', ylabel='velocity')
        model.plot_value(title=f'MountainCar Value for Episode={e + last_episode + 1}', xlabel='position', ylabel='velocity')
        model.save(e + last_episode + 1, args.output, horizon=horizon)
    replace_line('parameter.py', 'last_episode', e + last_episode)