from SMC import *
from model import *
from MCMC import *
from MountainCar import *
from GridWrold import *
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
    parser.add_argument('-f', '--file', default=f'Models/{env_name}_H{horizon}_P{n_particle}_B{bins[0]}_E{last_episode}')
    args = parser.parse_args()
    continue_training = args.continue_train
    file_path = args.file
    np.random.seed(seed)

    #Environment
    #env = gym.make('MountainCar-v0')
    #env = MountainCar()
    env = GridWorld(11, 6, 10)
    #SMC
    model = Tabular(env, n_particle, prior, discrete=discrete, bins=bins)
    if continue_training:
        model.set_parameter(np.load(f'{file_path}_tables.npy'))
        model.set_weights(np.load(f'{file_path}_weights.npy'))
        np.random.seed(last_episode)
    else:
        last_episode = 0
    sampler = SMC(model, min_ess=min_ess)

    s0, _ = env.reset()

    #Discretize
    if discrete:
        s0 = model.discrete_state(s0)

    obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
    samples_l = []
    for e in range(episodes):
        
        sampler._mcmc.reset()
        para = model.sample_para(sampler._weights)[0]
        s0, _ = env.reset()
        if discrete:
            s0 = model.discrete_state(s0)
        for t in range(horizon):
            #print('Time:', t, model.get_parameter())
            action = model.act(s0, para)
            s1, r, done, *info = env.step(action)
            #print(s1,r,action)
            if discrete:
                s1 = model.discrete_state(s1)
            obs.insert({'state0': s0, 'state1': s1, 'action': action, 'rewards': r, 'done': done})
            samples = model.r_hat(s0, s1, action)
            samples_l.append(samples)
            s0 = s1
            if ( t + 1 ) % update_frequency == 0:
                sampler.update(obs._buffers, samples_l, update_frequency)
            if done:
                print("Done!!")
                break
        print(obs._buffers['state0'][-horizon:])
        print(f'total accepted for episode {e}:', sampler._mcmc.accepted)
        # model.plot_policy(title=f'{env_name} Policy for Episode={e + last_episode + 1}', xlabel='position', ylabel='velocity', show=show)
        # model.plot_value(title=f'{env_name} Value for Episode={e + last_episode + 1}', xlabel='position', ylabel='velocity', show=show)
    #     model.save(e + last_episode + 1, args.output, horizon=horizon, env_name=env_name)
    # replace_line('parameter.py', 'last_episode', e + last_episode + 1)