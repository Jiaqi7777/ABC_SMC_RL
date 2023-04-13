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
<<<<<<< HEAD
=======
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue_train', default=False)
    parser.add_argument('-o', '--output', default='')
    parser.add_argument('-f', '--file', default=f'Models/{ENV_NAME}_H{HORIZON}_P{n_particle}_B{BINS[0]}_E{LAST_EPISODE}')
    args = parser.parse_args()
    continue_training = args.continue_train
    file_path = args.file
    # np.random.seed(SEED)
    
    time =  datetime.datetime.now()
    time = time.strftime("%f")
>>>>>>> d98a19051e20eac8898945d07964e45fef92a695

    #Environment
    #env = gym.make('MountainCar-v0')
    env = MountainCar()

    s0, _ = env.reset()

    horizon = 500

    
<<<<<<< HEAD
=======
    R_all_experiment = []
    for r in range(REPEAT_EXPERIMENT):
        #SMC
        model = Tabular(env, n_particle, prior, discrete=DISCRETE, bins=BINS)
        if continue_training:
            model.set_parameter(np.load(f'{file_path}_tables.npy'))
            model.set_weights(np.load(f'{file_path}_weights.npy'))
            np.random.seed(LAST_EPISODE)
        else:
            LAST_EPISODE = 0
        sampler = SMC(model, min_ess=min_ess)
        # pre_p = model.get_parameter()
>>>>>>> d98a19051e20eac8898945d07964e45fef92a695

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
<<<<<<< HEAD
        if discrete:
            s0 = model.discrete_state(s0)
        for t in range(horizon):
            action = model.act(s0, para)
            s1, r, done, *info = env.step(action)
            if discrete:
                s1 = model.discrete_state(s1)
            obs.insert({'state': s1, 'action': action, 'rewards': r, 'done': done})
            samples = model.r_hat(s0, s1, action)
            samples_l.append(samples)
            s0 = s1

        sampler.update(obs._buffers, samples_l)

=======

        #Discretize
        if DISCRETE:
            s0 = model.discrete_state(s0)
        
        obs = Buffer(['state0', 'state1', 'action', 'rewards', 'done'])
        samples_l = []
        R_l=[]
        for e in tqdm(range(EPISODES)):
            R = 0
            sampler._mcmc.reset()
            para = model.sample_para(sampler._weights)[0]
            s0, _ = env.reset()
            if DISCRETE:
                s0 = model.discrete_state(s0)
            for t in range(HORIZON):
                #print('Time:', t, model.get_parameter())
                action = model.act(s0)
                s1, r, done, *info = env.step(action)
                R += r
                # print('s1',s1,r,action)
                if DISCRETE:
                    s1 = model.discrete_state(s1)
                obs.insert({'state0': s0, 'state1': s1, 'action': action, 'rewards': r, 'done': done})
                samples = model.r_hat(s0, s1, action)
                samples_l.append(samples)
                s0 = s1
                if ( t + 1 ) % update_frequency == 0:
                    samples_l = list(sampler.update(obs._buffers, samples_l, update_frequency))
                if done:
                    print("Done!!")
                    print(t, obs._buffers['state0'][-t-1:])
                    print('============================================')
                    break
                # print('tables:', (model.get_parameter()==pre_p).all())
                # pre_p = model.get_parameter
            # print(t, obs._buffers['state0'][-t-1:])
            # print(f'total accepted for episode {e}:', round(sampler._mcmc.accepted / n_particle / HORIZON * update_frequency, 2))
            
            # model.plot_policy(title=f'{ENV_NAME} Policy for Episode={e + LAST_EPISODE + 1}', xlabel='position', ylabel='velocity', show=show)
            R_l.append(R)
        R_all_experiment.append(R_l)
    plot_return_vs_episodes_repeat(R_all_experiment, title=f'{ENV_NAME}_Return_{time}')
    model.plot_value(title=f'{ENV_NAME} Value for Episode={e + LAST_EPISODE + 1}', xlabel='position', ylabel='velocity', show=show)
    #     model.save(e + LAST_EPISODE + 1, args.output, HORIZON=HORIZON, ENV_NAME=ENV_NAME)
    # replace_line('parameter.py', 'LAST_EPISODE', e + LAST_EPISODE + 1)
>>>>>>> d98a19051e20eac8898945d07964e45fef92a695
