import time
import gym
import numpy as np
from MountainCar import*
import gym
import matplotlib.pyplot as plt 

from parameter import *


file_path=f'Models/MountainCar_H_{horizon}_P{n_particle}_B{bins[0]}_E{last_episode}'
env = MountainCar()
test_env = gym.make('MountainCar-v0', render_mode='human')
render = lambda : plt.imshow(test_env.render())
tables = np.load(f'{file_path}_tables.npy')

weights = np.load(f'{file_path}_weights.npy')
optimal_table = tables[np.argmax(weights)]
model = Tabular(env, n_particle, 'normal')
s0, _ = test_env.reset()
score = 0

for i in range(200):
    s0 = model.discrete_state(s0)
    test_env.render()
    time.sleep(0.01)
    a = np.argmax(optimal_table[s0])
    s, reward, done, *info = test_env.step(a)
    score += reward
    if done:
        print('score:', score)
        break
    s = s0
test_env.close()