#RL
#Env 
env_name = 'GridWorld'
#env_name = 'MountainCar'
horizon = 10
#Number of Episodes
episodes = 100
gamma = 0.95
repeat_experiment = 20
discrete = False
bins = (10, )
last_episode = 1
ONLINE_LEARNING = True
seed = 555
FROZEN_T = 3

#SMC parameters
prior = 'normal'
lld = 'normal'
n_particle = 4
min_ess = 0.5

#MCMC
prior_sigma = 10
update_frequency = 5
stepsize = 0.01
MCMC_T = 100
BURN_IN = 0.1
skip = 10

#ABC
epsilon = 0.01

#Results
show = True
