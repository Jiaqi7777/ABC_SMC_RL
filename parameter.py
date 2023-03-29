#RL
#Env 
env_name = 'GridWorld'
#env_name = 'MountainCar'

#Test parameters
# horizon = 15
# #Number of Episodes
# episodes = 2
# gamma = 0.95
# repeat_experiment = 2
# discrete = False
# bins = (10, )
# last_episode = 1
# ONLINE_LEARNING = True
# seed = 555
# FROZEN_T = 5

#Long Experiment
horizon = 50
#Number of Episodes
episodes = 6
gamma = 0.95
repeat_experiment = 10
discrete = False
bins = (10, )
last_episode = 1
ONLINE_LEARNING = True
seed = 555
FROZEN_T = 15
batch_training = True
batch_size = 64
buffer_size = 100

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
MCMC_SAMPLE = 10
BURN_IN = 0.1
skip = 10
target_accept_prob = 0.6
adapt_step_size = False

#ABC
epsilon = 0.1

#Results
save=False
show = True
