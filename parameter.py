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
batch_training = False
batch_size = 64
buffer_size = 100 if batch_training else 100000000000
#policy
GREEDY = False

#SMC parameters
prior = 'normal'
lld = 'normal'
n_particle = 4
min_ess = 0.5

#MCMC
prior_sigma = 10
update_frequency = 5
stepsize = 0.005
MCMC_T = 100
MCMC_SAMPLE = 10
BURN_IN = 0.1
skip = 10
target_accept_prob = 0.6
adapt_step_size = False
adapt_mass_matrix = False
num_steps = 2000
warmup_ratio = 0
full_mass = False
#Plot
plot_threshold = 2#percentile

#ABC
epsilon = 0.05

#Results
save = False
show = True
