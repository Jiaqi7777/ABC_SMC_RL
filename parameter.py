#RL
#Env 
ENV_NAME = 'GridWorld'
#ENV_NAME = 'MountainCar'

#Test parameters
# HORIZON = 15
# #Number of EPISODES
# EPISODES = 2
# GAMMA = 0.95
# REPEAT_EXPERIMENT = 2
# DISCRETE = False
# BINS = (10, )
# LAST_EPISODE = 1
# ONLINE_LEARNING = True
# SEED = 555
# FROZEN_T = 5

#Long Experiment
<<<<<<< HEAD
horizon = 100
#Number of Episodes
episodes = 50
gamma = 0.95
repeat_experiment = 1
discrete = False
bins = (10, )
last_episode = 1
=======
HORIZON = 100
#Number of EPISODES
EPISODES = 6
GAMMA = 0.95
REPEAT_EXPERIMENT = 4
DISCRETE = False
BINS = (10, )
LAST_EPISODE = 1
>>>>>>> d98a19051e20eac8898945d07964e45fef92a695
ONLINE_LEARNING = True
SEED = 555
FROZEN_T = 15
BATCH_TRAINING = False
BATCH_SIZE = 64
BUFFER_SIZE = 100 if BATCH_TRAINING else 100000000000
GREEDY = False
greedy_epsilon = 0.3


#SMC parameters
prior = 'normal'
lld = 'normal'
n_particle = 4
min_ess = 0.5

#MCMC
prior_sigma = 10
update_frequency = 5
initial_stepsize = stepsize = 0.01
decreasing_factor = 0.9
MCMC_T = 100
MCMC_SHOW_DISABLE = False
MCMC_SAMPLE = 10
BURN_IN = 0.1
skip = 10
target_accept_prob = 0.4
warmup_ratio = 0
adapt_step_size = True if warmup_ratio > 0 else False
adapt_mass_matrix = True if warmup_ratio > 0 else False
num_steps = 2000
full_mass = False
#Plot
plot_threshold = 2#percentile

#ABC
epsilon = 0.05

#Results
save = True
show = True
