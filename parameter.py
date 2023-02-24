#Env 
env_name = 'GridWorld'
#env_name = 'MountainCar'
horizon = 30
#Number of Episodes
repeat_experiment = 20
episodes = 100
discrete = False
bins = (10, )
last_episode = 1
seed = 555

#SMC parameters
prior = 'normal'
lld = 'normal'
n_particle = 4
min_ess = 0.5

#MCMC
prior_sigma = 20
update_frequency = 5
stepsize = 0.1

#ABC
epsilon = 1

#Results
show = True
