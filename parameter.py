#Env 
env_name = 'GridWorld'
#env_name = 'MountainCar'
horizon = 40
#Number of Episodes
episodes = 40
discrete = False
bins = (10, )
last_episode = 10
seed = 555

#SMC parameters
prior = 'normal'
lld = 'normal'
n_particle = 5
min_ess = 0.5

#MCMC
update_frequency = 5
sigma = 1

#Results
show = True
