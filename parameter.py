#Env 
env_name = 'GridWorld'
#env_name = 'MountainCar'
horizon = 10
#Number of Episodes
episodes = 5
discrete = False
bins = (10, )
last_episode = 10
seed = 555

#SMC parameters
prior = 'normal'
lld = 'normal'
n_particle = 2
min_ess = 1

#MCMC
update_frequency = 5

#Results
show = True
