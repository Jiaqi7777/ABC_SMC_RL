#Env 
env_name = 'GridWorld'
#env_name = 'MountainCar'
horizon = 40
#Number of Episodes
episodes = 300
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
update_frequency = 5
sigma = 1

#Results
show = True
