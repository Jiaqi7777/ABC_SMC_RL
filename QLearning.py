import numpy as np
from utils import *
import torch
import matplotlib
import matplotlib.pyplot as plt
'''module import'''
from parameter import *
def DynamicProgramming(Q, A, S, env, thresh=1e-2, gamma=0.95, show=False, alpha=1):
    initial_alpha = alpha
    delta_l=[]
    loop = 0
    delta = thresh + 0.01
    while delta > thresh:
        loop += 1
    # for i in range(800):
        '''1-d state'''
        delta = thresh*0.9
        for s in S:
            for a in A:
                for _ in range(10):
                    pre_q = Q[s + (a,) ]
                    s1, r, done, _ = env.step(a, s)
                    if done:
                        new_q = Q[s + (a,) ] = r
                    else:
                        new_q = Q[s + (a,) ] = alpha * (r + gamma * max(Q[s1])) + (1 - alpha) * pre_q
                    delta = max(delta, abs(pre_q - new_q))
        alpha = initial_alpha / loop
        # print(alpha)
        delta_l.append(delta)
    
    V = np.max(Q, axis=-1)
    print('Q', np.round(Q, 2), delta)
    print(f'Converged with loop {loop}')
    print('Value', np.round(V, 2))
    print('Policy:', np.argmax(Q, axis=-1))
    pi = np.argmax(Q, axis=-1)
    plt.imshow(V)
    plt.colorbar()
    if show:
        plt.show()
    return pi, Q, V#, delta_l

def QLearning(Q, env, n_episodes=10, horizon=50, gamma=0.95, epsilon=0.4, alpha=0.2):
    initial_alpha = alpha
    # while delta > thresh:
    #     loop += 1
    return_all_episodes_qlearning = []
    regret_all_episodes_qlearning = []
    Regret = 0
    for e in range(n_episodes):
        alpha = initial_alpha / (e+1)
        R = 0
        Regret = 0
        s0, _ = env.reset()
        for t in range(horizon):
            a = np.argmax(Q[s0]) if np.random.uniform(0, 1) > epsilon else random.choice(range(env.action_space.n))
            s1, r, done, _ = env.step(a)
            # v_optimal = V_star[s0]#env.R[tuple(env.P[s0 + (int(pi_star[s0]), )])]
            R += r #v_optimal - sum([r * gamma ** i for i in range(t + 1)])
            # Regret += V_star[s0] - Q_star[s0][a]
            Q[s0][a] = alpha * (r + gamma * max(Q[s1])) + (1 - alpha) * Q[s0][a]
            if done:
                print('Done in ', t + 1, 'steps')
                break
            s0 = s1
            
        return_all_episodes_qlearning.append(R)
        # regret_all_episodes_qlearning.append(Regret)
    V = np.max(Q, axis=-1)
    print('Value', np.round(V, 2))
    pi = np.argmax(Q, axis=-1)
    # print('Value', V)
    print('Policy:', pi)
    # plot_return_vs_episodes(r_all_episodes_qlearning, smooth=10, show=True)
    print(np.round(Q,3))
    return pi, Q, V, return_all_episodes_qlearning, regret_all_episodes_qlearning

def QLearningWithData(Q, obs, gamma=0.95, training_steps=100, alpha=0.2):
    # data = zip(obs['state0'], obs['state1'], obs['action'], obs['rewards'], obs['done'])
    for _ in range(training_steps):
        for data in zip(obs['state0'], obs['state1'], obs['action'], obs['rewards'], obs['done']):
            s0, s1, a, r, done = data
            Q[s0][a] += alpha * (r + gamma * max(Q[s1]) - Q[s0][a])
            
    V = np.max(Q, axis=-1)
    print('Value', np.round(V, 2))
    pi = np.argmax(Q, axis=-1)
    print('Policy:', pi)
    # print(np.round(Q,3))
    return pi, Q, V
    
if __name__ == '__main__':
    from Environment.GridWorld import *
    from Environment.Maze import *
    import datetime
    time =  datetime.datetime.now()
    time = time.strftime("%f")
    training_steps = 500
    repeat = 10
    N_PARTICLE = 1
    STEPSIZE = 0.01
    QLEARNING = True
    r = []
    random.seed(SEED)
    np.random.seed(SEED)
    
    env = GridWorld((3,4), obstacles=True, stochastic=STOCHASTIC)
    # env = Maze()
    S = []
    if len(env.n_cell) == 1:
        S = [(i, ) for i in range(env.n_cell[0])]
    else:
        for i in range(env.n_cell[0]):
            for j in range(env.n_cell[1]):
                S.append((i,j)) 
    V = np.ones(env.n_cell)/env.observation_space.n
    Q = np.ones(shape=(env.n_cell + (env.action_space.n, )))/env.observation_space.n / env.action_space.n
    print('Q table with shape', Q.shape)
    return_all_repeat_qlearning = []
    regret_all_repeat_qlearning = []
    # S = range(env.n_cell)
    A = range(env.action_space.n)
    smooth = 50
    # for alpha in [0.2, 0.3, 0.5, 0.7, 1]:
    #     pi, Q, V, return_,regret_ = QLearning(Q, env, n_episodes=1000, horizon=HORIZON)
    #     plt.plot(np.convolve(np.array(return_), np.ones(smooth)/smooth, mode='valid'), label=f'Initial alpha={alpha}')
    pi_star, Q_star, V_star = DynamicProgramming(Q, A, S, env, thresh=1e-1, gamma=1, show=True)
    # plt.plot(delta_l)
    # plt.legend()
    # plt.show()
    # if QLEARNING:
    #     for repeat in range(REPEAT_EXPERIMENT):
    #         env.reset()
    #         V = np.ones(env.n_cell)/env.observation_space.n
    #         Q = np.ones(shape=(env.n_cell + (env.action_space.n, )))/env.observation_space.n / env.action_space.n
    #         print('Q table with shape', Q.shape)
    #         # S = range(env.n_cell)
    #         A = range(env.action_space.n)
    #         # DynamicProgramming(V, A, S, env)
    #         pi, Q, V, return_,regret_ = QLearning(Q, env, n_episodes=EPISODES, horizon=HORIZON)
    #         return_all_repeat_qlearning.append(return_)
    #         regret_all_repeat_qlearning.append(regret_)
    #         # S = []
    #         # for i in range(env.n_cell[0]):
    #         #     for j in range(env.n_cell[1]):
    #         #         S.append((i,j)) 
    #         # DynamicProgramming(Q, A, S, env)
    #     with open(f'Returns/MCMC/q_learning_return_T{training_steps}_{time}.npy', 'wb') as f:
    #         np.save(f, return_all_repeat_qlearning)
    #         print('Return saved at', f'Returns/MCMC/q_learning_return_T{training_steps}_{time}.npy')
    #     with open(f'Regrets/MCMC/q_learning_return_T{training_steps}_{time}.npy', 'wb') as f:
    #         np.save(f, regret_all_repeat_qlearning)
    #         print('Regrets saved at', f'Regrets/MCMC/q_learning_return_T{training_steps}_{time}.npy')