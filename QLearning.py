import numpy as np
from utils import *
from parameter import *
import torch
def DynamicProgramming(Q, A, S, env, thresh=1e-5, gamma=0.95):
    loop = 0
    delta = thresh + 0.01
    while delta > thresh:
        loop += 1
    # for i in range(2):
        '''1-d state'''
        delta = thresh*0.9
        for s in S:
            for a in A:
                pre_q = Q[s + (a,) ]
                s1, r, done, _ = env.step(a, s)
                if done:
                    new_q = Q[s + (a,) ] = r
                else:
                    new_q = Q[s + (a,) ] = r + gamma * max(Q[s1])
                delta = max(delta, abs(pre_q - new_q))
    
    V = np.max(Q, axis=-1)
    print('Q', np.round(Q, 2), delta)
    print(f'Converged with loop {loop}')
    print('Value', np.round(V, 2))
    print('Policy:', np.argmax(Q, axis=-1))
    pi = np.argmax(Q, axis=-1)
    return pi, Q, V

def QLearning(Q, env, n_episodes=10, horizon=50, gamma=0.95, epsilon=0.4):
    # while delta > thresh:
    #     loop += 1
    r_all_episodes_qlearning = []
    for _ in range(n_episodes):
        R = 0
        s0, _ = env.reset()
        for t in range(horizon):
            a = np.argmax(Q[s0]) if np.random.uniform(0, 1) > epsilon else random.choice(range(env.action_space.n))
            s1, r, done, _ = env.step(a)
            # v_optimal = V_star[s0]#env.R[tuple(env.P[s0 + (int(pi_star[s0]), )])]
            R += r #v_optimal - sum([r * gamma ** i for i in range(t + 1)])
            Q[s0][a] = r + gamma * max(Q[s1])
            if done:
                # print('Done in ', t + 1, 'steps')
                break
            s0 = s1
            
        r_all_episodes_qlearning.append(R)
    V = np.max(Q, axis=-1)
    print('Value', np.round(V, 2))
    pi = np.argmax(Q, axis=-1)
    # print('Value', V)
    print('Policy:', pi)
    # plot_return_vs_episodes(r_all_episodes_qlearning, smooth=10, show=True)
    print(np.round(Q,3))
    return pi, Q, V, r_all_episodes_qlearning

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
    from GridWorld import *
    from Maze import *
    import datetime
    time =  datetime.datetime.now()
    time = time.strftime("%f")
    training_steps = 50
    repeat = 100
    N_PARTICLE = 1
    STEPSIZE = 0.03
    r = []
    random.seed(SEED)
    np.random.seed(SEED)
    
    env = GridWorld((3,4), obstacles=True)
    env = Maze()
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
    r_all_repeat_qlearning = []
    # S = range(env.n_cell)
    A = range(env.action_space.n)
    pi_star, Q_star, V_star = DynamicProgramming(Q, A, S, env, gamma=1)
    for repeat in range(REPEAT_EXPERIMENT):
        env.reset()
        V = np.ones(env.n_cell)/env.observation_space.n
        Q = np.ones(shape=(env.n_cell + (env.action_space.n, )))/env.observation_space.n / env.action_space.n
        print('Q table with shape', Q.shape)
        # S = range(env.n_cell)
        A = range(env.action_space.n)
        # DynamicProgramming(V, A, S, env)
        pi, Q, V, r_, = QLearning(Q, env, n_episodes=EPISODES, horizon=HORIZON)
        r_all_repeat_qlearning.append(r_)
        # S = []
        # for i in range(env.n_cell[0]):
        #     for j in range(env.n_cell[1]):
        #         S.append((i,j)) 
        # DynamicProgramming(Q, A, S, env)
    with open(f'Returns/MCMC/q_learning_return_T{training_steps}_{time}.npy', 'wb') as f:
        np.save(f, r_all_repeat_qlearning)
        print('Return saved at', f'Returns/MCMC/q_learning_return_T{training_steps}_{time}.npy')