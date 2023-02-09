import numpy as np
from utils import *
def DynamicProgramming(V, A, S, env, thresh = 0.2, gamma=0.95):
    loop = 0
    delta = thresh + 0.00000001
    pi = np.zeros(V.shape)
    while delta > thresh:
        loop += 1
    # for i in range(2):
        '''1-d state'''
        delta = 0.0001
        for s in S:
            pre_v = V[s]
            # print('prev', s, pre_v)
            qsa = []
            for a in A:
                s1, r, done, _ = env.step(a, s)
                qsa.append(r + gamma * V[s1])
            pi[s] = np.argmax(qsa)
            new_v = max(qsa)
            # print('newv',new_v)
            V[s] = new_v
            delta = max(delta, abs(pre_v - new_v))
        print('Value', V, delta)
    print(f'Converged with loop {loop}')
    print('Value', V)
    print('Policy:', pi)
    return pi

def QLearning(Q, env, n_episodes=10, horizon=40, gamma=0.95):
    # while delta > thresh:
    #     loop += 1
    r_all_episodes = []
    for _ in range(n_episodes):
        R = 0
        s0, _ = env.reset()
        for t in range(horizon):
            a = np.argmax(Q[s0])
            s1, r, done, _ = env.step(a)
            R += r
            Q[s0][a] = r + gamma * max(Q[s1])
            if done:
                print('Done!!!')
                break
            s0 = s1
            
        r_all_episodes.append(R)
    print('Value', np.max(Q, axis=1))
    pi = np.argmax(Q, axis=1)
    # print('Value', V)
    print('Policy:', pi)
    plot_return_vs_episodes(r_all_episodes, smooth=10)
    return pi
    
if __name__ == '__main__':
    from GridWrold import *
    env = GridWorld(11,6,9)
    V = np.ones(env.n_cell)/env.n_cell
    Q = np.ones(shape=(env.n_cell, env.action_space.n))
    print('Q table with shape', Q.shape, Q)
    S = range(env.n_cell)
    A = range(env.action_space.n)
    # DynamicProgramming(V, A, S, env)
    QLearning(Q, env, n_episodes=50, horizon=40)