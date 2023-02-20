import numpy as np
from utils import *
def DynamicProgramming(Q, A, S, env, thresh = 1e-4, gamma=0.95):
    loop = 0
    delta = thresh + 0.01
    pi = np.zeros(V.shape)
    while delta > thresh:
        loop += 1
    # for i in range(2):
        '''1-d state'''
        delta = 0.0001
        for s in S:
            for a in A:
                pre_q = Q[s + (a,) ]
                s1, r, done, _ = env.step(a, s)
                new_q = Q[s + (a,) ] = r + gamma * max(Q[s1])
                delta = max(delta, abs(pre_q - new_q))
            
    print('Q', np.round(Q, 3), delta)
    print(f'Converged with loop {loop}')
    print('Value', np.max(Q, axis=-1))
    print('Policy:', np.argmax(Q, axis=-1))
    return pi

def QLearning(Q, env, n_episodes=10, horizon=50, gamma=0.95, epsilon=0.4):
    # while delta > thresh:
    #     loop += 1
    r_all_episodes = []
    for _ in range(n_episodes):
        R = 0
        s0, _ = env.reset()
        for t in range(horizon):
            a = np.argmax(Q[s0]) if np.random.uniform(0, 1) > epsilon else random.choice(range(env.action_space.n))
            s1, r, done, _ = env.step(a)
            R += r
            Q[s0][a] = r + gamma * max(Q[s1])
            if False:#done:
                # print('Done in ', t + 1, 'steps')
                break
            s0 = s1
            
        r_all_episodes.append(R)
    print('Value', np.max(Q, axis=-1))
    pi = np.argmax(Q, axis=-1)
    # print('Value', V)
    print('Policy:', pi)
    # plot_return_vs_episodes(r_all_episodes, smooth=10)
    print(np.round(Q,3))
    return pi, Q
    
if __name__ == '__main__':
    from GridWorld import *
    import datetime
    time =  datetime.datetime.now()
    time = time.strftime("%f")
    training_steps = 100000
    repeat = 100
    n_particle = 1
    stepsize = 0.03
    r = []
    random.seed(10)
    env = GridWorld((3,4), obstacles=True)

    V = np.ones(env.n_cell)/env.observation_space.n
    Q = np.ones(shape=(env.n_cell + (env.action_space.n, )))/env.observation_space.n / env.action_space.n
    print('Q table with shape', Q.shape)
    # S = range(env.n_cell)
    A = range(env.action_space.n)
    # DynamicProgramming(V, A, S, env)
    # pi = QLearning(Q, env, n_episodes=20, horizon=150)
    S = []
    for i in range(env.n_cell[0]):
        for j in range(env.n_cell[1]):
            S.append((i,j)) 
    DynamicProgramming(Q, A, S, env)