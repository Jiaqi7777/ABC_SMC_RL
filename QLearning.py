import numpy as np
from utils import *
from parameter import *
def DynamicProgramming(Q, A, S, env, thresh = 1e-4, gamma=0.95):
    loop = 0
    delta = thresh + 0.01
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
    
    V = np.max(Q, axis=-1)
    print('Q', np.round(Q, 3), delta)
    print(f'Converged with loop {loop}')
    print('Value', V)
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
            r_optimal = V_star[s0]#env.R[tuple(env.P[s0 + (int(pi_star[s0]), )])]
            R = r_optimal - r + R * gamma
            Q[s0][a] = r + gamma * max(Q[s1])
            if done:
                print('Done in ', t + 1, 'steps')
                break
            s0 = s1
            
        r_all_episodes_qlearning.append(R)
    V = np.max(Q, axis=-1)
    print('Value', V)
    pi = np.argmax(Q, axis=-1)
    # print('Value', V)
    print('Policy:', pi)
    # plot_return_vs_episodes(r_all_episodes_qlearning, smooth=10, show=True)
    print(np.round(Q,3))
    return pi, Q, V, r_all_episodes_qlearning
    
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
    random.seed(seed)
    np.random.seed(seed)
    
    env = GridWorld((3,4), obstacles=True)
    S = []
    for i in range(env.n_cell[0]):
        for j in range(env.n_cell[1]):
             S.append((i,j)) 
    V = np.ones(env.n_cell)/env.observation_space.n
    Q = np.ones(shape=(env.n_cell + (env.action_space.n, )))/env.observation_space.n / env.action_space.n
    print('Q table with shape', Q.shape)
    r_all_repeat_qlearning = []
    # S = range(env.n_cell)
    A = range(env.action_space.n)
    pi_star, Q_star, V_star = DynamicProgramming(Q, A, S, env)
    for repeat in range(repeat_experiment):
        env.reset()
        V = np.ones(env.n_cell)/env.observation_space.n
        Q = np.ones(shape=(env.n_cell + (env.action_space.n, )))/env.observation_space.n / env.action_space.n
        print('Q table with shape', Q.shape)
        # S = range(env.n_cell)
        A = range(env.action_space.n)
        # DynamicProgramming(V, A, S, env)
        pi, Q, V, r_, = QLearning(Q, env, n_episodes=episodes, horizon=horizon)
        r_all_repeat_qlearning.append(r_)
        # S = []
        # for i in range(env.n_cell[0]):
        #     for j in range(env.n_cell[1]):
        #         S.append((i,j)) 
        # DynamicProgramming(Q, A, S, env)
    with open(f'Models/MCMC/q_learning_chains_T{training_steps}_{time}.npy', 'wb') as f:
        np.save(f, r_all_repeat_qlearning)
        print('model saved at', f'Models/MCMC/q_learning_chains_T{training_steps}_{time}.npy')