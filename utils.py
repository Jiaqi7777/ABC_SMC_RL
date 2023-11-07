import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from parameter import *

def argmaxs(arr):
    mask = arr == arr.max()
    return random.choice(np.array(range(len(arr)))[mask])

def torch_max_0(tensor):
    if torch.isnan(tensor):
        return -torch.tensor(1e9)
    return torch.minimum(torch.zeros(tensor.size()),tensor)

def is_diagonal(matrix):
    return torch.all(torch.eq(matrix, torch.diag(torch.diagonal(matrix))))

def expand_dims(arr, dim=2):
    arr = np.array(arr)
    if len(arr.shape) >= dim:
        return arr
    return np.expand_dims(arr, axis=0)

def submatrix_shape(table, indices):
    # Determine the number of rows and columns
    num_rows = len(table)
    num_columns = len(table[0]) if num_rows > 0 else 0

    if not indices:
        # If the indices list is empty, return the shape of the original table
        return (num_rows, num_columns)

    # Extract the sub-matrix using the indices
    submatrix = [table[i][j] for i, j, k in indices]

    # Determine the number of rows and columns in the sub-matrix
    sub_num_rows = len(set(i for i, _, _ in indices))
    sub_num_columns = len(set(j for _, j, _ in indices))

    return (sub_num_rows, sub_num_columns)

def plot_2d(X, Y, Z, env_name='GridWorld', action_dim=4, title=None, xlabel='s0', ylabel='s1', zlabel='Value', show=False, additional_info=[], save=False, figure_path = '../Figures/MCMC/'):
    Z = expand_dims(Z, dim=3)
    arrows = {2: (1, 0), 0: (-1, 0), 1: (0,1), 3: (0, -1)} if action_dim == 4 else {0: (1, 1), 1: (1, -1)}#{0: (0, 1), 1: (0, -1)}
    if env_name == 'DeepSea':
        arrows = {0: (1, 1), 1: (1, -1)}
    scale = 0.25
    size = 1
    fig, ax = plt.subplots(figsize=(Z.shape[0] * size, Z.shape[1] * size))
    if additional_info != []:
        additional_info = expand_dims(additional_info)
        im = ax.imshow(additional_info)
        cbar = fig.colorbar(im)
    if env_name == 'GridWorld':
        ax.set_xticks(np.arange(len(Y)), labels=Y) #Y is the column number
        ax.set_yticks(np.arange(len(X)), labels=X) #X is the row number
    elif env_name == 'Maze':
        ax.set_yticks(np.arange(len(X[0])), labels=X[0]) 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for i in range(len(X)):
        for j in range(len(Y)):
            for k in range(action_dim):
                scale = Z[i, j, k] / 3 + 1e-4
            # text = ax.text(j, i, Z[i, j], ha="center", va="center", color="w")
                ax.plot(j, i, marker='o', markersize=8, color='red')
                ax.arrow(j, i, scale*arrows[k][1], scale*arrows[k][0], head_width=0.05)

    ax.set_title(title)
    fig.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig(f'{figure_path+title}.png')
        plt.clf()
        print('figure saved at ', f'{figure_path+title}.png')
    
def plot_3d(X, Y, Z, title=None, xlabel='s0', ylabel='s1', zlabel='Value', show=False):
    if len(Z.shape) == 1:
        Z = np.expand_dims(Z, axis=0)
    figure_path = '../Figures/'
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
            cmap='viridis', edgecolor='none')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.view_init(60, 35)
    plt.title(title)
    plt.show()
    if show:
        plt.show()
    plt.savefig(f'{figure_path+title}.png')
    print('figure saved at ', f'{figure_path+title}.png')
    plt.clf()
    
def plot_obs(obs, env, env_name=ENV_NAME, title=None, xlabel='s0', ylabel='s1', zlabel='Path', show=False, additional_info=[], save=False, figure_path = '../Figures/MCMC/'):
    print('plot obs......')
    action_dim = env.action_space.n
    arrows = {2: (1, 0), 0: (-1, 0), 1: (0,1), 3: (0, -1)} if action_dim == 4 else {0: (1, 1), 1: (1, -1)}#{0: (0, 1), 1: (0, -1)}
    if env_name == 'DeepSea':
        arrows = {0: (1, 1), 1: (1, -1)}
    scale = 0.25
    size = 1
    fig, ax = plt.subplots(figsize=(env.n_cell[0] * size, env.n_cell[1] * size))
    if additional_info != []:
        additional_info = expand_dims(additional_info)
        im = ax.imshow(additional_info)
        cbar = fig.colorbar(im)
    if env_name == 'GridWorld':
        ax.set_xticks(np.arange(env.n_cell[1])) 
        ax.set_yticks(np.arange(env.n_cell[0])) 
    elif env_name == 'Maze':
        ax.set_yticks(np.arange(env.n_cell[1])) 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cmap = plt.get_cmap('twilight')
    e = 0
    for s0, s1, a, r, d in np.array(list(obs._buffers.values())).T:
        ax.plot(s0[1], s0[0], marker='o', markersize=8, color=cmap(e))
        ax.arrow(s0[1], s0[0], s1[1]-s0[1], s1[0]-s0[0], head_width=0.2, fc=cmap(e), ec=cmap(e))
        if d:
            e+=0.1
            e = e % EPISODES

    ax.set_title(title)
    fig.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig(f'{figure_path+title}.png')
        plt.clf()
        print('figure saved at ', f'{figure_path+title}.png')
        
def plot_qtable(q, title=None):
    if title is None:
        title = 'Leanrt Mean of Q table'
    fig, axes = plt.subplots(1, q.shape[-1], figsize=(12, 6))  # Adjust the number of subplots and figsize as needed
    for a in range(q.shape[-1]):
        im = axes[a].imshow(q[:, :, a])
        axes[a].set_title(f'{title} action {a}')
        cbar = plt.colorbar(im)
    
    plt.tight_layout()

def replace_line(file_path, variable, new_value):
    with open(file_path) as f:
        lines = f.readlines()
    f.close()
    with open(file_path, 'w') as f:
        for line in lines:
            if variable == line.split(' = ')[0]:
                line = f'{variable} = {new_value}\n'
                print("Replace to ", line)
            f.write(line)
    f.close()
    
def plot_return_for_epsiodes(r_all_episodes, figure_path='../Figures/', show=False):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(r_all_episodes)))
    for t, i in enumerate(r_all_episodes):
        plt.plot(i, label=f'Time {t}', color=colors[t])
    plt.legend()
    plt.xlabel('timesteps')
    plt.ylabel('Return')
    title = 'Return for episodes'
    plt.title(title)
    plt.savefig(f'{figure_path+title}.png')
    print('figure saved at ', f'{figure_path+title}.png')
    if show:
        plt.show()
    plt.clf()
    
def plot_return_vs_episodes(r_all_episodes, smooth=1, figure_path='../Figures/', show=False, save=False):
    r_all_episodes = np.convolve(np.array(r_all_episodes), np.ones(smooth)/smooth, mode='valid')
    plt.plot(r_all_episodes)
    plt.xlabel('episodes')
    plt.ylabel('Return')
    title = 'Return for each episodes'
    plt.title(title)
    if save:
        plt.savefig(f'{figure_path+title}.png')
        print('figure saved at ', f'{figure_path+title}.png')
    if show:
        plt.show()
    plt.clf()
    
def plot_return_vs_episodes_repeat(r_all_episodes_repeat, figure_path='../Figures/', show=False, title='', save=False):
    N = len(r_all_episodes_repeat)
    r_mean = np.mean(r_all_episodes_repeat, axis=0)
    r_std = np.std(r_all_episodes_repeat, axis=0)
    plt.plot(r_mean, label = f'Mean of the return')
    plt.fill_between(range(len(r_all_episodes_repeat[0])), r_mean-r_std/np.sqrt(N), r_mean+r_std/np.sqrt(N), alpha=0.2)
    plt.xlabel('episodes')
    plt.ylabel('Return')
    plt.title(f'Return for each episodes averaging over {N} random runs')
    if save:
        plt.savefig(f'{figure_path+title}.png')
        print('figure saved at ', f'{figure_path+title}.png')
    if show:
        plt.show()
    plt.clf()
    
def get_outliers(data, threshold=5):
    q = np.percentile(data, [threshold, 100-threshold])
    outliers = np.any((data < q[0]) | (data > q[1]), axis=1)
    return outliers

def compare_r_vs_episodes_repeat(r_1, r_2, figure_path='../Figures/', show=True, title='', save=False, smooth=1, datatype='Return'):
    r_1 = np.apply_along_axis(lambda m: np.convolve(m, np.ones(smooth)/smooth, mode='valid'), axis=-1, arr=np.array(r_1))
    r_2 = np.apply_along_axis(lambda m: np.convolve(m, np.ones(smooth)/smooth, mode='valid'), axis=-1, arr=np.array(r_2))

    N = len(r_1)
    r_mean = np.mean(r_1, axis=0)
    r_std = np.std(r_1, axis=0)
    plt.plot(r_mean, label = 'ts sampling')
    plt.fill_between(range(len(r_1[0])), r_mean-r_std/np.sqrt(N), r_mean+r_std/np.sqrt(N), alpha=0.2)
    plt.xlabel('episodes')
    plt.ylabel(datatype)
    plt.title(f'{datatype} for each episodes averaging over {N} random runs, smoothed over {smooth} steps')
    r_mean = np.mean(r_2, axis=0)
    r_std = np.std(r_2, axis=0)
    plt.plot(r_mean, label = 'epsilon greedy Q-Learning')
    plt.fill_between(range(len(r_2[0])), r_mean-r_std/np.sqrt(N), r_mean+r_std/np.sqrt(N), alpha=0.2)
    plt.legend()
    #title=f'ComparisonRegret{ENV_NAME}_Epsilon{EPSILON}_T{training_steps}_{time}'
    if save:
        plt.savefig(f'{figure_path+title}.png', bbox_inches='tight')
        print('figure saved at ', f'{figure_path+title}.png')
    
    if show:
        plt.show()
    plt.clf()
    
def save_results(results, folder, stochastic, episode, training_steps, greedy, epsilon, initial_stepsize, decreasing_factor, time, m_z, repeat, episodic=False):
    Episodes = f'Episode{episode}_' if episodic else ''
    file_path = f'../{folder}/MCMC/T{training_steps}_Repeat{repeat}_{Episodes}Sto{stochastic}_M{m_z}_Gdy{greedy}_Ep{epsilon}_Stp{initial_stepsize}_Dcrs{decreasing_factor}_{time}.npy'
    with open(file_path, 'wb') as f:
        np.save(f, results)
    print(f'{Episodes}{folder} for repeat {repeat} saved at', file_path)

def plot_repeat(data, x, smooth=5, figure_path='../Figures/', labels=[], label='', show=True, save=False, xlabel='', ylabel='MSE', title=None):
    data = np.swapaxes(np.array(data), 0, 1)
    if len(labels) != len(data):
        labels=[''] * len(data)
    for d, l in zip(data, labels):
        smoothed_d = np.apply_along_axis(lambda m: np.convolve(m, np.ones(smooth) / smooth, mode='valid'), axis=-1, arr=np.array(d))
        N = len(smoothed_d)
        d_mean = np.mean(smoothed_d, axis=0)
        d_std = np.std(smoothed_d, axis=0)
        plt.plot(d_mean, label = f'{label} = {l}')
        plt.fill_between(range(len(smoothed_d[0])), d_mean-d_std / np.sqrt(N), d_mean + d_std / np.sqrt(N), alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} averaging over {N} random runs, smoothed over {smooth} steps' if title is None else title)
    plt.xticks(range(len(smoothed_d[0])), x)
    plt.legend()
    if save:
        plt.savefig(f'{figure_path+title}.png', bbox_inches='tight')
        print('figure saved at ', f'{figure_path+title}.png')
    
    if show:
        plt.show()
    plt.clf()
    
def plot_block_accpt_prob(mcmc, block_accept, smooth=None, save=False, show=True, figure_path='../Figures/'):
    cmap = plt.get_cmap("coolwarm")
    block_accept = np.array(block_accept)
    var = 'z'
    if smooth is None:
        smooth = len(block_accept[0]) // 10 + 1
    title = f'mean of the accept prob with blocksize {mcmc.block_size[var]} and M={M_Z}'
    print('Mean of each blocks', np.mean(block_accept.T[:-1], axis=(1, 2)))
    plt.plot(np.mean(block_accept.T[:-1], axis=(1, 2)))
    plt.title(title)
    plt.xlabel('block no.')
    if show:
        plt.show()
    if save:
        plt.savefig(f'{figure_path+title}.png', bbox_inches='tight')
        print('figure saved at ', f'{figure_path+title}.png')
    title = f'M={M_Z}, smoothed over {smooth} steps with std over {len(block_accept)} repeatition'
    for i, acc in enumerate(np.array(block_accept).T[:]):
        acc = np.apply_along_axis(lambda m: np.convolve(m, np.ones(smooth) / smooth, mode='valid'), axis=0, arr=acc)
        acc_mean = np.mean(acc, axis=-1)
        acc_std = np.std(acc, axis=-1)
        plt.plot(acc_mean, label=f'block {i}', color=cmap(i / block_accept.shape[-1]))
        plt.fill_between(range(len(acc_mean)), acc_mean-acc_std/np.sqrt(len(acc)), acc_mean+acc_std/np.sqrt(len(acc)), alpha=0.2)
        
        
    plt.legend(loc='upper right', bbox_to_anchor=(1.3,1))
    plt.title(title)
    if show:
        plt.show()
    if save:
        plt.savefig(f'{figure_path+title}.png', bbox_inches='tight')
        print('figure saved at ', f'{figure_path+title}.png')