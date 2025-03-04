import random
import numpy as np
import pickle
from functools import partial
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import torch
from parameter import *
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 24,    # Font size for axis labels
    'axes.titlesize': 16,    # Font size for titles
    'legend.fontsize': 12,   # Font size for legend
    'xtick.labelsize': 12,   # Font size for x-tick labels
    'ytick.labelsize': 12,   # Font size for y-tick labels
    'figure.titlesize': 18,   # Font size for figure title
    'text.usetex': True,  
})
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

def plot_2d(X, Y, Z, env_name='GridWorld', action_dim=4, title=None, xlabel='s0', ylabel='s1', zlabel='Value', show=SHOW, additional_info=[], save=False, figure_path = '../Figures/MCMC/'):
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
    
def plot_3d(X, Y, Z, title=None, xlabel='s0', ylabel='s1', zlabel='Value', show=SHOW):
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
    if show:
        plt.show()
    plt.savefig(f'{figure_path+title}.png')
    print('figure saved at ', f'{figure_path+title}.png')
    plt.clf()
    
def plot_obs(obs, env, env_name=ENV_NAME, title=None, xlabel='s0', ylabel='s1', zlabel='Path', show=SHOW, additional_info=[], save=False, figure_path = '../Figures/MCMC/'):
    print('plot obs......')
    action_dim = env.action_space.n
    # arrows = {2: (1, 0), 0: (-1, 0), 1: (0,1), 3: (0, -1)} if action_dim == 4 else {0: (1, 1), 1: (1, -1)}#{0: (0, 1), 1: (0, -1)}
    # if env_name == 'DeepSea':
    #     arrows = {0: (1, 1), 1: (1, -1)}
    # scale = 0.25
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
    b = 0
    if len(obs._buffers['done']) != 0:
        for s0, s1, _, _, _ in np.array(list(obs._buffers.values())).T:
            ax.plot(s0[1], s0[0], marker='o', markersize=8, color=cmap(b))
            ax.arrow(s0[1], s0[0], s1[1]-s0[1], s1[0]-s0[0], head_width=0.2, fc=cmap(b), ec=cmap(b))
    b = 0.7
    if len(obs._new_data_buffers['done']) != 0:
        for s0, s1, _, _, _ in np.array(list(obs._new_data_buffers.values())).T:
            ax.plot(s0[1], s0[0], marker='o', markersize=8, color=cmap(b))
            ax.arrow(s0[1], s0[0], s1[1]-s0[1], s1[0]-s0[0], head_width=0.2, fc=cmap(b), ec=cmap(b))

    ax.set_title(title)
    fig.tight_layout()
    if save:
        plt.savefig(f'{figure_path+title}.png')
        plt.close()
        print('figure saved at ', f'{figure_path+title}.png')
    if show:
        plt.show()
        
def plot_qtable(q, title=None, save=False, figure_path=None):
    if title is None:
        title = 'Leanrt Mean of Q table'
    norm = Normalize(vmin=torch.min(q), vmax=torch.max(q))
    fig, axes = plt.subplots(1, q.shape[-1], figsize=(12, 6))  # Adjust the number of subplots and figsize as needed
    for a in range(q.shape[-1]):
        im = axes[a].imshow(q[:, :, a], norm=norm)
        axes[a].set_title(f'{title} action {a}')
        cbar = plt.colorbar(im)
    
    plt.tight_layout()
    if save:
        plt.savefig(f'{figure_path+title}.png')
        plt.close()
        print('figure saved at ', f'{figure_path+title}.png')

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
    
def plot_return_for_epsiodes(r_all_episodes, figure_path='../Figures/', show=SHOW):
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
    
def plot_return_vs_episodes(r_all_episodes, repeat, smooth=1, figure_path='../Figures/', show=False, save=False):
    r_all_episodes = np.convolve(np.array(r_all_episodes), np.ones(smooth)/smooth, mode='valid')
    plt.plot(r_all_episodes)
    plt.xlabel('episodes')
    plt.ylabel('Return')
    title = f'Return vs episodes at repeat {repeat}'
    plt.title(title)
    if save:
        plt.savefig(f'{figure_path+title}.png')
        print('figure saved at ', f'{figure_path+title}.png')
    if show:
        plt.show()
    plt.clf()
    
def plot_return_vs_episodes_repeat(r_all_episodes_repeat, figure_path='../Figures/', show=SHOW, title='', save=False, smooth=1):
    N = len(r_all_episodes_repeat)
    smooth = min(smooth, len(r_all_episodes_repeat[0]))
    smooth_L = len(r_all_episodes_repeat[0]) - smooth + 1
    r_all_episodes_repeat_smooth = np.zeros((N, smooth_L))
    for i in range(N):
        r_all_episodes_repeat_smooth[i] = np.convolve(np.array(r_all_episodes_repeat[i]), np.ones(smooth)/smooth, mode='valid')
    r_mean = np.mean(r_all_episodes_repeat_smooth, axis=0)
    r_std = np.std(r_all_episodes_repeat_smooth, axis=0)
    plt.plot(r_mean, label = f'Mean of the return')
    plt.fill_between(range(len(r_all_episodes_repeat_smooth[0])), r_mean-r_std/np.sqrt(N), r_mean+r_std/np.sqrt(N), alpha=0.2)
    plt.xlabel('episodes')
    plt.ylabel('Return')
    title = f'Return for each episodes averaging over {N} random runs' if title == '' else title
    plt.title(title)
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
    
def save_results(results, folder, training_steps, greedy, epsilon, time, m_z, dir=None, experiment='MCMC', episode=None, repeat=None, stochastic=False, episodic=False, prior_sigma=PRIOR_SIGMA):
    # dir = f'../{folder}/{experiment}/{time}'
    dir = dir if dir is not None else f'../{folder}/{experiment}/{time}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    Episodes = f'Episode{episode}_' if episodic else ''
    # repeat = f'Repeat{repeat}_' if episodic else ''
    file_path = f'{dir}{folder}_T{training_steps}_{Episodes}Sto{stochastic}_M{m_z}_Gdy{greedy}_Sigma{prior_sigma}.pt'
    torch.save(results, file_path)
    # print(f'{Episodes}{folder} for repeat {repeat} saved at', file_path)

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
        
def plot_save(data, title='', figure_path='', save=False, episode='', repeat='', show=False):
    plt.plot(data, '-o')
    plt.title(f'{title} Episode{episode} Repeat{repeat}')
    if save:
        plt.savefig(f'{figure_path}E{episode}{title}.png', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def save_faulty_ess(epsilon_0, smc_samples, weights, generate_weights_fn):
    ess_dict = {"smc_samples": smc_samples, "weights": weights, "epsilon_0": epsilon_0, 'generate_weights_fn':generate_weights_fn}

    # os.makedirs(f'../SMC/FaultyBisect/{epsilon_0}')
    with open(f'../SMC/FaultyBisect/{epsilon_0}/ess_info.pkl', "wb") as ess_file:
        pickle.dump(ess_dict, ess_file)

def ess_(epsilon, Model, weights, smc_samples, generate_weights_fn, epsilon_0, ess_fn):
    new_weights = generate_weights_fn(epsilon=epsilon, weights=weights, Model=Model, smc_samples=smc_samples, epsilon_0=epsilon_0)
    ess = ess_fn(new_weights)
    return ess

def plot_ess(e_l, ess_l, epsilon_0, ess, alpha, new_epsilon, i=-1, save=False, loop_num=0, episode='', repeat='', figure_path='', show=False):
    plt.hlines(xmin=0, xmax=e_l[i], y=1, colors='green', label='1', linestyle='--', alpha=0.5)
    plt.hlines(xmin=e_l[0], xmax=e_l[i], y=ess, colors='orange', label='original ess', alpha=0.4)
    plt.hlines(xmin=e_l[0], xmax=e_l[i], y=max(1, alpha * ess), colors='purple', label='target ess', alpha=0.4)
    plt.plot(e_l[:i], ess_l[:i])
    plt.axvline(x=epsilon_0, color='red', linestyle='-', label='epsilon_0', alpha=0.2)
    plt.axvline(x=new_epsilon, color='blue', linestyle='-.', label='new epsilon', alpha=0.5)
    plt.legend()
    plt.title(f'epsilon={epsilon_0}')
    if save:
        plt.savefig(f'{figure_path}E{episode}essPlotLoop{loop_num}Epsl{epsilon_0}.png', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    
def load_return(depth, adaptive=False):
    if adaptive:
        file_path = f'../SMC/0adaptive/{depth}'
    else:
        file_path = f'../SMC/{depth}'
    loaded_files = []

    # Loop through each item in the main directory (file_path)
    try:
        for date in os.listdir(file_path):
            if not date.startswith('R'):
                date_file_path = f'{file_path}/{date}'
            else:
                date_file_path = file_path
            for folder_name in os.listdir(date_file_path):
                folder_path = os.path.join(date_file_path, folder_name)
                # Check if it is a directory and its name starts with 'R' followed by a number
                if os.path.isdir(folder_path) and folder_name.startswith('R') and folder_name[1:].isdigit():
                    # Loop through files in the Rn directory
                    for file_name in os.listdir(folder_path):
                        if file_name.startswith('Returns') and file_name.endswith('.pt'):
                            return_path = os.path.join(folder_path, file_name)

                            # Load the .pt file using torch.load and append it to the list
                            loaded_data = torch.load(return_path)
                            loaded_files.append(loaded_data)
                            break
    except FileNotFoundError:
        return False
    if loaded_files == []:
        return False
    return loaded_files

def display_smc_results_notebook(samples, n_0=0, n_1=None, smooth=1, Q_star=None, figure_path=None, save=False, episode='', repeat='', show=False, cell_n=2, cell_idx=0, cell_idx_x=None, ylim=4):
    figure_name = f'E{episode}R{repeat}SMCSamples'
    len_sample = len(samples) - smooth + 1
    subsample_int = 1
    samples = np.apply_along_axis(lambda m: np.convolve(m, np.ones(smooth)/smooth, mode='valid'), axis=0, arr=samples)
    samples = samples[n_0:n_1:subsample_int]
    dim = samples.shape
    print(dim)
    means = np.mean(samples, axis=1)
#     means = np.apply_along_axis(lambda m: np.convolve(m, np.ones(smooth)/smooth, mode='valid'), axis=0, arr=means)
    stds = np.std(samples, axis=1)
#     stds = np.apply_along_axis(lambda m: np.convolve(m, np.ones(smooth)/smooth, mode='valid'), axis=0, arr=stds)
    upbd = means + stds
    lwbd = means - stds
    cell = (dim[2] - 1 )//cell_n
    if cell_idx_x is None:
        cell_idx_x = cell_idx
#     for a in [0, 1]:
#     print('action', a)
    n_1 = len_sample if n_1 is None else min(len_sample, n_1)
    fig2, ax2 = plt.subplots(cell, cell, figsize=(cell*4, cell*4))
    for i in range(cell * cell_idx, cell * (cell_idx + 1)):
        horizon_lim = i + 1 if cell_idx == cell_idx_x else cell * (cell_idx_x + 1)
        for k in range(cell * (cell_idx_x), horizon_lim):
            for a in [0, 1]:
                ax2[i%cell, k%cell].set_ylim(*ylim)#set_ylim(np.min(samples) * 0.8, np.max(samples)* 0.8)
                ax2[i%cell, k%cell].plot(range(n_0, n_1)[::subsample_int], means[:, i, k, a], label=f'Action {a}')
                ax2[i%cell, k%cell].fill_between(range(n_0, n_1)[::subsample_int], lwbd[:, i, k, a], upbd[:, i, k, a], alpha=0.5)
#                 for j in range(dim[1]//2):
                    # ax2[i,k].scatter(range(dim[0]), samples[:, j, i, k, 0], label=f"right {j}", alpha=smc._weights_history[:,j])
                    # ax2[i,k].plot(range(dim[0]), samples[:, j, i, k, 0], linestyle='--')
#                 ax2[i%cell, k%cell].scatter(range(n, len_sample)[::subsample_int], samples[:, j, i, k, a])
#                     ax2[i%cell, k%cell].plot(range(n, len_sample)[::subsample_int], samples[:, j, i, k, a], linestyle='-', alpha=0.6)
            if Q_star is not None:
                # Assuming Q_star[i, k, :] has two values and you want different labels for each
                ax2[i % cell, k % cell].hlines(Q_star[i, k, 0], xmin=n_0, xmax=n_1, linestyle="--", label="Ground Truth Action 0", color="Blue")
                ax2[i % cell, k % cell].hlines(Q_star[i, k, 1], xmin=n_0, xmax=n_1, linestyle="--", label="Ground Truth Action 1", color="Orange")
#                 ax2[i,k].hlines(Q_star[i, k, 1], xmin=n, xmax=len_sample-1, linestyle="--", label="true left", color="purple")
            ax2[i%cell, k%cell].set_title([i, k])
    if cell_idx == cell_idx_x:
        for i in range(cell * 0, cell * 1):
            for k in range(i+1, cell):
                for spine in ax2[i, k].spines.values():
                    spine.set_visible(False)
                ax2[i, k].set_xticks([])  # Remove x-axis ticks
                ax2[i, k].set_yticks([])  # Remove y-axis ticks
                ax2[i, k].set_xlabel('')  # Remove x-axis label
                ax2[i, k].set_ylabel('')  
    handles, labels = ax2[i%cell, k%cell].get_legend_handles_labels()
    ax2[i%cell, k%cell].legend(handles, labels, bbox_to_anchor=(0.9, 1.5), loc='right')
    fig2.text(0.5, 0.08, r'\textbf{Episodes}', ha='center', va='center', fontsize=20, fontweight='bold')
    fig2.text(0.1, 0.5, r'\textbf{Particle Values }$\mathbf{\theta}$', ha='center', va='center', rotation='vertical', fontsize=20, fontweight='bold')
#     plt.tight_layout(rect=[0, 1, 1, 0.95])  # Adjust for suptitle and axis labels


    plt.show()

    
def first_index_exceeding_average(arr_of_lists, threshold=0.5):
    # Check if the input is a single list
    if isinstance(arr_of_lists, list) and len(arr_of_lists) > 0 and not isinstance(arr_of_lists[0], list):
        arr = arr_of_lists  # Treat as a single list
        cumulative_sum = 0

        for i in range(len(arr)):
            cumulative_sum += arr[i]
            average = cumulative_sum / (i + 1)
            if average > threshold:
                return i
        
        return -1  # If no index exceeds the threshold

    # If input is a list of lists
    if not arr_of_lists or not all(arr_of_lists):  # Check for empty or invalid inputs
        return -1

    total_indices = 0
    total_runs = len(arr_of_lists)

    for arr in arr_of_lists:
        cumulative_sum = 0
        found_index = -1

        for i in range(len(arr)):
            cumulative_sum += arr[i]
            average = cumulative_sum / (i + 1)
            if average > threshold:
                found_index = i
                break
        
        # If no index found, count it as the end of the array
        if found_index == -1:
            total_indices += len(arr)  # Use length of array if no index exceeds the threshold
        else:
            total_indices += found_index

    # Return the average of the indices over the runs
    return total_indices / total_runs

def calculate_regret():
    all_depth_return = []
    all_depth_success = []
    all_depth_return_adaptive = []
    all_depth_success_adaptive = []
    d_l = [5, 7, 10, 12, 14, 15, 20, 25, 30, 40]
    d_l_fixed = []
    d_l_a = []
    for d in d_l:
        r_d = load_return(d)
        if r_d:
            d_l_fixed.append(d)
            all_depth_return.append(r_d)
            all_depth_success.append(first_index_exceeding_average(r_d))
        r_d_a = load_return(d, True)
        if r_d_a:
            d_l_a.append(d)
            all_depth_return_adaptive.append(r_d_a)
            all_depth_success_adaptive.append(first_index_exceeding_average(r_d_a))
    return all_depth_return, all_depth_success, all_depth_return_adaptive, all_depth_success_adaptive, d_l_fixed, d_l_a

def plot_log(d_l_fixed, all_depth_success, d_l_a, all_depth_success_adaptive):
    plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 14,    # Font size for axis labels
    'axes.titlesize': 16,    # Font size for titles
    'legend.fontsize': 12,   # Font size for legend
    'xtick.labelsize': 12,   # Font size for x-tick labels
    'ytick.labelsize': 12,   # Font size for y-tick labels
    'figure.titlesize': 18,   # Font size for figure title
        'text.usetex': True,  
    })
    coefficients = np.polyfit(np.log10(d_l_fixed), np.log10(all_depth_success), 1)
    coefficients_a = np.polyfit(np.log10(d_l_a), np.log10(all_depth_success_adaptive), 1)
    # coefficients[-3] = 0
    # 3. Get the fitted values using the polynomial
    # polynomial = np.poly1d(coefficients)

    plt.scatter(np.log10(d_l_fixed), np.log10(all_depth_success), label='ABRL - Non-Adaptive')
    plt.scatter(np.log10(d_l_a), np.log10(all_depth_success_adaptive), label='ABRL - Adaptive')

    x_range = np.linspace(np.min(np.log10(d_l_fixed)), np.max(np.log10(d_l_fixed)), num=100)
    x_range_a = np.linspace(np.min(np.log10(d_l_a)), np.max(np.log10(d_l_a)), num=100)
    # plt.plot(x_range, x_range*20-12.6, '--', label='slope 20', linewidth=0.5)
    plt.plot(x_range, x_range*coefficients[0] + coefficients[1], '--', label=f'slope {np.round(coefficients[0], 0)}', linewidth=0.5)
    plt.plot(x_range_a, x_range_a*coefficients_a[0] + coefficients_a[1], '--', label=f'slope {np.round(coefficients_a[0], 0)}', linewidth=0.5)

    # plt.plot(x_range, x_range*4-2, '--', label='slope 4', linewidth=0.5)
    # plt.plot(x_range, x_range*6-3.2, '--', label='slope 6', linewidth=0.5)
    # Add grid lines
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # Add labels and title
    plt.xlabel(r'\textbf{log}$_{10}$\textbf{N}')#, fontfamily='serif', fontstyle='italic')
    plt.ylabel(r'\textbf{log}$_{10}$\textbf{T}')#, fontfamily='serif', fontstyle='italic')

    # plt.title('Learning time by when the average regret drops below 0.5(Log scale)')
    plt.legend()
    plt.savefig('../SMC/LearningT.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def append_rewards(r, length=None):
    assert hasattr(r, '__len__')
    if len(r) > 1 :
        out_r = []
        
        if length is None:
            length = len(r[0])
            for ri in r[1:]:
                if len(ri) > length:
                    length = len(ri)
        for ri in r:
            out_r.append(append_rewards([ri], length=length))
        return out_r
    if length is None:
        print('No change')
        return r
    ri = r[0]
    for _ in range(len(ri), length):
        ri.append(ri[-1])
    return ri

def plot_return(input_r):
    if len(input_r) >= 2:
        r_m = np.mean(input_r, axis=0)
        r_std = np.std(input_r, axis=0) * 0.5
        plt.plot(r_m, label='Average Return')
        plt.fill_between(range(len(input_r[0])), r_m-r_std, r_m + r_std, alpha=0.2, label='0.5 std')
        plt.title('Average Return vs Episode')
        plt.legend()
        plt.show()
        
def plot_particle(particle_feature, title='', ref=None):
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    for i in range(len(particle_feature[0])):
        plt.plot(particle_feature[:, i], label=i)
        if ref is not None:
            plt.vlines(ref, ymin=torch.min(particle_feature), ymax=torch.max(particle_feature), color='r')
    plt.legend()
    plt.title(title)
    plt.show()
    
def last_match_index(lst, target):
    """
    Returns the index (counting from the last) of the first occurrence of 'target' in the list.
    
    Parameters:
    - lst (list): The list to search in.
    - target (any): The value to find.
    
    Returns:
    - int: The index from the end (1-based), or -1 if not found.
    """
    for i, val in enumerate(reversed(lst)):
        if val == target:
            return i  # Index counting from the end
    return -1  # Not found

def last_mismatch_index_value(lst, target):
    """
    Returns the index (counting from the last) of the first occurrence of 'target' in the list.
    
    Parameters:
    - lst (list): The list to search in.
    - target (any): The value to find.
    
    Returns:
    - int: The index from the end (1-based), or -1 if not found.
    """
    for i, val in enumerate(reversed(lst)):
        if val != target:
            print(lst, i, val, target)
            return i, val  # Index counting from the end
    return -1, None  # Not found

def find_match_segments(arr1, arr2):
    if len(arr1) != len(arr2):
        raise ValueError("Both arrays must have the same length")
    
    equal_segments = []
    not_equal_segments = []
    
    i = 0
    while i < len(arr1):
        if arr1[i] == arr2[i]:
            start = i
            while i < len(arr1) and arr1[i] == arr2[i]:
                i += 1
            equal_segments.append((start, i))
        else:
            start = i
            while i < len(arr1) and arr1[i] != arr2[i]:
                i += 1
            not_equal_segments.append((max(start - 1, 0), i + 1))
    
    return equal_segments, not_equal_segments

def normalize_and_plot(arr, segments, normalize=False):
    """
    Normalizes values between sharp increases and their match points and plots the result.
    
    Parameters:
    - arr: 1D numpy array
    - segments: List of (start, end) tuples where normalization is applied.
    """
    arr = np.array(arr, dtype=float)
    normalized_arr = np.full_like(arr, np.nan)  # Initialize with NaNs for spacing

    # Normalize and store only the valid segments
    for start, end in segments:
        segment = arr[start:end]
        if normalize and len(segment) >= 1:
#             segment -= segment[-1] 
            segment /= segment[0]# Normalize by the first value in the segment
#         segment = arr[start:end]/arr[start]
        normalized_arr[start:end] = segment  # Keep the same horizontal space
    return normalized_arr
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(normalized_arr, marker='o', linestyle='-', color='b', label="Normalized Data")
    plt.xlabel("Index")
    plt.ylabel("Normalized Value")
    plt.title("Normalized Data Between Sharp Increases and Match Points")
    plt.legend()
    plt.show()
    
def calculate_ess(weights):
    weights = np.array(weights)
    ess = np.sum(weights) ** 2 / np.sum(weights ** 2)
    return ess

def plot_metric(time, T=2, plot_range=200):
    prefix = f'../SMC/0adaptive/5/{time}/R0/'
    comp_ep = torch.load(f'{prefix}Epsilon{time}_T{T}_StoFalse_M10_GdyFalse_Sigma4.pt')[:]
    comp_ep_old = torch.load(f'{prefix}EpsilonOld{time}_T{T}_StoFalse_M10_GdyFalse_Sigma4.pt')[:]

    comp_w = torch.load(f'{prefix}Weights{time}_T{T}_StoFalse_M10_GdyFalse_Sigma4.pt')[1:]
    comp_ess = [calculate_ess(w) for w in comp_w]
    # comp_samples = torch.load(f'{prefix}Samples{time}_T{T}_StoFalse_M10_GdyFalse_Sigma4.pt')[1:plot_range]
    comp_be = torch.load(f'{prefix}BellmanErr_T{T}_StoFalse_M10_GdyFalse_Sigma4.pt')[:plot_range]
    comp_ep = comp_ep[:plot_range]
    comp_be = comp_be[:plot_range]
    comp_ess = comp_ess[:plot_range]
    time = range(len(comp_ep))
    metric1 = comp_ep
    # metric2 = comp_samples[:, :,3,0,1]
    metric3 = np.tan(time)

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)

    # Create subplots
    ax1 = plt.subplot(gs[1])
    # ax2 = plt.subplot(gs[3])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[0])
    segments, segments_peak = find_match_segments(comp_ep, comp_ep_old)#find_sharp_increases_and_match_points(comp_ep, 1.1)
    print(segments, segments_peak)

    # Plot each metric on a separate subplot
    ax1.plot(normalize_and_plot(comp_ep, segments), label='Reduced Epsilon')
    ax1.plot(normalize_and_plot(comp_ep, segments_peak), label='Increased Epsilon')
    ax1.plot(comp_ep_old, alpha=0.2, color='green')
    ax1.set_ylabel('Epsilon')
    # ax1.set_xticks([])
    ax1.legend()
    ax1.grid(True)

    # m2 = torch.mean(metric2, axis=1)
    # std2 = torch.std(metric2, axis=1)
    # ax2.plot(time, m2)
    # # ax2.hlines(0, len(comp_samples), 0.99, color='r', alpha=0.5, linestyle='-.', label='Ground Truth')
    # # ax2.fill_between(time, torch.max(metric2, axis=1).values, torch.min(metric2, dim=1).values, alpha=0.5)
    # ax2.fill_between(time, m2-3 * std2, m2+3 * std2, alpha=0.5)
    # ax2.set_ylabel('Particles')
    # ax2.grid(True)
    # ax2.legend()


    be_line = ax4.plot(normalize_and_plot(comp_be, segments, True), label='Bellman Error', color='orange')

    ax42 = ax4.twinx()
    ep_line = ax42.plot(comp_ep[:], alpha=0.5, label='Epsilon')
    for start, end in segments_peak:
        # Add horizontal dotted line connecting start and end
        ax42.hlines(y=comp_ep[start], xmin=start, xmax=end-1, colors='r', linestyles='dotted')

    ax4.set_ylabel('Bellman Error')

    # ax4lines = [be_line[0], ep_line[0]]
    # ax4labels = [line.get_label() for line in ax4lines]
    # ax4.legend(ax4lines, ax4labels, loc='upper right')
    # ax4.grid(True)

    ax3.plot(time, comp_ess, label='ESS')
    ax3.set_ylabel('ESS')
    ax3.plot(comp_ep[:], alpha=0.5, label='Epsilon')
    ax3.legend()
    ax3.grid(True)
    ax3.set_xlabel('Training Steps')
    # Add a title to the figure
    # fig.suptitle('Parallel Plots for Several Metrics')
    # plt.savefig('../Figures/epsilon.png', dpi=300, bbox_inches='tight')
    # Display the plot
    plt.show()
    
def plt_font():
    plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 24,    # Font size for axis labels
    'axes.titlesize': 16,    # Font size for titles
    'legend.fontsize': 12,   # Font size for legend
    'xtick.labelsize': 12,   # Font size for x-tick labels
    'ytick.labelsize': 12,   # Font size for y-tick labels
    'figure.titlesize': 18,   # Font size for figure title
    'text.usetex': True,  
})
