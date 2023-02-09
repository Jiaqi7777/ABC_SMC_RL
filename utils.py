import random
import numpy as np
import matplotlib.pyplot as plt
def argmaxs(arr):
    mask = arr == arr.max()
    return random.choice(np.array(range(len(arr)))[mask])


def plot_3d(X, Y, Z, title=None, xlabel='s0', ylabel='s1', zlabel='Value', show=False):
    if len(Z.shape) == 1:
        Z = np.expand_dims(Z, axis=0)
    figure_path = 'Figures/'
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
            cmap='viridis', edgecolor='none')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)
    plt.show()
    if show:
        plt.show()
    plt.savefig(f'{figure_path+title}.png')
    plt.clf()

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
    
def plot_return_for_epsiodes(r_all_episodes, figure_path='Figures/', show=False):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(r_all_episodes)))
    for t, r in enumerate(r_all_episodes):
        plt.plot(r, label=f'Time {t}', color=colors[t])
    plt.legend()
    plt.xlabel('timesteps')
    plt.ylabel('Return')
    title = 'Return for episodes'
    plt.title(title)
    plt.savefig(f'{figure_path+title}.png')
    if show:
        plt.show()
    plt.clf()
    
def plot_return_vs_episodes(r_all_episodes, smooth=1, figure_path='Figures/', show=False):
    r_all_episodes = np.convolve(np.array(r_all_episodes), np.ones(smooth)/smooth, mode='valid')
    plt.plot(r_all_episodes)
    plt.xlabel('episodes')
    plt.ylabel('Return')
    title = 'Return for each episodes'
    plt.title(title)
    plt.savefig(f'{figure_path+title}.png')
    if show:
        plt.show()
    plt.clf()
    
def plot_return_vs_episodes_repeat(r_all_episodes_repeat, figure_path='Figures/', show=False):
    N = len(r_all_episodes_repeat)
    r_mean = np.mean(r_all_episodes_repeat, axis=0)
    r_std = np.std(r_all_episodes_repeat, axis=0)
    plt.plot(r_mean, label = f'Mean of the return')
    plt.fill_between(range(len(r_all_episodes_repeat[0])), r_mean-r_std/np.sqrt(N), r_mean+r_std/np.sqrt(N), alpha=0.2)
    plt.xlabel('episodes')
    plt.ylabel('Return')
    title = f'Return for each episodes averaging over {N} random runs'
    plt.title(title)
    plt.savefig(f'{figure_path+title}.png')
    if show:
        plt.show()
    plt.clf()