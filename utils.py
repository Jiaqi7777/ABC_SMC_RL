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
    
def plot_return_for_epsiodes(r_all_episodes):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(r_all_episodes)))
    for t, r in enumerate(r_all_episodes):
        plt.plot(r, label=f'Time {t}', color=colors[t])
    plt.legend()
    plt.xlabel('timesteps')
    plt.ylabel('Return')
    plt.title('Return for episodes')
    plt.show()
    
def plot_return_vs_episodes(r_all_episodes, smooth=1):
    r_all_episodes = np.convolve(np.array(r_all_episodes), np.ones(smooth)/smooth, mode='valid')
    plt.plot(r_all_episodes)
    plt.xlabel('episodes')
    plt.ylabel('Return')
    plt.title('Return for each episodes')
    plt.show()