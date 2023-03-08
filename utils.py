import random
import numpy as np
import matplotlib.pyplot as plt
def argmaxs(arr):
    mask = arr == arr.max()
    return random.choice(np.array(range(len(arr)))[mask])

def plot_2d(X, Y, Z, title=None, xlabel='s0', ylabel='s1', zlabel='Value', show=False, additional_info=[], save=False, figure_path = 'Figures/MCMC/'):
    arrows = {2:(1,0), 0:(-1,0),1:(0,1),3:(0,-1)}
    scale = 0.25
    fig, ax = plt.subplots()
    if additional_info != []:
        im = ax.imshow(additional_info)
    ax.set_xticks(np.arange(len(Y)), labels=Y) #Y is the column number
    ax.set_yticks(np.arange(len(X)), labels=X) #X is the row number
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for i in range(len(X)):
        for j in range(len(Y)):
            # text = ax.text(j, i, Z[i, j], ha="center", va="center", color="w")
            ax.arrow(j, i, scale*arrows[Z[i, j]][1], scale*arrows[Z[i, j]][0], head_width=0.1)
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
    figure_path = 'Figures/'
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
    
def plot_return_vs_episodes(r_all_episodes, smooth=1, figure_path='Figures/', show=False):
    r_all_episodes = np.convolve(np.array(r_all_episodes), np.ones(smooth)/smooth, mode='valid')
    plt.plot(r_all_episodes)
    plt.xlabel('episodes')
    plt.ylabel('Return')
    title = 'Return for each episodes'
    plt.title(title)
    plt.savefig(f'{figure_path+title}.png')
    print('figure saved at ', f'{figure_path+title}.png')
    if show:
        plt.show()
    plt.clf()
    
def plot_return_vs_episodes_repeat(r_all_episodes_repeat, figure_path='Figures/', show=False, title='', save=False):
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