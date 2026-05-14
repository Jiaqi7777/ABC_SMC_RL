from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Set
import numpy as np
import random
from dataclasses import dataclass
from gym import spaces
import matplotlib.pyplot as plt
Pos = Tuple[int, int]
Action = int

DIRS = {0: (-1,0), 1: (0,1), 2: (1,0), 3: (0,-1)}

def next_pos(env, s: Pos, a: Action) -> Pos:
    di, dj = DIRS[a]
    ni, nj = s[0] + di, s[1] + dj
    if 0 <= ni < env.rows and 0 <= nj < env.cols and (not env.obstacle_mask[ni, nj]):
        return (ni, nj)
    return s

def legal_actions(env, s: Pos) -> List[Action]:
    return [a for a in range(4) if next_pos(env, s, a) != s]

def rollout_actions(env, s0: Pos, actions: Sequence[Action]) -> List[Pos]:
    """
    Returns trajectory [s0, s1, ..., sT].
    """
    s = s0
    trajectory = [s]

    for a in actions:
        s = next_pos(env, s, a)
        trajectory.append(s)

    return trajectory

def K_score(env, s: Pos, a: Action, beliefSample, alpha: float, belief_var=None) -> float:
    sp = next_pos(env, s, a)
    var = 0.0 if belief_var is None else float(belief_var[sp])
    return float(beliefSample[sp] + alpha * var) 

@dataclass
class ProposalConfig:
    w_r: float = 1.0
    w_v: float = 0.5
    w_rep: float = 2.0
    w_inv: float = 5.0
    w_ret: float = 0.6
    ret_power: float = 2.0
    temperature: float = 0.35
    
    
def proposal_action_score(
    *,
    env,
    s: Pos,
    a: Action,
    t: int,
    T: int,
    origin: Pos,
    visited: Set[Pos],
    beliefSample,
    belief_var,
    cfg: ProposalConfig,
) -> float:
    sp = next_pos(env, s, a)

    val = float(beliefSample[sp]) ** 2
    var = 0.0 if belief_var is None else float(belief_var[sp])

    sc = cfg.w_r * val + cfg.w_v * var

    if sp == s:
        sc -= cfg.w_inv

    if sp in visited:
        sc -= cfg.w_rep

    frac = (t + 1) / T
    w_ret_t = cfg.w_ret * (frac ** cfg.ret_power)

    dist = abs(sp[0] - origin[0]) + abs(sp[1] - origin[1])
    sc -= w_ret_t * dist

    return float(sc)



@dataclass
class TrueEnvironment:
    H: int
    W: int
    z_true: np.ndarray        # (M,)
    theta_true: np.ndarray    # (M,)

    def z_grid(self) -> np.ndarray:
        return self.z_true.reshape(self.H, self.W)

    def theta_grid(self) -> np.ndarray:
        return self.theta_true.reshape(self.H, self.W)


def visualize_path(
    env,
    true_field,
    start_pos,
    path,
    k,
    K,
    show_arrows=True,
    show_points=True,
    figsize=(7, 7),
    title="executed paths (color = episode)",
    seed=0,
):
    """
    Visualize the executed path in the environment.
    - env must have rows, cols, obstacle_mask (2D bool/0-1).
    - bmaps.rbar_map and bmaps.var_map indexed by (i,j).
    """
    rng = np.random.default_rng(seed)

    H, W = env.rows, env.cols
    obs = np.array(env.obstacle_mask, dtype=bool)

    # Background: obstacles black, free cells white
    bg = np.ones((H, W), dtype=float)
    bg[obs] = 0.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(true_field.theta_grid(), origin="upper")  # (0,0) is top-left to match (i,j)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", linewidth=0.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Colormap for episodes
    cmap = plt.get_cmap("hsv")
    colors = [cmap(i / max(1, K - 1)) for i in range(K)]

    # Convert (i,j) to plot coords (x=j, y=i)
    xs = [p[1] for p in path]
    ys = [p[0] for p in path]
    c = colors[k]

    # Draw path
    ax.plot(xs, ys, linewidth=2.0, alpha=0.9, color=c)
    if show_points:
        ax.scatter(xs[0], ys[0], s=60, marker="o", color=c, edgecolors="k", linewidths=0.5)  # start
        ax.scatter(xs[-1], ys[-1], s=70, marker="X", color=c, edgecolors="k", linewidths=0.5)  # end

    # Draw arrows for direction (optional)
    if show_arrows and len(xs) >= 2:
        dx = np.diff(xs)
        dy = np.diff(ys)
        ax.quiver(
            xs[:-1], ys[:-1], dx, dy,
            angles="xy", scale_units="xy", scale=1,
            width=0.004, alpha=0.35, color=c
        )
    ax.contour(obs.astype(int), levels=[0.5], colors="k", linewidths=1)

#                 print("Episode | steps | invalid | discounted_return | end_pos | path")
#                 for ep, steps, invalid, R, path in summaries:
#                     print(f"{ep:7d} | {steps:5d} | {invalid:7d} | {R:16.4f} | {path[-1]} | {path}")

    plt.show()