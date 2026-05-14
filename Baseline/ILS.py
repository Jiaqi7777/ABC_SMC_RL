from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np


Pos = Tuple[int, int]
Action = int  

def bfs_distances(env, start: Pos) -> np.ndarray:
    """4-neighbour BFS distances on obstacle grid; inf if unreachable."""
    H, W = env.rows, env.cols
    dist = np.full((H, W), np.inf, dtype=np.float32)
    sr, sc = start
    if env.obstacle_mask[sr, sc]:
        return dist
    dist[sr, sc] = 0.0
    q = [(sr, sc)]
    head = 0
    while head < len(q):
        r, c = q[head]
        head += 1
        d = dist[r, c] + 1.0
        for a in DIRS:
            rr, cc = next_pos(env, (r, c), a)
            if (rr, cc) == (r, c):  # obstacle or boundary
                continue
            if d < dist[rr, cc]:
                dist[rr, cc] = d
                q.append((rr, cc))
    return dist


def exact_return_feasible(dist_to_home: float, steps_left: int) -> bool:
    """
    Return-to-start in exactly steps_left steps on a 4-neighbour grid:
    feasible iff dist <= steps_left and parity matches (can waste 2 steps by backtrack).
    """
    if math.isinf(dist_to_home):
        return False
    d = int(dist_to_home)
    if d > steps_left:
        return False
    return ((steps_left - d) % 2) == 0


def rollout_plan_soft(
    env,
    start_pos: Pos,
    actions: List[Action],
    bmaps: BMaps,
    gamma: float = 1.0,
    invalid_penalty: float = -1.0,
    repeat_penalty: float = 1.0,
    return_penalty: float = 1.0,
) -> Tuple[List[Pos], float, int]:
    """
    rollout_plan but:
      - visited is counted properly (soft penalty rather than break)
      - invalid moves (stay-in-place) get invalid_penalty
      - revisits get repeat_penalty * (#previous visits)
      - end not at start => subtract return_penalty * shortest-path-distance-to-start (BFS dist if available)
    """
    s = start_pos
    path = [s]
    R = 0.0
    disc = 1.0
    vis: Dict[Pos, int] = {s: 1}  # count visits
    invalid = 0

    for t, a in enumerate(actions):
        sp = next_pos(env, s, a)
        if sp == s:
            invalid += 1
            r = invalid_penalty
        else:
            base = float(bmaps.rbar_map[sp])
            cnt = vis.get(sp, 0)
            r = base - repeat_penalty * cnt
        vis[sp] = vis.get(sp, 0) + 1

        R += disc * r
        disc *= gamma
        s = sp
        path.append(s)

    if s != start_pos:
        # Manhattan is OK if no obstacles; if obstacles exist, you can precompute BFS dist-to-home and use it.
        dist = abs(s[0] - start_pos[0]) + abs(s[1] - start_pos[1])
        R -= return_penalty * dist

    return path, R, invalid


# -----------------------------
# ILS baseline for fixed-length cyclic tour
# -----------------------------
@dataclass
class ILSConfig:
    T: int
    gamma: float = 1.0
    beta_ucb: float = 0.0              # if you build bmaps via mean+beta*std
    repeat_penalty: float = 0.5        # soft no-repeat
    invalid_penalty: float = -1.0
    return_penalty: float = 2.0
    eta_return_bias: float = 0.05      # shaping during construction (bias to keep return feasible)
    top_k: int = 6                     # randomized greedy
    n_starts: int = 30
    ils_iters: int = 300
    improve_passes: int = 80
    window_min: int = 6
    window_max: int = 12
    perturb_windows: int = 2
    seed: int = 0


class GridILS:
    """
    Works with your env + next_pos.
    Produces an action sequence of length T.
    Enforces exact-length return feasibility during construction (distance+parity),
    but revisits are only softly penalized (repeat_penalty).
    """

    def __init__(self, env, start: Pos, bmaps: BMaps, cfg: ILSConfig):
        self.env = env
        self.start = start
        self.bmaps = bmaps
        self.cfg = cfg

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.dist_home = bfs_distances(env, start)  # for feasibility checks

        self.all_actions = list(DIRS.keys())

    def eval_actions(self, actions: List[Action]) -> float:
        _, R, _ = rollout_plan_soft(
            self.env,
            self.start,
            actions,
            self.bmaps,
            gamma=self.cfg.gamma,
            invalid_penalty=self.cfg.invalid_penalty,
            repeat_penalty=self.cfg.repeat_penalty,
            return_penalty=self.cfg.return_penalty,
        )
        return R

    def feasible_next_actions(self, s: Pos, steps_left_after_move: int) -> List[Action]:
        out = []
        for a in self.all_actions:
            sp = next_pos(self.env, s, a)
            # allow invalid (stay) but it's usually bad; still keep it only if you want.
            # Here we keep it ONLY if it preserves return feasibility (it does) but it's penalized anyway.
            d = self.dist_home[sp]
            if exact_return_feasible(d, steps_left_after_move):
                out.append(a)
        return out

    def construct_randomized_greedy(self) -> List[Action]:
        """
        Build length-T actions with exact-return feasibility at every step.
        Score uses mean/UCB map - repeat penalty - eta*dist_to_home.
        """
        T = self.cfg.T
        lam = self.cfg.repeat_penalty
        eta = self.cfg.eta_return_bias
        top_k = max(1, self.cfg.top_k)

        s = self.start
        vis: Dict[Pos, int] = {s: 1}
        actions: List[Action] = []

        for t in range(T):
            steps_left_after_move = T - (t + 1)
            feas = self.feasible_next_actions(s, steps_left_after_move)
            if not feas:
                # restart caller will discard; return a poor walk
                return [random.choice(self.all_actions) for _ in range(T)]

            scored = []
            for a in feas:
                sp = next_pos(self.env, s, a)
                d = float(self.dist_home[sp])
                base = self.cfg.invalid_penalty if sp == s else float(self.bmaps.rbar_map[sp])
                cnt = vis.get(sp, 0)
                score = base - lam * cnt - eta * d
                scored.append((score, a))

            scored.sort(key=lambda x: x[0], reverse=True)
            k = min(top_k, len(scored))
            a = random.choice(scored[:k])[1]

            sp = next_pos(self.env, s, a)
            actions.append(a)
            vis[sp] = vis.get(sp, 0) + 1
            s = sp

        return actions

    # ---- local improvement: window DP with fixed endpoints & length ----
    def improve_window_dp(self, actions: List[Action], i: int, j: int) -> List[Action]:
        """
        Replan sub-actions i..j-1 (length L=j-i) while keeping endpoints positions fixed.
        DP state: (step l, position pos) -> best value (heuristic: repeat penalties use outside counts only).
        """
        assert 0 <= i < j <= self.cfg.T
        L = j - i

        # Build full path of current tour to know endpoints
        path, _, _ = rollout_plan_soft(
            self.env, self.start, actions, self.bmaps,
            gamma=1.0,  # for endpoint tracking, discount irrelevant
            invalid_penalty=0.0, repeat_penalty=0.0, return_penalty=0.0
        )
        p_i = path[i]
        p_j = path[j]

        # Outside counts (positions visited outside the window interior)
        outside: Dict[Pos, int] = {}
        for t, p in enumerate(path):
            if t <= i or t >= j:
                outside[p] = outside.get(p, 0) + 1

        lam = self.cfg.repeat_penalty

        # DP tables
        dp: List[Dict[Pos, float]] = [dict() for _ in range(L + 1)]
        prev_pos: List[Dict[Pos, Pos]] = [dict() for _ in range(L + 1)]
        prev_act: List[Dict[Pos, Action]] = [dict() for _ in range(L + 1)]

        dp[0][p_i] = 0.0

        for l in range(L):
            for pos, val in dp[l].items():
                for a in self.all_actions:
                    nxt = next_pos(self.env, pos, a)
                    if l == L - 1 and nxt != p_j:
                        continue
                    # step reward proxy
                    base = self.cfg.invalid_penalty if nxt == pos else float(self.bmaps.rbar_map[nxt])
                    cnt = outside.get(nxt, 0)
                    step_gain = base - lam * cnt
                    cand = val + step_gain
                    if (nxt not in dp[l + 1]) or (cand > dp[l + 1][nxt]):
                        dp[l + 1][nxt] = cand
                        prev_pos[l + 1][nxt] = pos
                        prev_act[l + 1][nxt] = a

        if p_j not in dp[L]:
            return actions  # no feasible replacement

        # Reconstruct best window actions
        new_window_actions: List[Action] = []
        cur = p_j
        for l in range(L, 0, -1):
            a = prev_act[l][cur]
            new_window_actions.append(a)
            cur = prev_pos[l][cur]
        new_window_actions.reverse()

        new_actions = actions[:i] + new_window_actions + actions[j:]
        return new_actions

    def perturb(self, actions: List[Action]) -> List[Action]:
        T = self.cfg.T
        for _ in range(self.cfg.perturb_windows):
            L = random.randint(self.cfg.window_min, self.cfg.window_max)
            if L >= T:
                continue
            i = random.randint(0, T - L)
            j = i + L
            actions = self.improve_window_dp(actions, i, j)
        return actions

    def solve(self) -> Tuple[List[Action], float]:
        # multi-start init
        best = None
        best_v = -1e18
        for _ in range(self.cfg.n_starts):
            a = self.construct_randomized_greedy()
            v = self.eval_actions(a)
            if v > best_v:
                best, best_v = a, v

        cur, cur_v = best, best_v

        for _it in range(self.cfg.ils_iters):
            cand = self.perturb(cur)

            # local improvement passes
            for _ in range(self.cfg.improve_passes):
                L = random.randint(self.cfg.window_min, self.cfg.window_max)
                if L >= self.cfg.T:
                    continue
                i = random.randint(0, self.cfg.T - L)
                j = i + L
                improved = self.improve_window_dp(cand, i, j)

                v_imp = self.eval_actions(improved)
                v_cand = self.eval_actions(cand)
                if v_imp > v_cand + 1e-9:
                    cand = improved

            cand_v = self.eval_actions(cand)
            if cand_v > cur_v + 1e-9:
                cur, cur_v = cand, cand_v
                if cand_v > best_v + 1e-9:
                    best, best_v = cand, cand_v

        return best, best_v

import matplotlib.pyplot as plt

def actions_to_path(env, start_pos, actions):
    s = start_pos
    path = [s]
    for a in actions:
        s = next_pos(env, s, a)
        path.append(s)
    return path

def plot_topk_paths(env, start_pos, bmaps, candidates, k=5, title=None):
    """
    candidates: list of (actions, value) where actions is length-T list
    """
    # sort by value desc, take top-k
    top = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]



    # overlay paths
    for i, (actions, val) in enumerate(top, start=1):
        # background: reward map (posterior mean/UCB)
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(bmaps.rbar_map, origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # mark start
        ax.scatter([start_pos[1]], [start_pos[0]], marker="*", s=200, label="start")
        path = actions_to_path(env, start_pos, actions)
        rr = [p[0] for p in path]
        cc = [p[1] for p in path]

        ax.plot(cc, rr, linewidth=2, label=f'{i}: R={val:.2f}')
        # optional: show endpoints
        ax.scatter([cc[-1]], [rr[-1]], s=30)

        ax.set_title(title or f"Top-{i} paths: R={val:.2f}")
        ax.set_xlim(-0.5, env.cols - 0.5)
        ax.set_ylim(env.rows - 0.5, -0.5)  # keep origin upper
        ax.legend(loc="best", fontsize=9)
        ax.grid(False)
        plt.tight_layout()
        plt.show()

    return top 

if __name__ == "__main__":
    import sys
    import os
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.path.append('/scratch/Rabbit/work/ABC_SMC_RL/')
    sys.path.append('/Users/guojiaqi/work/ABC_SMC_RL/')
    from Environment.Grid import IrregularGridWorld
    from model import *
    from Planning.MCTS_ep import *
    # Example:
    # particles: (N,H,W) posterior reward samples (e.g., theta^2)
    rows, cols = 6, 6
    env = IrregularGridWorld((rows, cols))
    particles = Tabular(env=env, n_particle=5, prior='mrf', mean=2, std=0.1, unary_sigma=0.5)#, gamma=GAMMA, initial_tables=initial_tables, initial_weights=initial_weights, idx=FROZEN_IDX)#For frozen all but one dimensions
    start_pos = (2, 2)
    random.seed(123)
    bmaps = belief_to_maps_theta2(particles, beta=0.5)
    def run_ils_fn(seed=123, solver=None):
        cfg = ILSConfig(
            T=20,
            repeat_penalty=0.8,
            invalid_penalty=-2.0,
            return_penalty=2.0,
            eta_return_bias=0.05,
            top_k=8,
            n_starts=40,
            ils_iters=300,
            improve_passes=80,
            window_min=6,
            window_max=12,
            perturb_windows=2,
            seed=seed,
        )

        # env must have: rows, cols, obstacle_mask (bool array)
        
        solver = GridILS(env=env, start=start_pos, bmaps=bmaps, cfg=cfg) if solver is None else solver
        best_actions, best_value = solver.solve()
        print("best_value:", best_value, "len(actions):", len(best_actions))
        return best_actions, best_value, solver

    def collect_ils_candidates(run_ils_fn, n_runs=10):
        """
        run_ils_fn(seed) -> (best_actions, best_value)
        """
        out = []
        solver = None
        for seed in range(123, n_runs + 123):
            actions, val, solver = run_ils_fn(seed, solver)
            out.append((actions, val))
        plot_return([[o[1] for o in out]])
        return out

    candidates = collect_ils_candidates(run_ils_fn, n_runs=5)
    plot_topk_paths(env, start_pos, bmaps, candidates, k=5)