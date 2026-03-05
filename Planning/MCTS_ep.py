import math
import random
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class BeliefMaps:
    rbar_map: np.ndarray   # E[theta^2]  shape (H,W)
    var_map: np.ndarray    # Var[theta]  shape (H,W)

def belief_to_maps_theta2(belief, obstacle_mask=None, eps: float = 1e-8) -> BeliefMaps:
    """
    belief.tables: torch.Tensor of shape (N,H,W)
    Returns:
      rbar_map = E[theta^2]
      var_map  = Var[theta]
    """
    tables = torch.log(1 + torch.exp(belief.tables)) # (N,H,W)
    rbar = (tables ** 2).mean(dim=0)                  # E[theta^2]
    var  = tables.var(dim=0, unbiased=False) + eps    # Var[theta]

    rbar_np = rbar.detach().cpu().numpy()
    var_np  = var.detach().cpu().numpy()

    if obstacle_mask is not None:
        rbar_np = rbar_np.copy()
        var_np  = var_np.copy()
        rbar_np[obstacle_mask] = -1e9   # discourage planner from aiming at obstacles
        var_np[obstacle_mask]  = 0.0

    return BeliefMaps(rbar_map=rbar_np, var_map=var_np)


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
    return [0,1,2,3]

def K_score(env, s: Pos, a: Action, bmaps: BeliefMaps, alpha: float) -> float:
    sp = next_pos(env, s, a)
    return float(bmaps.rbar_map[sp] + alpha * bmaps.var_map[sp])

def greedy_heuristic_plan(env, s0: Pos, bmaps: BeliefMaps, T: int, alpha: float, rng: random.Random) -> List[Action]:
    U = []
    s = s0
    for _ in range(T):
        A = legal_actions(env, s)
        scores = [K_score(env, s, a, bmaps, alpha) for a in A]
        m = max(scores)
        best = [a for a, sc in zip(A, scores) if sc == m]
        a_star = rng.choice(best)
        U.append(a_star)
        s = next_pos(env, s, a_star)
    return U

def pw_flip_kernel(env, s0: Pos, U0: Sequence[Action], T: int, p_flip: float, rng: random.Random) -> List[Action]:
    U = []
    s = s0
    for t in range(T):
        A = legal_actions(env, s)
        a0 = U0[t]
        a = rng.choice(A) if rng.random() < p_flip else a0
        U.append(a)
        s = next_pos(env, s, a)
    return U

def _softmax_sample(actions, scores, temperature, rng):
    # stable softmax sampling
    mx = max(scores)
    exps = [math.exp((s - mx) / max(1e-8, temperature)) for s in scores]
    Z = sum(exps)
    r = rng.random() * Z
    c = 0.0
    for a, e in zip(actions, exps):
        c += e
        if c >= r:
            return a
    return actions[-1]

def proposal_value_explore_return_norepeat(
    env,
    s0,
    origin,
    bmaps,
    T,
    rng,
    *,
    w_r=1.0,          # mean/value weight
    w_v=0.5,          # variance/exploration weight
    w_rep=1.5,        # repeat-visit penalty
    w_inv=5.0,        # invalid move penalty
    w_ret=0.5,        # return-to-origin weight (base)
    ret_power=2.0,    # how strongly return pressure ramps near the end
    temperature=0.35, # <1 = greedier, >1 = more random
    eps=1e-9
):
    """
    Returns a length-T action tuple.
    Favors high rbar, high var, novelty, and returning to origin near the end.
    """
    s = s0
    visited = {s0}
    U = []

    for t in range(T):
        A = legal_actions(env, s)

        # ramp return pressure as remaining steps shrink
        # near t=T-1, pressure ~ w_ret; near t=0, smaller
        frac = (t + 1) / T
        w_ret_t = w_ret * (frac ** ret_power)

        scores = []
        for a in A:
            sp = next_pos(env, s, a)

            # base value/exploration
            val = float(bmaps.rbar_map[sp])
            var = float(bmaps.var_map[sp])
            sc = w_r * val + w_v * var

            # invalid moves (should be rare since valid_actions filters, but keep safety)
            if sp == s:
                sc -= w_inv

            # avoid repeats
            if sp in visited:
                sc -= w_rep

            # encourage returning to origin (stronger later)
            dist = abs(sp[0] - origin[0]) + abs(sp[1] - origin[1])
            sc -= w_ret_t * dist

            scores.append(sc)

        a_star = _softmax_sample(A, scores, temperature, rng)
        U.append(a_star)

        s = next_pos(env, s, a_star)
        visited.add(s)

    return tuple(U)


@dataclass
class EdgeStats:
    N: int = 0
    W: float = 0.0
    Q: float = 0.0

@dataclass
class Node:
    s0: Pos
    depth: int
    N: int = 0
    plans: List[Tuple[Action, ...]] = field(default_factory=list)
    stats: Dict[Tuple[Action, ...], EdgeStats] = field(default_factory=dict)

class PWMCTSTheta2:
    def __init__(
        self,
        *,
        T: int,
        K: int,
        gamma: float,
        c_ucb: float,
        k_pw: float,
        alpha_pw: float,
        alpha_score: float,
        p_flip: float,
        rng: Optional[random.Random] = None
    ):
        self.T = T
        self.K = K
        self.gamma = gamma
        self.gamma_macro = gamma ** T
        self.c_ucb = c_ucb
        self.k_pw = k_pw
        self.alpha_pw = alpha_pw
        self.alpha_score = alpha_score
        self.p_flip = p_flip
        self.rng = rng or random.Random(0)
        self.tree: Dict[Tuple[Hashable, int], Node] = {}

    def _key(self, s0: Pos, depth: int) -> Tuple[Hashable, int]:
        return (s0, depth)

    def _node(self, s0: Pos, depth: int) -> Node:
        key = self._key(s0, depth)
        if key not in self.tree:
            self.tree[key] = Node(s0=s0, depth=depth)
        return self.tree[key]

    def _pw_allows_expand(self, node: Node) -> bool:
        if len(node.plans) == 0:
            return True

        n_min = getattr(self, "pw_min_visits_per_plan", 1)
        rho = getattr(self, "pw_coverage_rho", 0.8)

        covered = sum(1 for U in node.plans if node.stats[U].N >= n_min)
        if covered / len(node.plans) < rho:
            return False

        # standard PW budget (only widen if still under budget)
        return len(node.plans) < self.k_pw * (node.N ** self.alpha_pw)

    def _propose_plan(self, env, node, bmaps):
        origin = getattr(self, "_origin", node.s0)
        return proposal_value_explore_return_norepeat(
            env,
            s0=node.s0,
            origin=origin,
            bmaps=bmaps,
            T=self.T,
            rng=self.rng,
            w_r=1.0,
            w_v=self.alpha_score,   # reuse your alpha_score as variance weight
            w_rep=2.0,
            w_inv=5.0,
            w_ret=0.6,
            ret_power=2.0,
            temperature=0.35,
        )

    def _select_ucb(self, node: Node) -> Tuple[Action, ...]:
        logN = math.log(node.N + 1.0)
        best_U, best_val = None, -1e18
        for U in node.plans:
            st = node.stats[U]
            bonus = self.c_ucb * math.sqrt(logN / (st.N))
            val = st.Q + bonus
            if val > best_val:
                best_val = val
                best_U = U
        assert best_U is not None
        return best_U

    def _eval_plan_expected_theta2(self, env, s0: Pos, bmaps: BeliefMaps, U: Tuple[Action, ...]) -> Tuple[float, Pos]:
        """
        Planning reward = sum gamma^t * E[theta_{s_{t+1}}^2] (invalid move penalty optional)
        """
        s = s0
        R = 0.0
        disc = 1.0
        visited = set(s0)
        for t, a in enumerate(U):
            sp = next_pos(env, s, a)
            if sp in visited and t < self.T - 1:  # loop detected, break and return reward so far
                r = -5  # large penalty for loops
            else:
                r = float(bmaps.rbar_map[sp])  # E[theta^2] at next cell
            visited.add(s)
            R += disc * r
            disc *= self.gamma
            s = sp
        if s != s0:
            dist = abs(s[0]-s0[0]) + abs(s[1]-s0[1])
            R -= 2 * dist# large penalty for not returning to start proportional to the distance 
        return R, s

    def simulate(self, env, s0: Pos, bmaps: BeliefMaps, depth: int) -> float:
        if depth >= self.K:
            return 0.0

        node = self._node(s0, depth)

        if self._pw_allows_expand(node):
            U_new = self._propose_plan(env, node, bmaps)
            if U_new not in node.stats:
                node.plans.append(U_new)
                node.stats[U_new] = EdgeStats()

        untried = [U for U in node.plans if node.stats[U].N == 0]
        if untried:
            U = random.choice(untried)  # random or highest prior
        else:
            U = self._select_ucb(node)
        R_macro, s_next = self._eval_plan_expected_theta2(env, s0, bmaps, U)

        G_next = self.simulate(env, s_next, bmaps, depth + 1)
        G = R_macro + self.gamma_macro * G_next

        node.N += 1
        st = node.stats[U]
        st.N += 1
        st.W += G
        st.Q = st.W / st.N
        return G

    def plan(self, env, root_pos: Pos, bmaps: BeliefMaps, num_sims: int, return_most_visited: bool = True) -> List[Action]:
        self._origin = (int(root_pos[0]), int(root_pos[1]))
        for _ in range(num_sims):
            self.simulate(env, root_pos, bmaps, 0)

        root = self._node(root_pos, 0)
        if not root.plans:
            return greedy_heuristic_plan(env, root_pos, bmaps, self.T, self.alpha_score, self.rng)

        best = max(root.plans, key=lambda U: root.stats[U].N if return_most_visited else root.stats[U].Q)
        return list(best)


# -----------------------------
# Minimal example glue (replace with your own)
# -----------------------------
if __name__ == "__main__":
    import sys
    import os
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.path.append('/scratch/Rabbit/work/ABC_SMC_RL/')
    sys.path.append('/Users/guojiaqi/work/ABC_SMC_RL/')
    from Environment.Grid import IrregularGridWorld
    from model import *

    rows, cols = 6, 6
    obstacles = np.zeros((rows, cols), dtype=bool)
    start_pos = (2, 2)

    def normal_reward(mu, sigma):
        return lambda: np.random.normal(mu, sigma)
    
    # Create environment
    env = IrregularGridWorld((rows, cols), obstacles)
    belief = Tabular(env=env, n_particle=5, prior='mrf', mean=2, std=0.1, unary_sigma=0.5)#, gamma=GAMMA, initial_tables=initial_tables, initial_weights=initial_weights, idx=FROZEN_IDX)#For frozen all but one dimensions

    # After you update belief.tables via SMC:
    bmaps = belief_to_maps_theta2(belief, obstacle_mask=env.obstacle_mask)

    planner = PWMCTSTheta2(
        T=10, K=2, gamma=0.99,
        c_ucb=1.0,
        k_pw=2.0, alpha_pw=0.5,
        alpha_score=0.7,   # uncertainty bonus weight in K-score
        p_flip=0.05,
        rng=random.Random(123),
    )

    U_star = planner.plan(env, root_pos=start_pos, bmaps=bmaps, num_sims=500)
    # execute first action (receding horizon) or execute the full plan if you want
    a0 = U_star
    print(a0)
    # pos, reward, done = env.step(a0)

