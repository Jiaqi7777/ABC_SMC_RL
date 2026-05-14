import math
import random
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from dataclasses import dataclass

from MCMC_Algorithms.INLA import sample_fields, theta_posterior_delta_method
from Environment.env_util import Pos, Action, next_pos, legal_actions, proposal_action_score, ProposalConfig
from Environment.rewards import RewardConfig, eval_plan

@dataclass
class BeliefMaps:
    rbar_map: np.ndarray   # E[theta^2]  shape (H,W)
    var_map: np.ndarray    # Var[theta]  shape (H,W)

def belief_to_maps_theta2(belief, power=2, obstacle_mask=None, beta: float = 0.0, eps: float = 1e-8) -> BeliefMaps:
    """
    belief.tables: torch.Tensor of shape (N,H,W)
    Returns:
      rbar_map = E[theta^power]
      var_map  = Var[theta]
    """
    tables = torch.log(1 + torch.exp(belief.tables)) # (N,H,W)
    rbar = (tables ** power).mean(dim=0)                  # E[theta^power]
    var  = tables.var(dim=0, unbiased=False) + eps    # Var[theta]

    if beta>0:
        rbar = rbar + beta * var
    rbar_np = rbar.detach().cpu().numpy()
    var_np  = var.detach().cpu().numpy()

    if obstacle_mask is not None:
        rbar_np = rbar_np.copy()
        var_np  = var_np.copy()
        rbar_np[obstacle_mask] = -1e9   # discourage planner from aiming at obstacles
        var_np[obstacle_mask]  = 0.0

    return BeliefMaps(rbar_map=rbar_np, var_map=var_np)

def greedy_heuristic_plan(env, s0: Pos, beliefSample, T: int, alpha: float, rng: random.Random, belief_var=None) -> List[Action]:
    U = []
    s = s0
    for _ in range(T):
        A = legal_actions(env, s)
        scores = [K_score(env, s, a, beliefSample, alpha, belief_var=belief_var) for a in A]
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
    s0: Pos,
    origin: Pos,
    beliefSample,
    T: int,
    rng,
    *,
    cfg: Optional[ProposalConfig] = None,
    belief_var=None,
):
    """
    Returns a length-T action tuple.
    Favors high value, high variance, novelty, and returning to origin.
    """
    if cfg is None:
        cfg = ProposalConfig()

    s = s0
    visited = {s0}
    U = []

    for t in range(T):
        A = legal_actions(env, s)

        scores = [
            proposal_action_score(
                env=env,
                s=s,
                a=a,
                t=t,
                T=T,
                origin=origin,
                visited=visited,
                beliefSample=beliefSample,
                belief_var=belief_var,
                cfg=cfg,
            )
            for a in A
        ]

        a_star = _softmax_sample(A, scores, cfg.temperature, rng)
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
    node_id: int
    s0: Pos
    depth: int

    parent_id: Optional[int] = None
    incoming_plan: Optional[Tuple[Action, ...]] = None

    N: int = 0
    plans: List[Tuple[Action, ...]] = field(default_factory=list)
    stats: Dict[Tuple[Action, ...], EdgeStats] = field(default_factory=dict)

    # maps macro-plan U to child node id
    children: Dict[Tuple[Action, ...], int] = field(default_factory=dict)

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
        rng: Optional[random.Random] = None,
        reward_cfg: Optional[RewardConfig] = None,
        proposal_cfg: Optional[ProposalConfig] = None,
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
        self.reward_cfg = reward_cfg or RewardConfig()
        self.proposal_cfg = proposal_cfg or ProposalConfig(w_r=1.0, w_v=alpha_score, w_rep=2.0, w_inv=5.0, w_ret=0.6, ret_power=2.0, temperature=0.35)
        self.tree: Dict[int, Node] = {}
        self.root_id: Optional[int] = None
        self._next_node_id: int = 0
        
    def _key(self, s0: Pos, depth: int) -> Tuple[Hashable, int]:
        return (s0, depth)

    def _node(self, s0: Pos, depth: int) -> Node:
        key = self._key(s0, depth)
        if key not in self.tree:
            self.tree[key] = Node(s0=s0, depth=depth)
        return self.tree[key]

    def _new_node(
        self,
        *,
        s0: Pos,
        depth: int,
        parent_id: Optional[int] = None,
        incoming_plan: Optional[Tuple[Action, ...]] = None,
    ) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1

        self.tree[node_id] = Node(
            node_id=node_id,
            s0=s0,
            depth=depth,
            parent_id=parent_id,
            incoming_plan=incoming_plan,
        )
        
        return node_id


    def _get_or_create_root(self, root_pos: Pos) -> int:
        if self.root_id is None:
            self.root_id = self._new_node(
                s0=root_pos,
                depth=0,
                parent_id=None,
                incoming_plan=None,
            )
        return self.root_id
    
    def _child_after_plan(self, node: Node, U: Tuple[Action, ...]) -> Node:
        """
        Return the child node linked to macro-plan U.

        This does NOT compute a next physical state.
        The child is defined purely by the tree edge:
            node --U--> child
        """
        U = tuple(U)

        if U in node.children:
            return self.tree[node.children[U]]

        child_id = self._new_node(
            s0=node.s0,                 # state stays the same in your formulation
            depth=node.depth + 1,
            parent_id=node.node_id,
            incoming_plan=U,
        )

        node.children[U] = child_id
        return self.tree[child_id]
    
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

    def _propose_plan(self, env, node, beliefSample, belief_var=None):
        origin = getattr(self, "_origin", node.s0)
        return proposal_value_explore_return_norepeat(
            env,
            s0=node.s0,
            origin=origin,
            beliefSample=beliefSample,
            T=self.T,
            rng=self.rng,
            cfg=self.proposal_cfg,# reuse your alpha_score as variance weight
            belief_var=belief_var,  
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

    def _eval_plan_expected_theta2_original(self, env, s0: Pos, beliefSample, U: Tuple[Action, ...]) -> Tuple[float, Pos]:
        """
        Planning reward = sum gamma^t * E[theta_{s_{t+1}}^2] (invalid move penalty optional)
        """
        s = s0
        R = 0.0
        disc = 1.0
        visited = set({s0})
        for t, a in enumerate(U):
            sp = next_pos(env, s, a)
            if sp in visited and t < self.T - 1:  # loop detected, break and return reward so far
                r = -5  # large penalty for loops
            else:
                r = float(beliefSample[sp])**2  # E[theta^2] at next cell
            visited.add(sp)
            R += disc * r
            disc *= self.gamma
            s = sp
        if s != s0:
            dist = abs(s[0]-s0[0]) + abs(s[1]-s0[1])
            R -= 2 * dist# large penalty for not returning to start proportional to the distance 
        return R, s
    
    def _eval_plan_expected_theta2(self, env, s0: Pos, beliefSample, U: Tuple[Action, ...]):
        return eval_plan(
            env=env,
            s0=s0,
            theta_field=beliefSample,
            U=U,
            gamma=self.gamma,
            cfg=self.reward_cfg,
        )

    def simulate(self, env, node_id: int, beliefSample, belief_var=None) -> float:
        node = self.tree[node_id]

        if node.depth >= self.K:
            return 0.0

        # Progressive widening
        if self._pw_allows_expand(node):
            U_new = self._propose_plan(
                env,
                node,
                beliefSample,
                belief_var=belief_var,
            )
            U_new = tuple(U_new)

            if U_new not in node.stats:
                node.plans.append(U_new)
                node.stats[U_new] = EdgeStats()

        # Select a plan
        untried = [U for U in node.plans if node.stats[U].N == 0]

        if untried:
            U = self.rng.choice(untried)
        else:
            U = self._select_ucb(node)

        # Evaluate this macro-plan from the fixed state s0
        R_macro, _ = self._eval_plan_expected_theta2(
            env,
            node.s0,
            beliefSample,
            U,
        )

        # Move to linked child node, not physical next state
        child = self._child_after_plan(node, U)

        G_next = self.simulate(
            env,
            child.node_id,
            beliefSample,
            belief_var=belief_var,
        )

        G = R_macro + self.gamma_macro * G_next

        # Backprop
        node.N += 1
        st = node.stats[U]
        st.N += 1
        st.W += G
        st.Q = st.W / st.N

        return G

    def plan(self, env, root_pos: Pos, belief, num_sims: int, return_most_visited: bool = True, sample_fn=None, var_fn=None) -> List[Action]:
        self._origin = (int(root_pos[0]), int(root_pos[1]))
        root_id = self._get_or_create_root(root_pos)
        root = self.tree[root_id]

        # Keep root s0 fixed/updated
        root.s0 = root_pos
        root.depth = 0

        if sample_fn is None:
            samples = sample_fields(
                belief=belief,
                n_samples=num_sims
            )['theta_samples']
        else:
            samples = sample_fn(belief=belief, n_samples=num_sims)['theta_samples']
        belief_var = None if var_fn is None else var_fn(belief=belief)[1].reshape(env.n_cell)
        for i in range(num_sims):
            beliefSample = samples[i].reshape(env.n_cell)
            self.simulate(
                env,
                root_id,
                beliefSample,
                belief_var=belief_var,
            )
        root = self.tree[root_id]
        plan_weights = np.array([root.stats[U].N for U in root.plans], dtype=float)
        if not root.plans:
            return greedy_heuristic_plan(env, root_pos, beliefSample, self.T, self.alpha_score, self.rng, belief_var=belief_var)
        plan_weights = [root.stats[U].N for U in root.plans]
        return list(root.plans), np.array(plan_weights)

        # return list(root.plans), plan_weights
        # for i in range(num_sims):
        #     beliefSample = samples[i].reshape(env.n_cell)
        #     self.simulate(env, root_pos, beliefSample, 0, belief_var=belief_var)

        # root = self._node(root_pos, 0)

        #best = max(root.plans, key=lambda U: root.stats[U].N if return_most_visited else root.stats[U].Q)
    
    def _collect_descendants(self, node_id: int, keep: set) -> None:
        if node_id in keep:
            return

        keep.add(node_id)

        node = self.tree[node_id]

        for child_id in node.children.values():
            if child_id in self.tree:
                self._collect_descendants(child_id, keep)


    def _shift_subtree_depths(self, node_id: int, depth_shift: int) -> None:
        node = self.tree[node_id]
        node.depth -= depth_shift

        for child_id in node.children.values():
            if child_id in self.tree:
                self._shift_subtree_depths(child_id, depth_shift)


    def _decay_stats_in_place(self, node: Node, decay: float) -> None:
        node.N = int(round(decay * node.N))

        for U, st in node.stats.items():
            st.N = int(round(decay * st.N))
            st.W = decay * st.W

            if st.N > 0:
                st.Q = st.W / st.N
            else:
                st.W = 0.0
                st.Q = 0.0
    
    def reuse_subtree_from_selected_plan(
        self,
        selected_plan: Tuple[Action, ...],
        *,
        decay: float = 0.5,
        reduce_horizon: bool = True,
    ) -> Optional[int]:
        """
        Re-root the tree at the child linked by selected_plan.

        This follows:
            old_root --selected_plan--> new_root

        It does not compute any physical next state.
        """
        if self.root_id is None:
            return None

        selected_plan = tuple(selected_plan)
        old_root = self.tree[self.root_id]

        if selected_plan not in old_root.children:
            # Selected plan was never expanded into a child.
            # No subtree to reuse.
            self.tree = {}
            self.root_id = None
            self._next_node_id = 0
            return None

        new_root_id = old_root.children[selected_plan]

        # Keep only descendants of the new root.
        keep = set()
        self._collect_descendants(new_root_id, keep)

        self.tree = {
            node_id: node
            for node_id, node in self.tree.items()
            if node_id in keep
        }

        new_root = self.tree[new_root_id]
        old_depth = new_root.depth

        # Remove parent link because this is now the root.
        new_root.parent_id = None
        new_root.incoming_plan = None

        self.root_id = new_root_id

        # Shift depths so new root has depth 0.
        self._shift_subtree_depths(new_root_id, depth_shift=old_depth)

        # Decay old statistics.
        for node in self.tree.values():
            self._decay_stats_in_place(node, decay=decay)

        if reduce_horizon:
            self.K = max(0, self.K - 1)

        return new_root_id



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
    reward_cfg = RewardConfig(
            loop_penalty=5.0,
            return_penalty=2.0,
            reward_power=2.0,
            use_loop_penalty=True,
            use_return_penalty=True,
        )
    alpha_score=0.7
    proposal_cfg = ProposalConfig(w_r=1.0, w_v=alpha_score, w_rep=2.0, w_inv=5.0, w_ret=0.6, ret_power=2.0, temperature=0.35)
    planner = PWMCTSTheta2(
        T=10, K=2, gamma=0.99,
        c_ucb=1.0,
        k_pw=2.0, alpha_pw=0.5,
        alpha_score=alpha_score,   # uncertainty bonus weight in K-score
        p_flip=0.05,
        rng=random.Random(123),
        reward_cfg=reward_cfg,
        proposal_cfg=proposal_cfg,
    )

    plans, plan_weights = planner.plan(env, root_pos=start_pos, beliefSample=bmap.rbar_map, num_sims=500)
    # execute first action (receding horizon) or execute the full plan if you want
    U = rng.choice(plans, p=plan_weights/sum(plan_weights))
    print(U)
    # pos, reward, done = env.step(a0)

