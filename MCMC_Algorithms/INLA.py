import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from utils import softplus, softplus_prime, softplus_second, flatten_index
from Environment.env_util import TrueEnvironment, Pos
from Environment.rewards import RewardConfig, planning_step_reward

# ============================================================
# Types
# ============================================================

Obs = Tuple[float, Tuple[int, int]]  # (tilde_theta, (i, j))


# ============================================================
# Basic utilities
# ============================================================



# ============================================================
# Sparse GMRF prior
# ============================================================

def make_4nb_laplacian_precision(
    H: int,
    W: int,
    tau: float = 1.0,
    kappa: float = 1e-2,
    periodic: bool = False,
) -> sp.csr_matrix:
    """
    Sparse 4-neighbour GMRF precision:
        Q = tau * (L + kappa I)
    """
    M = H * W
    rows, cols, vals = [], [], []

    def add(r: int, c: int, v: float) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(v)

    for i in range(H):
        for j in range(W):
            idx = flatten_index(i, j, W)
            nbrs = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if periodic:
                    ni %= H
                    nj %= W
                    nbrs.append((ni, nj))
                else:
                    if 0 <= ni < H and 0 <= nj < W:
                        nbrs.append((ni, nj))

            add(idx, idx, tau * (len(nbrs) + kappa))
            for ni, nj in nbrs:
                nidx = flatten_index(ni, nj, W)
                add(idx, nidx, -tau)

    Q = sp.coo_matrix((vals, (rows, cols)), shape=(M, M)).tocsr()
    return Q


def sparse_cholesky_sample_unstable(
    mean: np.ndarray,
    precision: sp.csr_matrix,
    n_samples: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample from N(mean, precision^{-1}) using sparse solves.

    Method:
        If x ~ N(0, Q^{-1}), then x = Q^{-1/2} eps.
        We approximate by solving:
            L^T x = eps, where Q = L L^T
        Since sparse Cholesky is not directly available in scipy by default,
        we use dense fallback for moderate problems.

    Returns:
        samples: (n_samples, M)
    """
    if rng is None:
        rng = np.random.default_rng()

    M = mean.shape[0]
    Q_dense = precision.toarray()
    L = np.linalg.cholesky(Q_dense)  # Q = L L^T

    samples = []
    for _ in range(n_samples):
        eps = rng.standard_normal(M)
        x = np.linalg.solve(L.T, eps)   # cov = Q^{-1}
        samples.append(mean + x)
    return np.asarray(samples)

def sparse_cholesky_sample(
    mean,
    precision,
    n_samples,
    rng=None,
    initial_jitter=1e-8,
    max_tries=8,
):
    if rng is None:
        rng = np.random.default_rng()

    mean = np.asarray(mean)
    M = mean.shape[0]

    Q = precision.toarray()
    Q = 0.5 * (Q + Q.T)

    jitter = initial_jitter

    for attempt in range(max_tries):
        try:
            Q_reg = Q + jitter * np.eye(M)
            L = np.linalg.cholesky(Q_reg)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        eig_min = np.linalg.eigvalsh(Q)[0]
        raise np.linalg.LinAlgError(
            f"Precision matrix is not positive definite even after jitter. "
            f"min eigenvalue={eig_min:.3e}, final jitter={jitter:.3e}"
        )

    samples = np.empty((n_samples, M))

    for n in range(n_samples):
        eps = rng.normal(size=M)
        x = np.linalg.solve(L.T, eps)
        samples[n] = mean + x

    return samples

# ============================================================
# Belief representation
# ============================================================

@dataclass
class SparseGMRFBelief:
    H: int
    W: int
    mean: np.ndarray          # current Gaussian approx mean for z, shape (M,)
    precision: sp.csr_matrix  # current Gaussian approx precision for z, shape (M, M)

    @property
    def M(self) -> int:
        return self.H * self.W

    def mean_grid(self) -> np.ndarray:
        return self.mean.reshape(self.H, self.W)

    def var_grid(self) -> np.ndarray:
        return approximate_diag_of_inverse(self.precision).reshape(self.H, self.W)


# ============================================================
# True environment generation
# ============================================================


def theta_posterior_delta_method(
    belief,
    a: float = 1.0,
    b: float = 0.0,
    n_probes: int = 80,
    seed: int = 0,
):
    """
    Approximate Var(theta_s | D) for each site s using the delta method.

    belief.mean      : (M,)
    belief.precision : sparse (M, M)

    Returns
    -------
    theta_mean : (M,)
    theta_var  : (M,)
    """
    z_mean = belief.mean
    z_var = approximate_diag_of_inverse(belief.precision, n_probes=n_probes, seed=seed)

    gp = softplus_prime(z_mean, a=a, b=b)
    gpp = softplus_second(z_mean, a=a, b=b)
    g = softplus(z_mean, a=a, b=b)

    theta_mean = g + 0.5 * gpp * z_var
    theta_var = (gp ** 2) * z_var
    return theta_mean, theta_var

def generate_true_field(
    H: int,
    W: int,
    prior_precision: sp.csr_matrix,
    mean: Optional[np.ndarray] = None,
    a: float = 1.0,
    b: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> TrueEnvironment:
    """
    Generate one true latent field z_true ~ N(mean, Q^{-1})
    and theta_true = softplus(z_true).
    """
    if rng is None:
        rng = np.random.default_rng()

    M = H * W
    if mean is None:
        mean = np.zeros(M, dtype=float)

    z_true = sparse_cholesky_sample(mean, prior_precision, n_samples=1, rng=rng)[0]
    theta_true = softplus(z_true, a=a, b=b)
    return TrueEnvironment(H=H, W=W, z_true=z_true, theta_true=theta_true)


# ============================================================
# Observation model
# ============================================================

def observe_from_true_field_original(
    true_field: TrueEnvironment,
    visited_sites: List[Tuple[int, int]],
    sigma_theta: float,
    rng: Optional[np.random.Generator] = None,
) -> List[Obs]:
    """
    Simulate noisy observations at visited sites:
        y = theta_true(site) + noise
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W = true_field.H, true_field.W
    obs_list: List[Obs] = []
    reward_list: List[float] = []
    s0 = visited_sites[0] 
    visited = set({s0})  
    for t, (i, j) in enumerate(visited_sites[1:]):
        if not (0 <= i < H and 0 <= j < W):
            raise ValueError(f"Site {(i, j)} out of bounds for {(H, W)}")
        idx = flatten_index(i, j, W)
        y = true_field.theta_true[idx] + rng.normal(0.0, sigma_theta)
        obs_list.append((float(y), (i, j)))
        r = float(true_field.theta_true[idx])**2  # reward is theta^2 at the site
        if (i, j) in visited and t < len(visited_sites) - 2:
            r-= 5  # large penalty for loops
        if t >= len(visited_sites)//2 - 1:
            dist = abs(i-s0[0]) + abs(j-s0[1])
            r -= dist
        reward_list.append(r)
        visited.add((i, j))
    return obs_list, reward_list

def observe_from_true_field(
    true_field: TrueEnvironment,
    visited_sites: List[Pos],
    sigma_theta: float,
    rng: Optional[np.random.Generator] = None,
    reward_cfg: Optional[RewardConfig] = None,
) -> Tuple[List[Obs], List[float]]:
    """
    Simulate noisy observations at visited sites:
        y_s = theta_true(s) + noise

    Also returns realised rewards using the shared reward rule.
    """
    
    if rng is None:
        rng = np.random.default_rng()

    if reward_cfg is None:
        reward_cfg = RewardConfig(
            loop_penalty=5.0,
            return_penalty=0.0,  # because this function currently uses stepwise return pressure
            reward_power=2.0,
            use_loop_penalty=True,
            use_return_penalty=True,
        )

    theta_grid = np.asarray(true_field.theta_true).reshape(true_field.H, true_field.W)

    H, W = true_field.H, true_field.W
    obs_list: List[Obs] = []
    reward_list: List[float] = []

    s0 = visited_sites[0]
    visited = {s0}
    T = len(visited_sites) - 1

    for t, sp in enumerate(visited_sites[1:]):
        i, j = sp

        if not (0 <= i < H and 0 <= j < W):
            raise ValueError(f"Site {sp} out of bounds for {(H, W)}")

        y = theta_grid[sp] + rng.normal(0.0, sigma_theta)
        obs_list.append((float(y), sp))

        r = planning_step_reward(
            theta_field=theta_grid,
            sp=sp,
            visited=visited,
            t=t,
            T=T,
            origin=s0,
            cfg=reward_cfg,
        )

        # Optional: keep your original gradual return pressure
        if t >= T // 2 - 1:
            dist = abs(sp[0] - s0[0]) + abs(sp[1] - s0[1])
            r -= dist

        reward_list.append(float(r))
        visited.add(sp)

    return obs_list, reward_list
# ============================================================
# Laplace update pieces
# ============================================================

def neg_log_posterior_sparse(
    z: np.ndarray,
    m0: np.ndarray,
    Q0: sp.csr_matrix,
    y: np.ndarray,
    sites: np.ndarray,
    sigma_theta: float,
    a: float,
    b: float,
) -> float:
    dz = z - m0
    prior_term = 0.5 * dz @ (Q0 @ dz)
    gz = softplus(z[sites], a=a, b=b)
    resid = y - gz
    ll_term = 0.5 * np.sum(resid**2) / (sigma_theta**2)
    return float(prior_term + ll_term)


def approximate_diag_of_inverse(
    Q: sp.csr_matrix,
    n_probes: int = 80,
    seed: int = 0,
) -> np.ndarray:
    """
    Hutchinson estimator for diag(Q^{-1}).
    """
    rng = np.random.default_rng(seed)
    M = Q.shape[0]
    acc = np.zeros(M, dtype=float)

    for _ in range(n_probes):
        v = rng.choice([-1.0, 1.0], size=M)
        x = spla.spsolve(Q, v)
        acc += x * v

    return acc / n_probes


def compress_observations(
    obs_list: List[Obs],
    H: int,
    W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ys = []
    sites = []
    for y, (i, j) in obs_list:
        if not (0 <= i < H and 0 <= j < W):
            raise ValueError(f"Observation site {(i, j)} out of bounds for {(H, W)}")
        ys.append(float(y))
        sites.append(flatten_index(i, j, W))
    return np.asarray(ys, dtype=float), np.asarray(sites, dtype=np.int64)

def observation_precision_weights(
    z_sites: np.ndarray,
    y: np.ndarray,
    sigma_theta: float,
    a: float,
    b: float,
    method: str = "gauss_newton",
    min_weight: float = 1e-10,
) -> np.ndarray:
    """
    Returns diagonal observation precision weights for the Laplace/Gaussian update.

    method="exact":
        Uses exact Hessian:
            (g'^2 - residual g'') / sigma^2
        Can be negative.

    method="gauss_newton":
        Uses positive Gauss-Newton/Fisher approximation:
            g'^2 / sigma^2
        Recommended for stable sequential updates.
    """
    inv_sigma2 = 1.0 / (sigma_theta**2)

    g = softplus(z_sites, a=a, b=b)
    gp = softplus_prime(z_sites, a=a, b=b)

    if method == "gauss_newton":
        w = (gp**2) * inv_sigma2

    elif method == "exact":
        gpp = softplus_second(z_sites, a=a, b=b)
        resid = y - g
        w = (gp**2 - resid * gpp) * inv_sigma2

    elif method == "clipped_exact":
        gpp = softplus_second(z_sites, a=a, b=b)
        resid = y - g
        w = (gp**2 - resid * gpp) * inv_sigma2
        w = np.maximum(w, min_weight)

    else:
        raise ValueError(f"Unknown method: {method}")

    return np.maximum(w, min_weight)

def sparse_gmrf_softplus_update(
    belief: SparseGMRFBelief,
    obs_list: List[Obs],
    sigma_theta: float,
    a: float = 1.0,
    b: float = 0.0,
    max_newton: int = 25,
    tol: float = 1e-6,
    damping: float = 1.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Batch Laplace update:
        prior approx belief: z ~ N(m0, Q0^{-1})
        posterior approx:    z | D ~ N(z_map, Q_post^{-1})
    """
    H, W = belief.H, belief.W
    M = belief.M
    m0 = belief.mean.copy()
    Q0 = belief.precision.tocsr()

    y, sites = compress_observations(obs_list, H, W)

    if len(y) == 0:
        return {
            "z_map": m0.copy(),
            "Q_post": Q0.copy(),
            "site_var": approximate_diag_of_inverse(Q0),
            "theta_mean_plug": softplus(m0, a=a, b=b),
            "success": True,
            "n_iter": 0,
            "final_grad_norm": 0.0,
        }

    inv_sigma2 = 1.0 / (sigma_theta**2)
    z = m0.copy()
    success = False
    final_grad_norm = np.inf

    for it in range(max_newton):
        z_sites = z[sites]
        g = softplus(z_sites, a=a, b=b)
        gp = softplus_prime(z_sites, a=a, b=b)
        gpp = softplus_second(z_sites, a=a, b=b)
        resid = y - g

        # gradient of negative log posterior
        grad = Q0 @ (z - m0)
        obs_grad = -(resid * gp) * inv_sigma2
        np.add.at(grad, sites, obs_grad)

        final_grad_norm = float(np.linalg.norm(grad))
        if verbose:
            print(f"[update] iter={it:02d}, ||grad||={final_grad_norm:.3e}")

        if final_grad_norm < tol:
            success = True
            break

        # Hessian / precision update term
        w_diag = observation_precision_weights(
            z_sites=z_sites,
            y=y,
            sigma_theta=sigma_theta,
            a=a,
            b=b,
            method="gauss_newton",
        )
        site_weights = np.zeros(M, dtype=float)
        np.add.at(site_weights, sites, w_diag)

        Q_post = Q0 + sp.diags(site_weights, offsets=0, shape=(M, M), format="csr")

        delta = spla.spsolve(Q_post, grad)
        z_new = z - damping * delta

        # simple backtracking
        old_obj = neg_log_posterior_sparse(z, m0, Q0, y, sites, sigma_theta, a, b)
        new_obj = neg_log_posterior_sparse(z_new, m0, Q0, y, sites, sigma_theta, a, b)
        if new_obj > old_obj:
            step = damping
            accepted = False
            for _ in range(10):
                step *= 0.5
                z_try = z - step * delta
                obj_try = neg_log_posterior_sparse(z_try, m0, Q0, y, sites, sigma_theta, a, b)
                if obj_try < old_obj:
                    z_new = z_try
                    accepted = True
                    break
            if not accepted:
                z_new = z

        z = z_new

    # final precision at MAP
    z_sites = z[sites]
    g = softplus(z_sites, a=a, b=b)
    gp = softplus_prime(z_sites, a=a, b=b)
    gpp = softplus_second(z_sites, a=a, b=b)
    resid = y - g
    w_diag = observation_precision_weights(
        z_sites=z_sites,
        y=y,
        sigma_theta=sigma_theta,
        a=a,
        b=b,
        method="gauss_newton",
    )

    site_weights = np.zeros(M, dtype=float)
    np.add.at(site_weights, sites, w_diag)
    Q_post = Q0 + sp.diags(site_weights, offsets=0, shape=(M, M), format="csr")

    site_var = approximate_diag_of_inverse(Q_post)

    return {
        "z_map": z,
        "Q_post": Q_post,
        "site_var": site_var,
        "theta_mean_plug": softplus(z, a=a, b=b),
        "success": success,
        "n_iter": it + 1,
        "final_grad_norm": final_grad_norm,
    }


# ============================================================
# Belief update wrapper across episodes
# ============================================================

def update_belief_from_episode(
    belief: SparseGMRFBelief,
    episode_obs: List[Obs],
    sigma_theta: float,
    a: float = 1.0,
    b: float = 0.0,
    verbose: bool = False,
) -> SparseGMRFBelief:
    """
    Update the Gaussian belief approximation after one episode.
    The posterior approximation becomes the prior for the next episode.
    """
    out = sparse_gmrf_softplus_update(
        belief=belief,
        obs_list=episode_obs,
        sigma_theta=sigma_theta,
        a=a,
        b=b,
        verbose=verbose,
    )
    new_belief = SparseGMRFBelief(
        H=belief.H,
        W=belief.W,
        mean=out["z_map"],
        precision=out["Q_post"],
    )
    return new_belief

def check_precision_matrix(Q, name="precision"):
    Q_dense = Q.toarray() if hasattr(Q, "toarray") else np.asarray(Q)
    Q_dense = 0.5 * (Q_dense + Q_dense.T)

    eigvals = np.linalg.eigvalsh(Q_dense)

    print(f"{name}:")
    print(f"  shape: {Q_dense.shape}")
    print(f"  symmetry error: {np.max(np.abs(Q_dense - Q_dense.T)):.3e}")
    print(f"  min eig: {eigvals[0]:.3e}")
    print(f"  max eig: {eigvals[-1]:.3e}")
    print(f"  condition number approx: {eigvals[-1] / max(eigvals[0], 1e-300):.3e}")
# ============================================================
# Sampling from belief for planning
# ============================================================

def sample_fields(
    belief: SparseGMRFBelief,
    n_samples: int,
    a: float = 1.0,
    b: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Sample latent fields and transformed theta fields from the current Gaussian belief.

    Returns:
        {
            "z_samples":     (n_samples, M)
            "theta_samples": (n_samples, M)
        }
    """
    # check_precision_matrix(belief.precision, "belief.precision before sampling")
    if rng is None:
        rng = np.random.default_rng()

    z_samples = sparse_cholesky_sample(
        mean=belief.mean,
        precision=belief.precision,
        n_samples=n_samples,
        rng=rng,
    )
    theta_samples = softplus(z_samples, a=a, b=b)

    return {
        "z_samples": z_samples,
        "theta_samples": theta_samples,
    }


# ============================================================
# Example planner stub
# ============================================================

def random_episode_plan(
    H: int,
    W: int,
    T: int,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[int, int]]:
    """
    Placeholder planner: randomly chooses T sites.
    Replace this with your MCTS / trajectory planner.
    """
    if rng is None:
        rng = np.random.default_rng()

    sites = []
    for _ in range(T):
        i = int(rng.integers(0, H))
        j = int(rng.integers(0, W))
        sites.append((i, j))
    return sites


# ============================================================
# Full sequential loop over episodes
# ============================================================

def run_sequential_episodes(
    H: int = 12,
    W: int = 12,
    K: int = 5,                    # number of episodes
    T: int = 8,                    # observations / decisions per episode
    tau_prior: float = 2.0,
    kappa_prior: float = 1e-2,
    sigma_theta: float = 0.01,
    a: float = 1.0,
    b: float = 0.0,
    planning_samples: int = 20,
    seed: int = 0,
    reward_cfg: Optional[RewardConfig] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    M = H * W

    # ----- prior / initial belief
    Q0 = make_4nb_laplacian_precision(H, W, tau=tau_prior, kappa=kappa_prior, periodic=False)
    m0 = np.zeros(M, dtype=float)

    belief = SparseGMRFBelief(H=H, W=W, mean=m0.copy(), precision=Q0.copy())

    # ----- true environment
    env = generate_true_field(
        H=H,
        W=W,
        prior_precision=Q0,
        mean=m0,
        a=a,
        b=b,
        rng=rng,
    )

    episode_records = []

    for k in range(K):
        if verbose:
            print(f"\n===== Episode {k+1}/{K} =====")

        # 1) sample from current belief for planning
        plan_samples = sample_fields(
            belief=belief,
            n_samples=planning_samples,
            a=a,
            b=b,
            rng=rng,
        )

        # 2) planner chooses next episode trajectory
        # Replace this stub with your actual planner using plan_samples
        visited_sites = random_episode_plan(H=H, W=W, T=T, rng=rng)

        # 3) environment generates observations along the chosen route
        episode_obs, _ = observe_from_true_field(
            env=env,
            visited_sites=visited_sites,
            sigma_theta=sigma_theta,
            rng=rng,
            reward_cfg=reward_cfg,
        )

        # 4) belief update at end of episode
        belief = update_belief_from_episode(
            belief=belief,
            episode_obs=episode_obs,
            sigma_theta=sigma_theta,
            a=a,
            b=b,
            verbose=False,
        )

        # summaries
        z_var = approximate_diag_of_inverse(belief.precision)
        theta_mean = softplus(belief.mean, a=a, b=b)

        record = {
            "episode": k + 1,
            "visited_sites": visited_sites,
            "episode_obs": episode_obs,
            "belief_mean_z": belief.mean.copy(),
            "belief_var_z": z_var.copy(),
            "belief_mean_theta_plugin": theta_mean.copy(),
            "planning_z_samples": plan_samples["z_samples"].copy(),
            "planning_theta_samples": plan_samples["theta_samples"].copy(),
        }
        episode_records.append(record)

        if verbose:
            print(f"Visited {len(visited_sites)} sites.")
            print(f"Belief mean theta range: [{theta_mean.min():.3f}, {theta_mean.max():.3f}]")

    return {
        "true_env": env,
        "new_belief": belief,
        "episode_records": episode_records,
    }


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    out = run_sequential_episodes(
        H=10,
        W=10,
        K=4,
        T=6,
        tau_prior=2.0,
        kappa_prior=1e-2,
        sigma_theta=0.01,
        a=1.0,
        b=0.0,
        planning_samples=10,
        seed=123,
        verbose=True,
    )

    env = out["true_env"]
    new_belief = out["new_belief"]

    print("\nTrue z field shape:", env.z_grid().shape)
    print("True theta field shape:", env.theta_grid().shape)
    print("Final belief mean shape:", new_belief.mean_grid().shape)