import numpy as np
import sys
import os
import gc

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.append('/scratch/Rabbit/work/ABC_SMC_RL/')
sys.path.append('/Users/guojiaqi/work/ABC_SMC_RL/')
print('system path', sys.path)

from Environment.Grid import *
from model import *
from parameter import *
from MCMC_Algorithms.INLA import *
from Planning.MCTS_ep import *
from Planning.Brute_Force_Search import *
from Environment.env_util import *
from Environment.rewards import *
    
def run_sequential_episodes(
    env,
    K: int = 50,                    # number of episodes
    T: int = 12,                    # observations / decisions per episode
    tau_prior: float = 2.0,
    kappa_prior: float = 1e-2,
    sigma_theta: float = 0.2,
    a: float = 1.0,
    b: float = 0.0,
    planning_sims: int = 100,
    reward_cfg: RewardConfig = RewardConfig(),
    proposal_cfg: ProposalConfig = ProposalConfig(),
    reuse_decay: float = 0.5,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    H, W = env.n_cell
    M = H * W
    start_pos = env.start_pos
    # ----- prior / initial belief
    Q0 = make_4nb_laplacian_precision(H, W, tau=tau_prior, kappa=kappa_prior, periodic=False)
    m0 = np.zeros(M, dtype=float)

    belief = SparseGMRFBelief(H=H, W=W, mean=m0.copy(), precision=Q0.copy())

    # ----- true environment
    true_field = generate_true_field(
        H=H,
        W=W,
        prior_precision=Q0,
        mean=m0,
        a=a,
        b=b,
        rng=rng,
    )

    episode_records = []
    episode_returns = []
    total_return = 0.0
    planner = PWMCTSTheta2(
        T=T, K=K, gamma=0.99,
        c_ucb=1.0,
        k_pw=2.0, alpha_pw=0.5,
        alpha_score=0.7,   # uncertainty bonus weight in K-score
        p_flip=0.05,
        rng=random.Random(seed),
        reward_cfg=reward_cfg,
        proposal_cfg=proposal_cfg,
    )
    for k in range(K):
        if verbose:
            print(f"\n===== Episode {k+1}/{K} =====")

        root_id = planner._get_or_create_root(start_pos)
        root = planner.tree[root_id]

        # 1) planner chooses next episode trajectory
        # Replace this stub with your actual planner using plan_samples
        sample_fn = lambda belief, n_samples: sample_fields(belief, n_samples, a=a, b=b, rng=rng)
        var_fn = lambda belief: theta_posterior_delta_method(belief, a=a, b=b)
        plans, plan_weights = planner.plan(env, root_pos=root.s0, belief=belief, num_sims=planning_sims, sample_fn=sample_fn, var_fn=var_fn)
        # execute first action (receding horizon) or execute the full plan if you want
        U = rng.choice(plans, p=plan_weights/sum(plan_weights))
        visited_sites = env.rollout(U)
        planner.reuse_subtree_from_selected_plan(
            selected_plan=U,
            decay=reuse_decay,
            reduce_horizon=True,
        )
        

        # 2) environment generates observations along the chosen route
        episode_obs, episode_rewards = observe_from_true_field(
            true_field=true_field,
            visited_sites=visited_sites,
            sigma_theta=sigma_theta,
            rng=rng,
            reward_cfg=reward_cfg,
        )
        total_return += sum(episode_rewards)
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
            "episode_rewards": episode_rewards,
            "belief_mean_z": belief.mean.copy(),
            "belief_var_z": z_var.copy(),
            "belief_mean_theta_plugin": theta_mean.copy(),
        }
        episode_records.append(record)
        episode_returns.append(sum(episode_rewards))

        if verbose:
            print(f"Visited {visited_sites}")
            print(f"Episode {k+1} return: {sum(episode_rewards):.4f}, total return: {total_return:.4f}, rewards: {[f'{x:.2f}' for x in episode_rewards]}")
            # print(f"Belief mean theta range: [{theta_mean.min():.3f}, {theta_mean.max():.3f}]")
            visualize_path(
                env,
                true_field,
                start_pos,
                visited_sites,
                k,
                K
            )
    return {
        "true_field": true_field,
        "new_belief": belief,
        "episode_records": episode_records,
        "episode_returns": episode_returns,
    }


    
if __name__ == "__main__":
    K = 20
    T = 12
    rows, cols = 6, 6
    obstacles = np.zeros((rows, cols), dtype=bool)
    start_pos = (2, 2)
    repeat = 4
    
    # Create environment
    env = IrregularGridWorld((rows, cols), obstacles, starting_position=start_pos)
    reward_cfg = RewardConfig(
            loop_penalty=5.0,
            return_penalty=2.0,
            reward_scale=3.0,
            reward_power=2.0,
            use_loop_penalty=True,
            use_return_penalty=True,
        )
    alpha_score=0.7
    proposal_cfg = ProposalConfig(w_r=1.0, w_v=alpha_score, w_rep=2.0, w_inv=5.0, w_ret=0.6, ret_power=2.0, temperature=0.35)
    return_all_repeats = []
    for r in range(repeat):
        print(f"\n\n=== Repeat {r+1}/{repeat} ===")
        out = run_sequential_episodes(
            env=env,
            K=K,
            T=T,
            tau_prior=2.0,
            kappa_prior=1e-2,
            sigma_theta=0.1,
            a=1.0,
            b=0.0,
            planning_sims=1000,
            reward_cfg=reward_cfg,
            proposal_cfg=proposal_cfg,
            reuse_decay=0.8,
            seed=123+r,
            verbose=True if r == 0 else False,  # only verbose for first repeat
        )
        return_all_repeats.append(out["episode_returns"])
        true_field = out["true_field"]
        path, actions, reward = brute_force_max_path_gridworld(start_pos, T, true_field.theta_grid(), reward_cfg)  # optimal path for comparison
        print("Best path:", path)
        print("Total reward:", reward)
        plot_path(true_field.theta_grid(), path)
    plot_return(return_all_repeats)
        # new_belief = out["new_belief"]

        # print("\nTrue z field shape:", true_field.z_grid().shape)
        # print("True theta field shape:", true_field.theta_grid().shape)
        # print("Final belief mean shape:", new_belief.mean_grid().shape)
        