from Environment.env_util import *
@dataclass
class RewardConfig:
    loop_penalty: float = 5.0
    return_penalty: float = 2.0
    reward_scale: float = 1.0
    use_loop_penalty: bool = True
    use_return_penalty: bool = True
    reward_power: float = 2.0
    
def cell_value_theta_power(theta_field, pos: Pos, power: float = 2.0) -> float:
    """
    Returns theta(pos)^power.

    theta_field can either be:
    - a 2D array indexed by [i, j]
    - or something compatible with theta_field[pos]
    """
    return float(theta_field[pos]) ** power


def planning_step_reward(
    *,
    theta_field,
    sp: Pos,
    visited: Set[Pos],
    t: int,
    T: int,
    origin: Pos,
    cfg: RewardConfig,
) -> float:
    """
    Shared reward rule used by true environment evaluation, planning evaluation,
    and proposal scoring if desired.
    """
    r = cfg.reward_scale * cell_value_theta_power(theta_field, sp, cfg.reward_power)

    if cfg.use_loop_penalty:
        if sp in visited and t < T - 1:
            r -= cfg.loop_penalty

    return r

def terminal_return_penalty(
    *,
    final_pos: Pos,
    origin: Pos,
    cfg: RewardConfig,
) -> float:
    if not cfg.use_return_penalty:
        return 0.0

    dist = abs(final_pos[0] - origin[0]) + abs(final_pos[1] - origin[1])
    return cfg.return_penalty * dist

def eval_plan(
    env,
    s0: Pos,
    theta_field,
    U: Sequence[Action],
    gamma: float,
    cfg: RewardConfig,
) -> Tuple[float, Pos]:
    """
    Evaluate a plan under a given theta field or belief sample.

    Reward:
        sum_t gamma^t theta(s_{t+1})^2
        - loop penalties
        - terminal return penalty
    """
    s = s0
    visited = {s0}
    R = 0.0
    disc = 1.0
    T = len(U)

    for t, a in enumerate(U):
        sp = next_pos(env, s, a)

        r = planning_step_reward(
            theta_field=theta_field,
            sp=sp,
            visited=visited,
            t=t,
            T=T,
            origin=s0,
            cfg=cfg,
        )

        R += disc * r
        disc *= gamma

        visited.add(sp)
        s = sp

    R -= terminal_return_penalty(
        final_pos=s,
        origin=s0,
        cfg=cfg,
    )

    return R, s