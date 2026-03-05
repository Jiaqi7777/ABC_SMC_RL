import numpy as np

def brute_force_max_path_gridworld(start_state, T, rewards):
    """
    Find the best path in GridWorld of exactly T steps without revisiting.
    
    Parameters:
        start_state (tuple): (x, y) coordinates
        T (int): number of steps
        grid (list of list): 2D grid representation
        rewards: array[state] → reward
        
    Returns:
        (max_reward, best_path): max cumulative reward and path taken (list of states)
    """
    rows, cols = len(rewards), len(rewards[0])
    best_reward = float('-inf')
    best_path = []
    best_actions = []

    def is_valid(state):
        x, y = state
        return 0 <= x < rows and 0 <= y < cols and rewards[x][y] != '#'

    def get_neighbors(state):
        x, y = state
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        neighbors = []
        actions = []
        for i, (dx, dy) in enumerate(directions):
            next_state = (x + dx, y + dy)
            if is_valid(next_state):
                neighbors.append(next_state)
                actions.append(i)
        return neighbors, actions
    
    def dfs(state, path, actions, visited, total_reward, depth):
        # print('state', state, path, depth)
        nonlocal best_reward, best_path, best_actions
        # print(path, total_reward)
        if depth == T:
            if state == start_state and total_reward > best_reward:
                best_reward = total_reward
                best_path = path[:]
                best_actions = actions[:]
            return

        for next_state, next_action in zip(*get_neighbors(state)):
            is_revisiting_start = next_state == start_state and depth == T - 1
            if next_state in visited and not is_revisiting_start:
                continue
            # print('next_state', next_state)
            reward = rewards[state[0], state[1]]
            added = False
            if not is_revisiting_start:
                visited.add(next_state)
                added = True

            dfs(next_state, path + [next_state], actions + [next_action], visited, total_reward + reward, depth + 1)

            if added:
                visited.remove(next_state)

    dfs(start_state, [start_state], [], set([start_state]), 0, 0)
    return best_path, best_actions, best_reward


if __name__ == '__main__':
    import sys, os
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.path.append('/scratch/Rabbit/work/ABC_SMC_RL/')
    sys.path.append('/Users/guojiaqi/work/ABC_SMC_RL/')
    from Environment.Grid import IrregularGridWorld
    from utils import plot_path
    from parameter import SEED
    np.random.seed(SEED)
    # Example usage

    rows, cols = 16, 16
    obstacles = np.zeros((rows, cols), dtype=bool)
    # Define stochastic reward distributions

    def normal_reward(mu, sigma):
        return lambda: np.random.normal(mu, sigma)
    
    # Create environment
    start = (2, 2)
    env = IrregularGridWorld((rows, cols), obstacles, starting_position=start)
    
    T = 50
    path, actions, reward = brute_force_max_path_gridworld(start, T, env.rewards)
    print("Best path:", path)
    print("Total reward:", reward)
    plot_path(env.rewards, path)
