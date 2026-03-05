import numpy as np
import math
import random

# Node = namedtuple('Node', ['state', 'parent', 'children', 'visits', 'total_reward', 'untried_actions'])

class MCTSNode:
    def __init__(self, state, parent=None, children={}, visits=0, total_reward=0.0, untried_actions=None):
        self.state = state
        self.parent = parent
        self.children = children
        self.visits = visits
        self.total_reward = total_reward
        self.untried_actions = untried_actions
        
    def ucb_score(self, c=1.4):
        if self.visits == 0:
            return float('inf')
        return self.total_reward / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)


class MCTS:
    def __init__(self, reward_map, T, start, gamma=1.0, cpuct=2.):
        self.reward_map = reward_map
        self.T = T
        self.start = start
        self.gamma = gamma
        self.cpuct = cpuct
        self.grid_shape = reward_map.shape
        self.reset()

    def reset(self):
        x0, y0 = self.start
        initial_count = np.zeros(self.grid_shape, dtype=np.int32)
        state = (x0, y0, initial_count.tobytes(), 0)
        self.root = self._new_node(state)
        self.nodes = {state: self.root}

    def _new_node(self, state, parent=None):
        return MCTSNode(state, parent, {}, 0, 0.0, None)

    def get_neighbors(self, x, y):
        neighbors = []
        for dx, dy, a in [(-1, 0, 'L'), (1, 0, 'R'), (0, -1, 'U'), (0, 1, 'D')]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_shape[0] and 0 <= ny < self.grid_shape[1]:
                neighbors.append(((nx, ny), a))
        return neighbors

    def ucb_node(self, node):
        if node.visits == 0:
            return float('inf')
        q = node.total_reward / node.visits
        parent_visits = node.parent.visits if node.parent else node.visits
        print(self.cpuct * np.sqrt(np.log(parent_visits) / node.visits), q)
        return q + self.cpuct * np.sqrt(np.log(parent_visits) / node.visits)

    def ucb(self, node, action):
        child = node.children[action]
        if child.visits == 0:
            return float('inf')
        q = child.total_reward / child.visits
        return q + self.cpuct * np.sqrt(np.log(node.visits + 1) / child.visits)

    def select(self):
        node = self.root
        path = []
        while node.children and not node.untried_actions and self._time(node) < self.T:
            action = max(node.children, key=lambda a: self.ucb(node, a))
            node = node.children[action]
            path.append(node)
        return node, path

    def _time(self, node):
        return node.state[3]

    def _count_array(self, node):
        return np.frombuffer(node.state[2], dtype=np.int32).reshape(self.grid_shape)

    def expand(self, node):
        x, y, count_bytes, t = node.state
        count = np.frombuffer(count_bytes, dtype=np.int32).reshape(self.grid_shape)
        if node.untried_actions is None:
            node.untried_actions = [
                nbr for nbr in self.get_neighbors(x, y) if count[nbr[0][0], nbr[0][1]] == 0]
        if not node.untried_actions:
            print('return without expansion')
            return node
        next_pos = random.choice(node.untried_actions)
        (nx, ny), a = next_pos
        node.untried_actions.remove(next_pos)
        new_count = count.copy()
        new_count[nx, ny] = 1
        s_new = (nx, ny, new_count.tobytes(), t + 1)
        if s_new not in self.nodes.keys():
            self.nodes[s_new] = self._new_node(s_new, node)
            node.children[a] = self.nodes[s_new]
        else:
            print(f"Node {s_new} already exists, reusing it")
        return node.children[a]

    def simulate(self, node):
        x, y, count_bytes, t = node.state
        count = np.frombuffer(count_bytes, dtype=np.int32).reshape(self.grid_shape).copy()
        total_reward = 0.0
        discount = 1.0

        for step in range(t, self.T):
            next_cells = self.get_neighbors(x, y)
            actions = [(nx, ny, a) for (nx, ny), a in next_cells if count[nx, ny] == 0]
            if not actions:
                break
            nx, ny, a = random.choice(actions)
            reward = self.reward_map[nx, ny]
            count[nx, ny] = 1
            total_reward += discount * reward
            discount *= self.gamma
            x, y = nx, ny

        if (x, y) == self.start:
            total_reward += discount * self.reward_map[x, y]
        else:
            total_reward -= discount * 5  # penalty

        return total_reward

    def backpropagate(self, path, reward):
        for node in path:
            node.visits += 1
            node.total_reward += reward
            # self.nodes[node.state].visits += 1
            # self.nodes[node.state].total_reward += reward

    def run(self, num_simulations=1000):
        for step in range(num_simulations):
            if step % 1000 == 0 and step > 0:
                path, rewards = self.best_closed_loop_path()
                print(f"Simulation {step}/{num_simulations} with best rewards {rewards}")
                visualize_tree_on_grid(planner.root, planner.reward_map, best_path=path)
            node, path = self.select()
            if self._time(node) < self.T:
                new_node = self.expand(node)
                path.append(new_node)
            reward = self.simulate(new_node)
            self.backpropagate([self.root] + path, reward)
        return self.best_closed_loop_path()
    
    def get_ucb_best_closed_loop_path(self):
        """
        Follow the path from the root by selecting children with the highest UCB score.
        Stop when a closed loop of length T+1 is found (returns to root position).
        """
        root = self.root
        path = [root.state[:2]]
        node = root
        start_pos = self.start
        total_rewards = 0.0
        
        while True:
            if node.state[-1] == self.T:
                if node.state[:2] == start_pos:
                    return path, total_rewards
                else:
                    # reached max depth but not closed loop
                    print("Reached max depth without closed loop")
                    return path, total_rewards
            
            if not node.children:
                print("No children to explore, returning current path and rewards")
                return path, total_rewards
            
            def ucb(n):
                if n.visits == 0:
                    return float('inf')
                avg = n.total_reward / n.visits
                bonus = self.cpuct * (np.sqrt(np.log(node.visits) / n.visits)) if node.visits > 0 else 0
                return avg + bonus

            # Pick child with highest UCB
            best_child = max(node.children.values(), key=ucb)
            path.append(best_child.state[:2])
            total_rewards += self.reward_map[best_child.state[0], best_child.state[1]]
            node = best_child
            
    def best_closed_loop_path(self):
        terminal_nodes = [
            node for state, node in self.nodes.items()
            if sum(np.frombuffer(state[2], dtype=np.int32)) == self.T and state[:2] == self.start
        ]
        if not terminal_nodes:
            return None, None

        best_node = max(
            terminal_nodes,
            key=lambda n: n.total_reward / n.visits if n.visits > 0 else -np.inf
        )

        # Reconstruct path backward
        path = []
        total_rewards = 0.0
        node = best_node
        while node.parent is not None:
            path.append(node.state[:2])
            total_rewards += self.reward_map[node.state[0], node.state[1]]
            node = node.parent
        path.append(self.root.state[:2])
        path.reverse()

        return path, total_rewards


if __name__ == '__main__':
    import sys
    import os
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.path.append('/scratch/Rabbit/work/ABC_SMC_RL/')
    sys.path.append('/Users/guojiaqi/work/ABC_SMC_RL/')
    from Environment.Grid import IrregularGridWorld
    from utils import plot_path, visualize_tree_on_grid
    from parameter import SEED
    np.random.seed(SEED)
    random.seed(SEED)
    
    # Example usage

    rows, cols = 6, 6
    start = (2, 2)
    obstacles = np.zeros((rows, cols), dtype=bool)
    # Create environment
    env = IrregularGridWorld((rows, cols), starting_position=start)
        
    planner = MCTS(env.rewards, T=10, start=start)
    path, rewards = planner.run(num_simulations=1000)
    print("Best path:", path, "Total reward:", rewards)
    if path:
        plot_path(planner.reward_map, path)
    # visualize_tree_on_grid(planner.root, planner.reward_map, best_path=path)
