import numpy as np
import random
import math
import sys
import os
import copy

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # (position, path)
        assert state[0] == state[1][-1]
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = None

    def ucb_score(self, c=1.4):
        if self.visits == 0:
            return float('inf')
        return self.total_reward / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

class MCTSPlanner:
    def __init__(self, reward_map, T, start):
        self.grid = reward_map
        self.T = T
        self.start = start
        self.rows, self.cols = reward_map.shape
        self.directions = [(-1,0), (1,0), (0,-1), (0,1)]

    def is_valid(self, i, j):
        return 0 <= i < self.rows and 0 <= j < self.cols and not np.isnan(self.grid[i, j])

    @staticmethod
    def min_steps_to_start(pos, start):
        # Manhattan distance
        return abs(pos[0] - start[0]) + abs(pos[1] - start[1])

    def get_neighbors(self, pos):
        i, j = pos
        return [
            (i + di, j + dj)
            for di, dj in self.directions
            if self.is_valid(i + di, j + dj)
        ]

    def run(self, simulations=1000):
        root = MCTSNode((self.start, [self.start]))

        for _ in range(simulations):
            node = root
            # 1. Selection
            while node.children and not node.untried_actions:
                # print([n.ucb_score() for n in root.children])
                node = max(node.children, key=lambda n: n.ucb_score())
            print(node.state, f"Selected node: {node.state[0]} with untried actions: {node.untried_actions}")
            # 2. Expansion
            if node.untried_actions is None:
                node.untried_actions = [
                    nbr for nbr in self.get_neighbors(node.state[0])
                    if (len(node.state[1]) < self.T and nbr != self.start)or (nbr == self.start and len(node.state[1]) == self.T)
                    and nbr not in node.state[1]
                ]
                print(node.state, f"Untried actions: {node.untried_actions}")

            if node.untried_actions:
                next_pos = random.choice(node.untried_actions)
                node.untried_actions.remove(next_pos)
                new_path = node.state[1] + [next_pos]
                child = MCTSNode((next_pos, new_path), parent=node)
                node.children.append(child)
                node = child

            # 3. Simulation (Rollout)
            reward = self.simulate(node.state)
            
            # 4. Backpropagation
            while node:
                if reward != -1:
                    # print(node.state, f"Backpropagating reward: {reward}, visits: {node.visits}")
                    node.visits += 1
                    node.total_reward += reward
                node = node.parent
        return root
        # Return best terminal path from root
        best_final = max(
            (child for child in root.children if len(child.state[1]) == self.T and child.state[0] == self.start),
            key=lambda n: n.total_reward / n.visits if n.visits > 0 else -np.inf,
            default=None
        )
        return best_final.state if best_final else None

    def simulate(self, state):
        print("Simulating state:", state[0])
        pos, path = copy.deepcopy(state)
        visited = set(path)
        total_reward = sum(self.grid[i, j] for i, j in path)

        while len(path) < self.T + 1:
            remaining_steps = self.T + 1 - len(path)

            # Prune infeasible
            if remaining_steps < self.min_steps_to_start(pos, self.start):
                # print("Infeasible path:", path)
                pos = path[-2]
                neighbors = self.get_neighbors(pos)
                neighbors.remove(path[-1])
                return -10
            else:
                neighbors = self.get_neighbors(pos)

            # Valid moves: unvisited, or return to start at last step
            valid = [
                nbr for nbr in neighbors
                if (nbr not in visited and remaining_steps > 1)
                or (nbr == self.start and remaining_steps == 1)
            ]
            # print(remaining_steps)
            if not valid:
                print("No valid moves from", pos)
                return -10

            # When close to end, prefer moving closer to start
            if remaining_steps <= self.T // 2 + 1:
                valid.sort(key=lambda n: self.min_steps_to_start(n, self.start))
                next_pos = valid[0]
                # print("Choosing next position based on distance to start:", pos, next_pos)
            else:
                next_pos = random.choice(valid)
            # print(pos, path)
            path.append(next_pos)
            visited.add(next_pos)
            total_reward += self.grid[next_pos[0], next_pos[1]]
            pos = next_pos
        # print(pos, path)
        if path[-1] != self.start:
            return -10  # path is invalid
        return total_reward
    
    def get_best_closed_loop(self, root):
        def is_valid_terminal(node):
            return (
                len(node.state[1]) == self.T + 1 and
                node.state[0] == self.start
            )

        # Recursively collect all nodes in the tree
        def collect_all_nodes(node):
            nodes = [node]
            for child in node.children:
                nodes.extend(collect_all_nodes(child))
            return nodes

        all_nodes = collect_all_nodes(root)
        # return [node.state for node in all_nodes if is_valid_terminal(node)]
        return all_nodes
        best_final = max(
            (node for node in all_nodes if is_valid_terminal(node)),
            key=lambda n: n.total_reward / n.visits if n.visits > 0 else -np.inf,
            default=None
        )

        return best_final.state if best_final else None


if __name__ == '__main__':
    import sys
    import os
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.path.append('/scratch/Rabbit/work/ABC_SMC_RL/')
    sys.path.append('/Users/guojiaqi/work/ABC_SMC_RL/')
    from Environment.Grid import IrregularGridWorld

    # Example usage

    rows, cols = 6, 6
    obstacles = np.zeros((rows, cols), dtype=bool)
    # Create irregular shape
    # obstacles[3:5, 2:7] = True
    # obstacles[7, :] = True
    # obstacles[:, 0] = True
    # obstacles[0, :] = True

    # Define stochastic reward distributions

    def normal_reward(mu, sigma):
        return lambda: np.random.normal(mu, sigma)
    
    # Create environment
    env = IrregularGridWorld((rows, cols), obstacles)
        
    planner = MCTSPlanner(env.rewards, T=8, start=(2, 2))
    root = planner.run(simulations=100)
    result = (planner.get_best_closed_loop(root))
    if result:
        end_pos, path = result
        print("Best path:", path)
        print("Total reward:", sum(env.rewards[i, j] for i, j in path))
    else:
        print("No valid closed loop found")
