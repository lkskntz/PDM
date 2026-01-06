"""
rrt_planner.py

A generic 3D RRT (Rapidly-exploring Random Tree) global planner.

Author: [Your Name]
Date: 2025-12-15

Usage:
    from rrt_planner import RRT3D

    def collision_fn(pos):
        # Return True if `pos` is collision-free
        pass

    planner = RRT3D(start=[0,0,0], goal=[5,5,2], bounds=[[-7,7],[-5,5],[0,5]], is_collision_free=collision_fn)
    path = planner.plan(max_iter=2000, step_size=0.5)
"""

import numpy as np
from scipy.spatial import KDTree


class RRTNode:
    def __init__(self, pos, parent=None):
        self.pos = np.array(pos)
        self.parent = parent


class RRT3D:
    def __init__(self, start, goal, bounds, is_collision_free):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.bounds = bounds
        self.is_collision_free = is_collision_free
        self.nodes = [RRTNode(self.start)]
        self.vertices = []  # list of edges as tuples: (start_pos, end_pos)

    def plan(self, max_iter=1000, step_size=0.5, goal_sample_rate=0.1):
        for i in range(max_iter):
            # Sample point
            if np.random.rand() < goal_sample_rate:
                rand_point = self.goal
            else:
                rand_point = np.array([np.random.uniform(low, high) for low, high in self.bounds])

            # Find nearest node
            tree_positions = np.array([n.pos for n in self.nodes])
            tree = KDTree(tree_positions)
            _, idx = tree.query(rand_point)
            nearest_node = self.nodes[idx]

            # Extend towards sampled point
            direction = rand_point - nearest_node.pos
            distance = np.linalg.norm(direction)
            if distance == 0:
                continue
            direction = direction / distance * min(step_size, distance)
            new_pos = nearest_node.pos + direction

            # Collision check
            if not self.is_collision_free(new_pos):
                continue

            # Add new node
            new_node = RRTNode(new_pos, parent=nearest_node)
            self.nodes.append(new_node)
            self.vertices.append((nearest_node.pos.tolist(), new_node.pos.tolist()))  # add edge

            # Check if goal is reached
            if np.linalg.norm(new_pos - self.goal) <= step_size:
                goal_node = RRTNode(self.goal, parent=new_node)
                self.nodes.append(goal_node)
                self.vertices.append((new_node.pos.tolist(), goal_node.pos.tolist()))
                return self._reconstruct_path(goal_node)

        return None

    def _reconstruct_path(self, node):
        path = []
        while node is not None:
            path.append(node.pos.tolist())
            node = node.parent
        return path[::-1]

    def get_tree_positions(self):
        """
        Returns all nodes in the tree as a position and parent
        """
        return [n.pos.tolist() for n in self.nodes]

    def get_tree_vertices(self):
        """
        Returns all edges in the tree as pairs of positions [[start, end], ...]
        """
        return self.vertices

