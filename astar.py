from map import Map
from node import Node
from manipulator import Manipulator
from search_tree import DefaultSearchTree
import numpy as np

def default_heuristic_func(manipulator, finish):
	coordinates = manipulator.get_joint_coordinates()
	dist = np.linalg.norm(coordinates[-1] - finish)
	r = np.max(np.linalg.norm(coordinates - coordinates[-1]))

	if dist > 2 * r:
		return np.pi

	return np.arccos(1 - dist ** 2 / (2 * r ** 2))

def astar(initial_state: Manipulator, grid_map: Map, heuristic_func=default_heuristic_func, search_tree=DefaultSearchTree):
	ast = search_tree()
	ast.add_to_open(Node(initial_state, h=heuristic_func(initial_state, grid_map.finish)))

	last_node = None
	while not ast.open_is_empty():
		current_node = ast.get_best_node_from_open()
		if current_node is None:
			break

		if grid_map.is_in_finish(current_node.state):
			last_node = current_node
			break

		for next_state in current_node.state.get_successors():
			if not grid_map.valid(next_state):
				continue

			next_node = Node(
				next_state, 
				g=current_node.g + current_node.state.calc_angle_distance(next_state), 
				h=heuristic_func(next_state, grid_map.finish),
				parent=current_node
			)

			if not ast.was_expanded(next_node):
				ast.add_to_open(next_node)


	return last_node

