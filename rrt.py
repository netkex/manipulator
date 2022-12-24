from map import Map
from node import Node
from manipulator import Manipulator
import numpy as np


def _gen_random_state(manip: Manipulator, map: Map) -> Manipulator:
    new_state = None
    while new_state is None or not map.valid(new_state):
        new_state = manip.copy_with_new_angles(np.random.rand(manip.joint_num) * 2 * np.pi)
    return new_state


def rrt(manip: Manipulator, map: Map):
    nodes = [Node(manip)]
    #print(map.obstacles[0].intersect(manip))
    #print(manip.get_joint_coordinates())
    while True:
        new_state = _gen_random_state(manip, map)
        prv = nodes[0]
        #print(len(nodes))
        for node in nodes:
            if new_state.calc_angle_distance(node.state) < new_state.calc_angle_distance(prv.state):
                prv = node

        best_approx = None
        for succ in prv.state.get_successors():
            #print(succ.calc_angle_distance(new_state))
            if map.valid(succ) and (best_approx is None or succ.calc_angle_distance(new_state) < best_approx.calc_angle_distance(new_state)):
                best_approx = succ
        #print()

        if best_approx is None:
            continue

        new_node = Node(best_approx, prv.g + prv.state.calc_angle_distance(best_approx), parent=prv)
        nodes.append(new_node)
        #print(best_approx.get_joint_coordinates()[-1])
        if map.is_in_finish(best_approx):
            return new_node