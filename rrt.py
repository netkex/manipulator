from map import Map
from node import Node
from manipulator import Manipulator
import numpy as np


def _gen_random_state(manip: Manipulator, map: Map) -> np.ndarray:
    if np.random.rand(1)[0] < 0.1:
        return map.finish

    maxlen = sum(manip.arm_len)
    x, y, z = np.random.random(3)
    x = x * 2 * maxlen - maxlen
    y = y * 2 * maxlen - maxlen
    z *= maxlen
    return np.array([x, y, z])


def rrt(manip: Manipulator, map: Map):
    nodes = [Node(manip)]
    #print(map.obstacles[0].intersect(manip))
    #print(manip.get_joint_coordinates())
    while True:
        new_point = _gen_random_state(manip, map)
        prv = nodes[0]

        #print(len(nodes), new_point)
        for node in nodes:
            if node.state.calc_raw_distance(new_point) < prv.state.calc_raw_distance(new_point):
                prv = node

        best_approx = None
        for succ in prv.state.get_successors():
            #print(succ.calc_angle_distance(new_state))
            if map.valid(succ) and (best_approx is None or succ.calc_raw_distance(new_point) < best_approx.calc_raw_distance(new_point)):
                best_approx = succ
        #print()

        if best_approx is None:
            continue

        new_node = Node(best_approx, prv.g + prv.state.calc_angle_distance(best_approx), parent=prv)
        nodes.append(new_node)
        #print(best_approx.get_joint_coordinates()[-1])
        if map.is_in_finish(best_approx):
            print(len(nodes))
            return new_node