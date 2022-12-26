from manipulator import Manipulator
import numpy as np


class Node:
    '''
    Node class represents a search node

    - g: g-value of the node
    - h: h-value of the node
    - F: f-value of the node
    - parent: pointer to the parent-node

    '''
    eps = 1e-6

    def __init__(self, state: Manipulator, g=0, h=0, f=None, parent=None):
        self.state = state
        self.g = g
        self.h = h
        if f is None:
            self.f = self.g + h
        else:
            self.f = f
        self.parent = parent

    def __eq__(self, other: Manipulator):
        '''
        Estimating where the two search nodes are the same,
        which is needed to detect dublicates in the search tree.
        '''
        angles1 = np.array(self.state.joint_angles)
        angles2 = np.array(other.joint_angles)
        return np.abs(angles1 - angles2).max() < Node.eps

    def __hash__(self):
        '''
        To implement CLOSED as set of nodes we need Node to be hashable.
        '''
        return hash(str([int(angle / self.eps) for angle in self.state.joint_angles]))

    def __lt__(self, other):
        '''
        Comparing the keys (i.e. the f-values) of two nodes,
        which is needed to sort/extract the best element from OPEN.
        '''
        if self.f == other.f:
            return self.g > other.g

        return self.f < other.f