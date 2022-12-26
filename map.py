from typing import List

import numpy as np

from manipulator import Manipulator
from obstacle import Obstacle


class Map:

    eps = 1e-6

    def __init__(self, obstacles: List[Obstacle], finish: np.ndarray, finish_size):
        '''
        Default constructor
        '''

        self.obstacles = obstacles
        self.finish = finish
        self.finish_size = finish_size

    def valid(self, manipulator: Manipulator):
        if manipulator.get_joint_coordinates()[:,2].min() < -Map.eps:
            return False

        for obs in self.obstacles:
            if obs.intersect(manipulator):
                return False
        return True

    def dist_to_finish(self, manipulator: Manipulator):
        return np.sqrt(((manipulator.get_joint_coordinates()[-1] - self.finish) ** 2).sum())

    def is_in_finish(self, manipulator: Manipulator):
        return self.dist_to_finish(manipulator) < self.finish_size
