from typing import List

import numpy as np

from manipulator import Manipulator
from obstacle import Obstacle


class Map:

    def __init__(self, obstacles: List[Obstacle], finish: np.ndarray, finish_size):
        '''
        Default constructor
        '''

        self.obstacles = obstacles
        self.finish = finish
        self.finish_size = finish_size

    def valid(self, manipulator: Manipulator):
        for obs in self.obstacles:
            if obs.intersect(manipulator):
                return False
        return True

    def is_in_finish(self, manipulator: Manipulator):
        position = manipulator.get_joint_coordinates()[-1]
        return np.abs(position - self.finish).max() < self.finish_size