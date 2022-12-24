from manipulator import Manipulator
from abc import ABC, abstractmethod
import numpy as np

class Obstacle(ABC):
    @abstractmethod
    def intersect(self, manipulator: Manipulator) -> bool:
        pass


class SphereObstacle(Obstacle): 
    def __init__(self, center: np.ndarray, r: float): 
        self.center = center
        self.r = r 

    def intersect(self, manipulator: Manipulator) -> bool:
        joint_coordinates = manipulator.get_joint_coordinates() 
        for i in range(len(joint_coordinates) - 1): 
            if np.linalg.norm(joint_coordinates[i] - self.center) < self.r or np.linalg.norm(joint_coordinates[i + 1] - self.center) < self.r: 
                return True
            arm_vec = joint_coordinates[i + 1] - joint_coordinates[i] 
            t = (self.center @ arm_vec - joint_coordinates[i] @ arm_vec) / (np.linalg.norm(arm_vec) ** 2)  # arm contains all points: joint_coordinates[i] + t * arm_vec, if t \in [0, 1]
            if 0 <= t <= 1 and np.linalg.norm(joint_coordinates[i] + arm_vec * t - self.center) < self.r:
                return True
        return False
