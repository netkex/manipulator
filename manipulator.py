from typing import Optional, List
import numpy as np


class Manipulator():
    def __init__(self, 
        joint_num: int, 
        angle_delta: Optional[float] = np.pi / 180,
        joint_angles: Optional[List[float]] = None, 
        arm_len: Optional[List[float]] = None):
        assert (joint_num >= 3) 

        self.joint_num = joint_num
        self.angle_delta = angle_delta

        if joint_angles is None: 
            self.joint_angles = [0] + [np.pi] * (joint_num - 1)
        else: 
            self.joint_angles = joint_angles

        if arm_len is None: 
            self.arm_len = [1] * (joint_num - 1)
        else: 
            self.arm_len = arm_len
        
        self.joint_coordinates = None  # lazy init
        
    def get_joint_coordinates(self) -> np.ndarray: 
        if self.joint_coordinates is not None: 
            return self.joint_coordinates
        self.joint_coordinates = np.zeros((self.joint_num, 3))
        self.joint_coordinates[1] = np.array([
            self.arm_len[0] * np.cos(self.joint_angles[1]), 
            0, 
            self.arm_len[0] * np.sin(self.joint_angles[1])
        ])
        for i in range(1, self.joint_num - 1): 
            prev_arm_vec = self.joint_coordinates[i] - self.joint_coordinates[i - 1]
            new_arm_vec = prev_arm_vec / np.linalg.norm(prev_arm_vec) * self.arm_len[i]
            new_arm_vec[[0, 2]] = Manipulator.rotation_matrix_2d(np.pi + self.joint_angles[i + 1]) @ new_arm_vec[[0, 2]] 
            self.joint_coordinates[i + 1] = self.joint_coordinates[i] + new_arm_vec
        turn_around_rot = Manipulator.rotation_matrix_2d(self.joint_angles[0])
        for i in range(0, self.joint_num):
            self.joint_coordinates[i][[0, 1]] = turn_around_rot @ self.joint_coordinates[i][[0, 1]]
        return self.joint_coordinates

    def copy_with_new_angles(self, new_angles):
        return Manipulator(
            self.joint_num,
            self.angle_delta,
            new_angles,
            self.arm_len,
        )

    def get_successors(self):
        successors = []
        for i in range(self.joint_num):
            for dangle in [-self.angle_delta, self.angle_delta]:
                new_joint_angle = self.joint_angles[i] + dangle
                if new_joint_angle < 0:
                    new_joint_angle += 2 * np.pi 
                if new_joint_angle > 2 * np.pi: 
                    new_joint_angle -= 2 * np.pi 
                successors.append(self.copy_with_new_angles(
                    self.joint_angles[:i] + [new_joint_angle] + self.joint_angles[i+1:]
                ))
        return successors

    def calc_angle_distance(self, other):
        return np.abs(np.array(self.joint_angles) - np.array(other.joint_angles)).sum()

    @staticmethod
    def rotation_matrix_2d(angle: float) -> np.ndarray: 
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])