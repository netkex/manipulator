from typing import List
from manipulator import Manipulator
from obstacle import Obstacle, SphereObstacle
import numpy as np
import scipy.stats as ss

eps = 1e-9

class ManipulatorTest:
    def __init__(self,
        init_manipulator: Manipulator,
        goal_point: np.ndarray, 
        obstacles: List[Obstacle],
        ) -> None:
        self.init_manipulator = init_manipulator
        self.goal_point = goal_point 
        self.obstacles = obstacles


def generate_test_obstacle_on_way(
    joint_num: int, 
    obstacles_num: int,
    sigma: float = 0.5, 
) -> ManipulatorTest: 
    goal = ss.uniform(0.1, joint_num / 2 - 0.1).rvs(size=3) * np.random.choice([-1, 1], 3)
    goal[2] = np.abs(goal[2])
    obstacles = []
    for _ in range(obstacles_num): 
        obstacle_noise = ss.norm(loc=0, scale=sigma).rvs(size=3)
        obstacle_center = goal * ss.uniform(0.1, 0.8).rvs(size=1) + obstacle_noise
        r_bound = min(np.linalg.norm(obstacle_center), np.linalg.norm(goal - obstacle_center))
        r = ss.norm(loc=r_bound / 2, scale=r_bound / 4).rvs(size=1)
        obstacles.append(SphereObstacle(center=obstacle_center, r=r))
    
    while True: 
        manipulator = gen_random_manipulator(joint_num) 
        intersect = False 
        for obstacle in obstacles: 
            if obstacle.intersect(manipulator):
                intersect = True
                break
        if not intersect: 
            break 
    
    return ManipulatorTest(
        init_manipulator=manipulator,
        goal_point=goal,
        obstacles=obstacles,
    )


def gen_random_manipulator(joint_num: int) -> Manipulator: 
    while (True):
        angles = ss.uniform(0, 2 * np.pi).rvs(size=1).tolist() + ss.uniform(0, np.pi).rvs(size=1).tolist() + ss.uniform(0, 2 * np.pi).rvs(size=joint_num - 2).tolist() 
        manipulator = Manipulator(joint_num=joint_num, joint_angles=angles)
        if np.all(manipulator.get_joint_coordinates()[:, 2] > -eps): 
            return manipulator
