from typing import List

import numpy as np
import scipy.stats as ss

from manipulator import Manipulator
from map import Map
from node import Node


class Motion:
    def __init__(self, start_node: Node, joint: int, dir: int, dur_max: float):
        self.start_node = start_node
        self.joint = joint
        self.dir = dir
        self.dur_max = dur_max


class GridCell:
    def __init__(self, parent, coords: np.ndarray, motions: List[Motion] = None, inner_grid=None, iter=2):
        self.parent = parent
        self.motions = motions
        self.coords = coords
        self.inner_grid = inner_grid
        self.iter = 2
        self.s = 1

    def importance(self):
        n = self.parent.cnt_neighbors(self)
        c = 1
        if self.motions is not None:
            c = len(self.motions)
        else:
            c = len(self.inner_grid.cells)
        return np.log(self.iter) / (self.s * (n + 1) * c)


    def __le__(self, other):
        return self.importance() > other.importance()


class Grid:
    eps = 1e-6
    bias = 0.8

    def __init__(self, level: int, lbounds: np.ndarray, edge_size: float, k: int = 2):
        self.level = level
        self.lbounds = lbounds
        self.edge_size = edge_size
        self.k = k
        self.cells = []
        self.all_codes = set()

    def _add_motion(self, motion: Motion, cc: int):
        coords = np.floor_divide(motion.start_node.state.joint_angles - self.lbounds, self.edge_size)
        #print(coords)
        cell = None
        for elem in self.cells:
            if np.abs(elem.coords - coords).sum() < Grid.eps:
                cell = elem
                break
        #print("angles", motion.start_node.state.joint_angles, "level = ", self.level, "coords = ", coords, "lbounds = ", self.lbounds)
        if cell is None:
            cell = GridCell(self, coords, iter=cc)
            self.cells.append(cell)
            self.all_codes.add(self._coords_to_code(coords))
            if self.level > 1:
                cell.inner_grid = Grid(self.level - 1, self.lbounds + coords * self.edge_size, self.edge_size / self.k, self.k)
            else:
                cell.motions = []
        if self.level > 1:
            cell.inner_grid.add_motion(motion, cc)
        else:
            cell.motions.append(motion)

    def add_motion(self, motion: Motion, cc: int):
        coords = np.floor_divide(motion.start_node.state.joint_angles - self.lbounds, self.edge_size)
        init_pos = motion.start_node.state.joint_angles[motion.joint]
        tm_needed = None
        if motion.dir == 1:
            tm_needed = self.lbounds[motion.joint] + self.edge_size * (coords[motion.joint] + 1) - init_pos
        else:
            tm_needed = init_pos - (self.lbounds[motion.joint] + self.edge_size * coords[motion.joint])

        assert tm_needed >= 0
        eps = 1e-4
        if tm_needed < motion.dur_max:
            dump_node = motion.start_node
            cntdump = round((tm_needed + eps) / motion.start_node.state.angle_delta) + 1
            for i in range(cntdump):
                dump_node = Node(dump_node.state.apply(motion.joint, motion.dir, (tm_needed + eps) / cntdump), dump_node.g + (tm_needed + eps) / cntdump, parent=dump_node)
            self._add_motion(Motion(motion.start_node, motion.joint, motion.dir, tm_needed - eps), cc)
            self.add_motion(Motion(dump_node, motion.joint, motion.dir, motion.dur_max - tm_needed - eps), cc)
        else:
            self._add_motion(motion, cc)

    def _coords_to_code(self, coords: List[int]):
        res = 0
        for elem in coords:
            res *= self.k
            res += elem
        return res

    def cnt_neighbors(self, cell: GridCell):
        cnt = 0
        for i in range(self.k):
            new_coords = cell.coords.copy()
            new_coords[i] = (new_coords[i] + 1) % self.k
            if self._coords_to_code(new_coords) in self.all_codes:
                cnt += 1
            new_coords[i] = (cell.coords[i] - 1 + self.k) % self.k
            if self._coords_to_code(new_coords) in self.all_codes:
                cnt += 1
        #print("Cnt neighbors = ", cnt)
        return cnt

    def _split_int_ext(self):
        interior, exterior = [], []
        for elem in self.cells:
            if self.cnt_neighbors(elem) == self.k * 2:
                interior.append(elem)
            else:
                exterior.append(elem)
        #print("int/ext = ", len(interior), len(exterior))
        return interior, exterior

    def select_motion(self) -> (Motion, GridCell):
        interior, exterior = self._split_int_ext()
        cells = None
        if np.random.rand() < Grid.bias:
            cells = exterior
        else:
            cells = interior

        if len(cells) == 0:
            cells = self.cells

        best = None
        #print(len(cells))
        for elem in cells:
            if best is None or elem.importance() > best.importance():
                best = elem
        best.s += 1
        if self.level > 1:
            return best.inner_grid.select_motion()

        sz = len(best.motions)
        #print("size = ", sz)
        #ind = np.random.randint(0, sz)
        #return best.motions[ind], best
        ind = int((ss.halfnorm().rvs()))
        return best.motions[max(sz - ind - 1, 0)], best


def kpiece(manip: Manipulator, map: Map, grid_k=2):
    curlen = maxlen = 2 * np.pi
    lvls = 1
    while curlen > map.finish_size:
        lvls += 1
        curlen /= grid_k
    grid = Grid(lvls, np.zeros(manip.joint_num), edge_size=2*np.pi/grid_k, k=grid_k)
    start_node = Node(manip)
    grid.add_motion(Motion(start_node, 0, 0, 0.0001), 2)
    cc = 0
    while True:
        cc += 1
        motion, cell = grid.select_motion()
        #print(motion, motion.dur_max, motion.start_node.state.get_joint_coordinates())
        t = np.random.rand() * motion.dur_max
        nxt_state = motion.start_node.state.apply(motion.joint, motion.dir, t)
        nxt_node = Node(nxt_state, motion.start_node.g + t, parent=motion.start_node)
        joint_ind = np.random.randint(0, manip.joint_num - 1)
        dir = 1
        if np.random.rand() > 0.5:
            dir = -1
        max_dur = 0
        dump_node = nxt_node
        while max_dur + manip.angle_delta < np.pi:
            after_move = nxt_state.apply(joint_ind, dir, max_dur + manip.angle_delta)
            dump_node = Node(after_move, dump_node.g + manip.angle_delta, parent=dump_node)
            if not map.valid(after_move):
                break
            if map.is_in_finish(after_move):
                return dump_node#Node(after_move, motion.start_node.g + max_dur + manip.angle_delta, parent=motion.start_node)
            max_dur += manip.angle_delta

        if max_dur > 0:
            grid.add_motion(Motion(nxt_node, joint_ind, dir, max_dur), cc)
