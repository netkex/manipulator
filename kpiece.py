from typing import List, Tuple

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
    def __init__(self, parent, coords: np.ndarray, motions: List[Motion] = None, inner_grid=None, iter=2, mindist=100, selection_rule='original'):
        self.parent = parent
        self.motions = motions
        self.coords = coords
        self.inner_grid = inner_grid
        self.iter = iter
        self.s = 1
        self.mindist = mindist
        self.selection_rule = selection_rule

    def importance(self):
        n = self.parent.cnt_neighbors(self)
        if self.motions is not None:
            c = len(self.motions)
        else:
            c = len(self.inner_grid.cells)
        if self.selection_rule == 'shortest':
            return 1 / self.mindist
        elif self.selection_rule == 'original':
            return np.log(self.iter) / (self.s * (n + 1) * c)
        elif self.selection_rule == 'mixed':
            return np.log(self.iter) / (self.s * (n + 1) * c * self.mindist)
        raise "wrong selection rule"


class Grid:
    eps = 1e-6
    bias = 0.8

    def __init__(self, level: int, lbounds: np.ndarray, edge_size: float, k: int, map: Map, selection_rule: str):
        self.level = level
        self.lbounds = lbounds
        self.edge_size = edge_size
        self.k = k
        self.d = len(lbounds)
        self.cells = []
        self.all_codes = set()
        self.map = map
        self.selection_rule = selection_rule

    def _add_motion(self, motion: Motion, cc: int):
        coords = np.floor_divide(motion.start_node.state.joint_angles - self.lbounds, self.edge_size)
        cell = None
        for elem in self.cells:
            if np.abs(elem.coords - coords).sum() < Grid.eps:
                cell = elem
                break
        if cell is None:
            cell = GridCell(self, coords, iter=cc, mindist=self.map.dist_to_finish(motion.start_node.state) ** 2)
            self.cells.append(cell)
            self.all_codes.add(self._coords_to_code(coords))
            if self.level > 1:
                cell.inner_grid = Grid(self.level - 1, self.lbounds + coords * self.edge_size, self.edge_size / self.k, self.k, self.map, self.selection_rule)
            else:
                cell.motions = []
        if self.level > 1:
            cell.inner_grid.add_motion(motion, cc)
        else:
            cell.motions.append(motion)
        cell.mindist = min(cell.mindist, self.map.dist_to_finish(motion.start_node.state) ** 2)

    def add_motion(self, motion: Motion, cc: int):
        coords = np.floor_divide(motion.start_node.state.joint_angles - self.lbounds, self.edge_size)
        init_pos = motion.start_node.state.joint_angles[motion.joint]
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
        for i in range(self.d):
            new_coords = cell.coords.copy()
            new_coords[i] = (new_coords[i] + 1) % self.k
            if self._coords_to_code(new_coords) in self.all_codes:
                cnt += 1
            new_coords[i] = (cell.coords[i] - 1 + self.k) % self.k
            if self._coords_to_code(new_coords) in self.all_codes:
                cnt += 1
        return cnt

    def _split_int_ext(self):
        interior, exterior = [], []
        for elem in self.cells:
            if self.cnt_neighbors(elem) == self.d * 2:
                interior.append(elem)
            else:
                exterior.append(elem)
        return interior, exterior

    def select_motion(self) -> Tuple[Motion, GridCell]:
        interior, exterior = self._split_int_ext()
        if np.random.rand() < Grid.bias:
            cells = exterior
        else:
            cells = interior

        if len(cells) == 0:
            cells = self.cells

        best = None
        for elem in cells:
            if best is None or elem.importance() > best.importance():
                best = elem
        best.s += 1
        if self.level > 1:
            return best.inner_grid.select_motion()

        sz = len(best.motions)
        ind = min(sz - 1, int((ss.halfnorm(scale=sz/6).rvs())))
        return best.motions[max(sz - ind - 1, 0)], best


def kpiece(manip: Manipulator, map: Map, grid_k=2, lvls=None, selection_rule='mixed'):
    if lvls is None:
        lvls = np.ceil(np.log(2 * np.pi / 0.1) / np.log(grid_k))
    grid = Grid(lvls, np.zeros(manip.joint_num), edge_size=2*np.pi/grid_k, k=grid_k, map=map, selection_rule=selection_rule)
    start_node = Node(manip)
    grid.add_motion(Motion(start_node, 0, 1, 0.0001), 2)
    cc = 1
    while True:
        cc += 1
        motion, cell = grid.select_motion()
        t = np.random.rand() * motion.dur_max
        nxt_state = motion.start_node.state.apply(motion.joint, motion.dir, t)
        nxt_node = Node(nxt_state, motion.start_node.g + t, parent=motion.start_node)
        joint_ind = np.random.randint(0, manip.joint_num)
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
                return dump_node
            max_dur += manip.angle_delta

        if max_dur > 0:
            grid.add_motion(Motion(nxt_node, joint_ind, dir, max_dur), cc)
