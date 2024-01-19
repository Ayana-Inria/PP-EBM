from dataclasses import dataclass

import numpy as np
import shapely.geometry
from numpy.random import Generator
from scipy.spatial import distance_matrix

from base.geometry import rect_to_poly
from base.sampler2d import sample_point_2d
from data.synthetic.pointssimulator import PointsSimulator


class Align(PointsSimulator):

    def __init__(self, config):
        self.config = config
        self.sim_args = config.get('simulator_args', {})

        self.marks_min = np.array(self.config['marks_min'])
        self.marks_max = np.array(self.config['marks_max'])

        self.n_marks = len(self.marks_min)

        self.intensity = self.config['intensity']

        self.max_dist = self.marks_max[:2].max() * 3

    def make_points(self, rng: Generator, image) -> np.ndarray:
        if len(image.shape) == 3:
            image = np.mean(image, axis=-1)
        allowed_area = image < 0.5
        n_points = int(self.intensity * np.sum(allowed_area))
        locations = sample_point_2d(
            image.shape[:2], size=n_points, density=allowed_area)

        # add marks
        # marks = rng.uniform(self.marks_min, self.marks_max, size=(len(locations), self.n_marks))

        n_groups = self.config['n_groups']

        group_loc = sample_point_2d(image.shape[:2], size=n_groups)
        dist = distance_matrix(locations, group_loc)
        closest = np.argmin(dist, axis=-1)

        mark_per_group = rng.uniform(
            self.marks_min, self.marks_max, size=(n_groups, self.n_marks))
        marks = mark_per_group[closest]

        sigma = np.sqrt((self.marks_max - self.marks_min)) * 0.05
        marks = np.clip(marks + rng.normal(0, sigma,
                        size=marks.shape), self.marks_min, self.marks_max)

        points = np.concatenate([locations, marks], axis=-1)

        valid_points = []
        intersect_mode = self.sim_args.get('intersection_mode', 'rectangle')
        # reject overlaps
        if intersect_mode == 'rectangle':
            def shape_fn(x): return shapely.geometry.Polygon(
                rect_to_poly(x[:2], x[2], x[3], x[4]))
        elif intersect_mode == 'circle':
            def shape_fn(x): return shapely.geometry.Point(x[:2]).buffer(x[2])
        else:
            raise ValueError

        for p in points:
            shape_1 = shape_fn(p)
            flag = True
            for p2 in valid_points:
                # if np.linalg.norm(p[:2] - p2[:2]) > self.max_dist:
                #     continue
                shape_2 = shape_fn(p2)
                intersection = shape_1.intersection(shape_2).area
                union = shape_1.union(shape_2).area

                if union > 0 and intersection / union > 0.01:
                    flag = False
            if flag:
                valid_points.append(p)

        points = np.array(valid_points)

        return points


class LinesOfRect(PointsSimulator):

    def __init__(self, config):
        self.config = config
        self.sim_args = config.get('simulator_args', {})

        self.marks_min = np.array(self.config['marks_min'])
        self.marks_max = np.array(self.config['marks_max'])

        self.n_marks = len(self.marks_min)

        self.intensity = self.config['intensity']
        self.margin = config['margin']

    def make_points(self, rng: Generator, image) -> np.ndarray:
        if len(image.shape) == 3:
            image = np.mean(image, axis=-1)
        allowed_area = image < 0.5
        shape = image.shape[:2]

        n_lines = rng.integers(
            self.config['n_lines'][0], self.config['n_lines'][1])

        contour = shapely.Polygon(
            [[0, 0], [0, shape[1]], [shape[0], shape[1]], [shape[0], 0], [0, 0]])
        lines_p1_t = rng.random(size=n_lines)
        lines_p2_t = (np.ceil(lines_p1_t * 4) / 4 +
                      0.75 * rng.random(size=n_lines)) % 1
        lines_p1 = contour.boundary.line_interpolate_point(
            lines_p1_t, normalized=True)
        lines_p2 = contour.boundary.line_interpolate_point(
            lines_p2_t, normalized=True)
        lines_p1_np = np.concatenate([p.coords.__array__() for p in lines_p1])
        lines_p2_np = np.concatenate([p.coords.__array__() for p in lines_p2])
        n_point_per_line = rng.poisson(
            lam=self.intensity / n_lines, size=n_lines)

        mark_per_group = rng.uniform(
            self.marks_min[:2], self.marks_max[:2], size=(n_lines, 2))

        line_angles = np.arctan2(
            lines_p2_np[:, 1] - lines_p1_np[:, 1], lines_p2_np[:, 0] - lines_p1_np[:, 0])
        objet_angles = line_angles % np.pi
        spacing = mark_per_group[:, 0]

        locations = []
        marks = []
        for p1, p2, nb, s, a, m in zip(lines_p1, lines_p2, n_point_per_line, spacing, objet_angles, mark_per_group):
            line = shapely.LineString([p1, p2])
            length = line.length
            samples = np.arange(s, length - s, step=s + self.margin)
            n_pts = len(samples)
            if n_pts > 0:
                line_points = line.line_interpolate_point(samples)
                current_locations = np.concatenate(
                    [p.coords.__array__() for p in line_points])
                locations.append(current_locations)
                current_marks = np.tile(np.array([m[0], m[1], a]), (n_pts, 1))
                marks.append(current_marks)

        locations = np.concatenate(locations)
        marks = np.concatenate(marks)

        # group_loc = sample_point_2d(image.shape[:2], size=n_groups)
        # dist = distance_matrix(locations, group_loc)
        # closest = np.argmin(dist, axis=-1)

        points = np.concatenate([locations, marks], axis=-1)

        valid_points = []
        intersect_mode = self.sim_args.get('intersection_mode', 'rectangle')
        # reject overlaps
        if intersect_mode == 'rectangle':
            def shape_fn(x): return shapely.geometry.Polygon(
                rect_to_poly(x[:2], x[2], x[3], x[4]))
        elif intersect_mode == 'circle':
            def shape_fn(x): return shapely.geometry.Point(x[:2]).buffer(x[2])
        else:
            raise ValueError

        for p in points:
            shape_1 = shape_fn(p)
            flag = True
            for p2 in valid_points:
                # if np.linalg.norm(p[:2] - p2[:2]) > self.max_dist:
                #     continue
                shape_2 = shape_fn(p2)
                intersection = shape_1.intersection(shape_2).area
                union = shape_1.union(shape_2).area

                if union > 0 and intersection / union > 0.01:
                    flag = False
            if flag:
                valid_points.append(p)

        points = np.array(valid_points)

        return points


class CurvesOfRect(PointsSimulator):

    def __init__(self, config):
        self.config = config
        self.sim_args = config.get('simulator_args', {})

        self.marks_min = np.array(self.config['marks_min'])
        self.marks_max = np.array(self.config['marks_max'])

        self.n_marks = len(self.marks_min)

        self.intensity = self.config['intensity']
        self.margin = config['margin']
        self.curving_range = config['curving_range']

    def make_points(self, rng: Generator, image) -> np.ndarray:
        if len(image.shape) == 3:
            image = np.mean(image, axis=-1)
        allowed_area = image < 0.5
        shape = image.shape[:2]

        n_points = rng.poisson(lam=self.intensity * np.prod(shape))
        points = []
        curve_ended_flag = True
        curr_loc = None
        curr_heading = None
        curving = None
        curr_obj_size = None
        while len(points) < n_points:
            if curve_ended_flag:  # start new curve:
                starting_edge = rng.integers(0, 4)
                c1, c2, c3, c4 = np.array([0, 0]), np.array([0, shape[1]]), np.array([shape[0], shape[1]]), np.array(
                    [shape[0], 0])
                if starting_edge == 0:
                    p1, p2 = c1, c2
                    heading_range = [-0.5 * np.pi, 0.5 * np.pi]
                elif starting_edge == 1:
                    p1, p2 = c2, c3
                    heading_range = [0, np.pi]
                elif starting_edge == 2:
                    p1, p2 = c3, c4
                    heading_range = [0.5 * np.pi, 1.5 * np.pi]
                elif starting_edge == 3:
                    p1, p2 = c4, c1
                    heading_range = [np.pi, 2 * np.pi]
                else:
                    raise ValueError

                curving = rng.uniform(
                    self.curving_range[0], self.curving_range[1])
                x = rng.random()
                curr_loc = x * p1 + (1 - x) * p2
                curr_heading = rng.uniform(
                    heading_range[0], heading_range[1]) % (2 * np.pi)
                curr_obj_size = rng.uniform(
                    self.marks_min[:2], self.marks_max[:2], size=2)
                curve_ended_flag = False

            delta = (curr_obj_size[0] + self.margin) * \
                np.array([np.cos(curr_heading), np.sin(curr_heading)])
            curr_loc = curr_loc + delta
            curr_heading += curving

            inbound = curr_loc[0] >= 0 and curr_loc[0] <= shape[0] and curr_loc[1] >= 0 and curr_loc[1] <= shape[1]
            if inbound:
                new_pt = np.array([
                    curr_loc[0], curr_loc[1],
                    curr_obj_size[0], curr_obj_size[1], curr_heading % np.pi
                ])
                points.append(new_pt)
            else:
                curve_ended_flag = True
        if len(points) > 0:
            points = np.stack(points)
        else:
            points = np.empty((0, 5))

        # print(n_points)

        assert len(points) == n_points

        # prune overlaps
        valid_points = []
        intersect_mode = self.sim_args.get('intersection_mode', 'rectangle')
        assert intersect_mode == 'rectangle'

        def shape_fn(x): return shapely.geometry.Polygon(
            rect_to_poly(x[:2], x[2], x[3], x[4]))
        for p in points:
            shape_1 = shape_fn(p)
            flag = True
            for p2 in valid_points:
                # if np.linalg.norm(p[:2] - p2[:2]) > self.max_dist:
                #     continue
                shape_2 = shape_fn(p2)
                intersection = shape_1.intersection(shape_2).area
                union = shape_1.union(shape_2).area

                if union > 0 and intersection / union > 0.01:
                    flag = False
            if flag:
                valid_points.append(p)
        if len(points) > 0:
            points = np.stack(valid_points)
        else:
            points = np.empty((0, 5))

        return points
