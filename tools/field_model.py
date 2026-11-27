"""Geometry of the curling field for tools/live_rerun.py (numpy only, no ROS, no Rerun).

Built from the measurements, with the lab2 Blender files as the model:
  field     assignment Fig. 7 (RoboCup SPL): 9 x 6 m, lines 5 cm (measured to the line
            centres), 0.7 m border strip, penalty area 1.65 x 4.0 m, goal area 0.6 x 2.2 m,
            centre circle 1.5 m, penalty mark 1.3 m from the goal line. Turf: noisy green
            like field.blend's noise texture.
  barrier   barrier.blend: 0.985 m long, jersey-barrier profile 0.55 m wide at the bottom,
            straight up to 0.144 m, sloping in to a 0.164 m top at 0.40 m. 3.5 cm apart.
  box       box.blend: black 0.30 m cube, white 0.208 m paper with the tag on one face.
  duck      duck.blend: yellow toy duck with orange beak, 4 cm high.

Frame = localization's field frame: (0, 0) centre spot, x towards the target goal, y left.
Every function returns plain arrays: positions (N, 3), triangles (M, 3), normals (N, 3).
"""
import math
import os
import sys

import numpy as np

# barriers + marker boxes: shared with the localization (lidar scan matching), one source of truth
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                'src', 'control_localization_pkg'))
from control_localization_pkg import field_map  # noqa: E402
from control_localization_pkg.field_map import (  # noqa: E402,F401
    CARPET_HALF, BARRIER_LENGTH, BARRIER_PROFILE, BARRIER_GAP, GOAL_GAP_HALF, BOX_SIZE, MARKER_NORMALS,
    barrier_layout)

# line centres (xmin, xmax, ymin, ymax): outer edge of each line = tag face of the boxes beside it
# (localization.py FIELD_BOUNDS, minus half the 5 cm line). -x end mirrored from +x (no box there).
FIELD_LINES = (-4.487, 4.487, -2.887, 2.994)
LINE_W = 0.05
CIRCLE_R = 0.75
PENALTY_AREA = (1.65, 2.0)       # depth from the goal line, half width
GOAL_AREA = (0.6, 1.1)
PENALTY_MARK_X = FIELD_LINES[1] - 1.3
SPOT_R = 0.05                    # penalty mark / centre spot: 100 mm

PAPER_SIZE = 0.2084
DUCK_HEIGHT = 0.04



class MeshBuilder:
    """Collects flat-shaded triangles (every face gets its own vertices -> crisp edges)."""

    def __init__(self):
        self.pos, self.nrm, self.col = [], [], []

    def polygon(self, pts, colour, normal=None, outward_from=None):
        """Convex polygon as a triangle fan. `outward_from`: a point inside the solid, so the
        face is turned to point away from it (correct lighting whatever the corner order)."""
        pts = np.asarray(pts, float)
        if normal is None:
            normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            normal = normal / max(np.linalg.norm(normal), 1e-12)
            if outward_from is not None and np.dot(normal, pts.mean(0) - outward_from) < 0:
                pts, normal = pts[::-1], -normal
        for k in range(1, len(pts) - 1):
            self.pos += [pts[0], pts[k], pts[k + 1]]
            self.nrm += [normal] * 3
            self.col += [colour] * 3

    def arrays(self):
        n = len(self.pos)
        return (np.array(self.pos, np.float32), np.arange(n, dtype=np.uint32).reshape(-1, 3),
                np.array(self.nrm, np.float32), np.array(self.col, np.uint8))


def _noise(x, y, seed=0):
    """Smooth value noise in [-1, 1] (a few octaves of interpolated random grids)."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(x)
    for cell, weight in ((1.3, 0.5), (0.45, 0.3), (0.15, 0.2)):
        gx, gy = np.floor(x / cell).astype(int), np.floor(y / cell).astype(int)
        fx, fy = x / cell - gx, y / cell - gy
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        size = 512
        table = rng.uniform(-1, 1, (size, size))
        g = lambda i, j: table[i % size, j % size]
        out += weight * ((g(gx, gy) * (1 - fx) + g(gx + 1, gy) * fx) * (1 - fy)
                         + (g(gx, gy + 1) * (1 - fx) + g(gx + 1, gy + 1) * fx) * fy)
    return out


def turf(z=-0.004, step=0.1, colour=(70, 112, 44)):
    """Carpet as a grid mesh with noisy green vertex colours (like field.blend's noise texture)."""
    cx, cy = CARPET_HALF
    xs, ys = np.arange(-cx, cx + 1e-9, step), np.arange(-cy, cy + 1e-9, step)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pos = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1).astype(np.float32)
    n = 1.0 + 0.13 * _noise(X.ravel(), Y.ravel(), seed=1)
    col = np.clip(np.outer(n, colour), 0, 255).astype(np.uint8)
    nx, ny = len(xs), len(ys)
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing='ij')
    a = (i * ny + j).ravel()
    tris = np.concatenate([np.stack([a, a + ny, a + ny + 1], 1), np.stack([a, a + ny + 1, a + 1], 1)]).astype(np.uint32)
    return pos, tris, np.tile(np.float32([0, 0, 1]), (len(pos), 1)), col


def lines(z=-0.002, colour=(232, 232, 226)):
    """All white field lines and spots, 5 cm wide, centred on the measured positions."""
    mb, up, w = MeshBuilder(), np.array([0.0, 0.0, 1.0]), LINE_W / 2

    def segment(p, q):
        p, q = np.array(p, float), np.array(q, float)
        d = (q - p) / np.linalg.norm(q - p)
        p, q = p - d * w, q + d * w                      # overlap at the corners
        s = np.array([-d[1], d[0]]) * w
        mb.polygon([[*(p - s), z], [*(q - s), z], [*(q + s), z], [*(p + s), z]], colour, up)

    def rect(x0, y0, x1, y1):
        for p, q in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            segment(p, q)

    def ring(r, n=96):
        a = np.linspace(0, 2 * math.pi, n + 1)
        for k in range(n):
            c0, s0, c1, s1 = math.cos(a[k]), math.sin(a[k]), math.cos(a[k + 1]), math.sin(a[k + 1])
            mb.polygon([[(r - w) * c0, (r - w) * s0, z], [(r + w) * c0, (r + w) * s0, z],
                        [(r + w) * c1, (r + w) * s1, z], [(r - w) * c1, (r - w) * s1, z]], colour, up)

    def spot(x, y, n=24):
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        mb.polygon([[x + SPOT_R * math.cos(t), y + SPOT_R * math.sin(t), z] for t in a], colour, up)

    x0, x1, y0, y1 = FIELD_LINES
    rect(x0, y0, x1, y1)
    segment((0, y0), (0, y1))
    ring(CIRCLE_R)
    spot(0, 0)
    for side in (-1, 1):
        gx = FIELD_LINES[1] if side > 0 else FIELD_LINES[0]
        rect(gx, -PENALTY_AREA[1], gx - side * PENALTY_AREA[0], PENALTY_AREA[1])
        rect(gx, -GOAL_AREA[1], gx - side * GOAL_AREA[0], GOAL_AREA[1])
        spot(side * PENALTY_MARK_X, 0)
    return mb.arrays()


def measured_barriers(segments, colour=(196, 38, 32), top_colour=(222, 70, 60), depth=0.40, height=0.40):
    """Measured barrier faces (config barrier_segments) as simple red blocks behind each face
    (the face towards the field centre)."""
    mb = MeshBuilder()
    for a, b in np.asarray(segments, float).reshape(-1, 2, 2):
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        n = np.array([-d[1], d[0]])
        if np.dot(n, (a + b) / 2) < 0:
            n = -n                                         # outward, away from the field
        a2, b2 = a + n * depth, b + n * depth
        inside = np.array([*((a + b2) / 2), height / 2])
        bottom = [a, b, b2, a2]
        P = lambda p, h: [p[0], p[1], h]
        mb.polygon([P(p, height) for p in bottom], top_colour, outward_from=inside)
        for k in range(4):
            p, q = bottom[k], bottom[(k + 1) % 4]
            mb.polygon([P(p, 0), P(q, 0), P(q, height), P(p, height)], colour, outward_from=inside)
    return mb.arrays()


def barriers(colour=(196, 38, 32), top_colour=(222, 70, 60), segments=None):
    """Jersey-barrier prisms (barrier.blend profile) at barrier_layout(), or the measured
    barrier faces when `segments` is given."""
    if segments is not None:
        return measured_barriers(segments, colour, top_colour)
    mb = MeshBuilder()
    prof = np.array(BARRIER_PROFILE)
    half = BARRIER_LENGTH / 2
    for x, y, along_x in barrier_layout():
        def P(u, v, h):          # u along the barrier, v across, h up -> field coordinates
            return [x + u, y + v, h] if along_x else [x - v, y + u, h]
        inside = np.array(P(0.0, 0.0, 0.15))
        # end caps (hexagon)
        mb.polygon([P(half, v, h) for v, h in prof], colour, outward_from=inside)
        mb.polygon([P(-half, v, h) for v, h in prof], colour, outward_from=inside)
        # sides around the profile
        for k in range(len(prof)):
            (v0, h0), (v1, h1) = prof[k], prof[(k + 1) % len(prof)]
            if h0 == 0 and h1 == 0:
                continue                                    # bottom: never seen
            c = top_colour if h0 == h1 else colour          # flat top a bit lighter
            mb.polygon([P(-half, v0, h0), P(half, v0, h0), P(half, v1, h1), P(-half, v1, h1)], c, outward_from=inside)
    return mb.arrays()


def marker_boxes(markers, tag_size):
    """Per marker id: box centre, yaw, paper centre, tag centre. The marker map gives the tag
    (front face of the box); the box stands behind it, facing MARKER_NORMALS."""
    out = {}
    h = BOX_SIZE / 2
    for mid, (x, y) in markers.items():
        nx, ny = MARKER_NORMALS.get(mid, (-1.0, 0.0))
        out[mid] = dict(centre=[x - nx * h, y - ny * h, h], yaw=math.atan2(ny, nx),
                        paper=[x + nx * 0.0005, y + ny * 0.0005, h],
                        tag=[x + nx * 0.0012, y + ny * 0.0012, h], tag_size=tag_size)
    return out


def lidar_map(markers, lidar_height, barriers=None):
    """Line segments the localization matches the lidar scan against: [[(x, y, z), (x, y, z)], ...]."""
    segs, _ = field_map.map_segments(markers, lidar_height=lidar_height, barriers=barriers)
    return [[[a[0], a[1], lidar_height], [b[0], b[1], lidar_height]] for a, b in segs]


def duck(x, y, yaw=math.pi):
    """Toy duck (ellipsoids): body, head, tail, beak. Facing `yaw` (default -x: the start side).
    Returns (centres, half_sizes, colours)."""
    s = DUCK_HEIGHT / 0.04
    parts = [  # (forward, left, up) centre, half sizes, colour
        ((0.000, 0.0, 0.012), (0.024, 0.017, 0.012), (250, 205, 20)),   # body
        ((-0.020, 0.0, 0.019), (0.007, 0.008, 0.006), (250, 205, 20)),  # tail
        ((0.013, 0.0, 0.030), (0.010, 0.010, 0.010), (250, 210, 30)),   # head (top at 4 cm)
        ((0.025, 0.0, 0.028), (0.006, 0.0045, 0.0025), (245, 120, 0)),  # beak
    ]
    c, sn = math.cos(yaw), math.sin(yaw)
    centres = [[x + s * (f * c - l * sn), y + s * (f * sn + l * c), s * u] for (f, l, u), _, _ in parts]
    return centres, [[s * a for a in hs] for _, hs, _ in parts], [col for _, _, col in parts], yaw


# goals (RoboCup SPL rule book, approx.): inner width 1.5 m, posts/crossbar 10 cm, height 0.8 m,
# net depth 0.5 m behind the goal line
GOAL_INNER_HALF = 0.75
GOAL_POST_R = 0.05
GOAL_HEIGHT = 0.80
GOAL_DEPTH = 0.50


def goals():
    """Both goals: frame as cylinders (centres, lengths, radii, rotations as (axis, angle)),
    net as thin lines. Returns (cylinders dict, net line strips)."""
    cyl = dict(centers=[], lengths=[], radii=[], rotations=[])
    net = []
    yp = GOAL_INNER_HALF + GOAL_POST_R                  # post centre
    h, d = GOAL_HEIGHT, GOAL_DEPTH

    def bar(p, q, r):
        p, q = np.array(p, float), np.array(q, float)
        v = q - p
        length = float(np.linalg.norm(v))
        axis = np.cross([0.0, 0.0, 1.0], v / length)     # Cylinders3D are along z: turn z onto v
        ang = math.acos(np.clip(v[2] / length, -1, 1))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.array([1.0, 0.0, 0.0])
        cyl['centers'].append(((p + q) / 2).tolist())
        cyl['lengths'].append(length)
        cyl['radii'].append(r)
        cyl['rotations'].append((axis / np.linalg.norm(axis), ang))

    for s in (-1, 1):
        gx = FIELD_LINES[1] if s > 0 else FIELD_LINES[0]
        bx = gx + s * d
        for sy in (-1, 1):
            bar((gx, sy * yp, 0), (gx, sy * yp, h), GOAL_POST_R)                 # front posts
            bar((bx, sy * yp, 0), (bx, sy * yp, h), 0.015)                       # back posts
            bar((gx, sy * yp, h), (bx, sy * yp, h), 0.015)                       # top sides
            bar((gx, sy * yp, 0.01), (bx, sy * yp, 0.01), 0.015)                 # ground sides
        bar((gx, -yp - GOAL_POST_R, h), (gx, yp + GOAL_POST_R, h), GOAL_POST_R)  # crossbar
        bar((bx, -yp, h), (bx, yp, h), 0.015)
        bar((bx, -yp, 0.01), (bx, yp, 0.01), 0.015)
        # net: 10 cm mesh on the back, the sides and the top
        for y in np.arange(-yp, yp + 1e-9, 0.1):
            net.append([[bx, y, 0], [bx, y, h], [gx, y, h]])
        for z in np.arange(0.1, h, 0.1):
            net.append([[gx, -yp, z], [bx, -yp, z], [bx, yp, z], [gx, yp, z]])
        for x in np.arange(0.1, d, 0.1):
            xx = gx + s * x
            net.append([[xx, -yp, 0], [xx, -yp, h], [xx, yp, h], [xx, yp, 0]])
    return cyl, net
