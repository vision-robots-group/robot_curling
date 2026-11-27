"""Static geometry of the curling field (numpy only, no ROS).

Single source of truth for the barriers and the marker boxes: used by the localization
(lidar scan matching) and by tools/field_model.py (Rerun drawing).

Frame = localization's field frame: (0, 0) centre spot, x towards the target goal, y left.
"""
import json
import math

import numpy as np

CARPET_HALF = (5.2, 3.7)         # carpet half size; the barriers stand just outside it
BARRIER_LENGTH = 0.985
# jersey-barrier cross-section (across, up): 0.55 m wide at the bottom, straight up to 0.144 m,
# sloping in to a 0.164 m top at 0.40 m (barrier.blend)
BARRIER_PROFILE = [(-0.275, 0.0), (0.275, 0.0), (0.275, 0.144), (0.082, 0.40), (-0.082, 0.40), (-0.275, 0.144)]
BARRIER_GAP = 0.035
GOAL_GAP_HALF = 0.8              # short sides: no barriers within +-0.8 m of the goal (placement guess)

BOX_SIZE = 0.30

# direction each tag faces (into the field), per physical marker id
MARKER_NORMALS = {0: (-1.0, 0.0), 1: (0.0, 1.0), 2: (0.0, -1.0), 3: (0.0, 1.0), 4: (0.0, -1.0), 5: (-1.0, 0.0)}


def barrier_depth():
    return BARRIER_PROFILE[1][0] - BARRIER_PROFILE[0][0]


def barrier_layout():
    """[(centre x, centre y, along_x)] of the barriers, just outside the carpet, 3.5 cm apart:
    both long sides full length, the short sides from the corners up to the goal.
    The placement is a guess from the photos: change it here if the lab is different
    (tools/calibrate_lidar.py prints how far each wall is from this model)."""
    depth = barrier_depth()
    cx, cy = CARPET_HALF
    step = BARRIER_LENGTH + BARRIER_GAP
    out = []
    n = int((2 * cx + BARRIER_GAP) // step)
    for side in (-1, 1):
        out += [((i - (n - 1) / 2) * step, side * (cy + depth / 2), True) for i in range(n)]
    n = int((cy - GOAL_GAP_HALF + BARRIER_GAP) // step)
    for side in (-1, 1):
        for sy in (-1, 1):
            out += [(side * (cx + depth / 2), sy * (cy - BARRIER_LENGTH / 2 - i * step), False) for i in range(n)]
    return out


def barrier_face_half(height):
    """Half width of the barrier cross-section at `height` (where a horizontal lidar beam hits it),
    or None if the beam passes over the barrier."""
    prof = BARRIER_PROFILE
    if height < 0 or height > prof[3][1]:
        return None
    if height <= prof[2][1]:
        return prof[2][0]
    f = (height - prof[2][1]) / (prof[3][1] - prof[2][1])
    return prof[2][0] + f * (prof[3][0] - prof[2][0])


def marker_box_square(tag_xy, normal):
    """Corners (4, 2) of a marker box seen from above; the tag is the front face, the box behind it."""
    nx, ny = normal
    h = BOX_SIZE / 2
    cx, cy = tag_xy[0] - nx * h, tag_xy[1] - ny * h
    yaw = math.atan2(ny, nx)
    c, s = math.cos(yaw), math.sin(yaw)
    local = [(h, h), (-h, h), (-h, -h), (h, -h)]
    return np.array([[cx + c * u - s * v, cy + s * u + c * v] for u, v in local])


def parse_barriers(text):
    """Measured barrier faces (config `barrier_segments`, JSON [[[x1, y1], [x2, y2]], ...],
    written by tools/map_barriers.py) -> (K, 2, 2) array, or None if not set."""
    if not text or not str(text).strip():
        return None
    segs = np.asarray(json.loads(text), dtype=float).reshape(-1, 2, 2)
    return segs if len(segs) else None


def map_segments(markers=None, lidar_height=0.20, boxes=True, barriers=None):
    """2-D line segments a horizontal lidar at `lidar_height` sees.

    markers: {physical id: (x, y) of the tag} (the boxes); None/empty = barriers only.
    barriers: measured barrier faces (K, 2, 2) from parse_barriers(); None = the guessed
    barrier_layout() (rectangle around the carpet).
    Returns (segments (N, 2, 2), labels [str] * N). Labels: 'barrier_<k>' (measured) or
    'wall_left' (y > 0 long side), 'wall_right', 'wall_goal_left' ... (guessed), and 'box_<id>'.
    """
    segs, labels = [], []
    fh = barrier_face_half(lidar_height)
    cx, cy = CARPET_HALF
    half = BARRIER_LENGTH / 2
    if barriers is not None:
        for k, seg in enumerate(np.asarray(barriers, dtype=float).reshape(-1, 2, 2)):
            segs.append(seg)
            labels.append(f'barrier_{k}')
    elif fh is not None:
        for x, y, along_x in barrier_layout():
            if along_x:
                side = 1 if y > 0 else -1
                fy = y - side * fh                     # face towards the field
                segs.append([[x - half, fy], [x + half, fy]])
                labels.append('wall_left' if side > 0 else 'wall_right')
            else:
                side = 1 if x > 0 else -1
                fx = x - side * fh
                segs.append([[fx, y - half], [fx, y + half]])
                end = 'goal' if side > 0 else 'start'
                labels.append(f'wall_{end}_{"left" if y > 0 else "right"}')
    if boxes and markers and lidar_height <= BOX_SIZE:
        for mid, xy in markers.items():
            corners = marker_box_square(xy, MARKER_NORMALS.get(int(mid), (-1.0, 0.0)))
            for k in range(4):
                segs.append([corners[k], corners[(k + 1) % 4]])
                labels.append(f'box_{int(mid)}')
    return np.array(segs, dtype=float).reshape(-1, 2, 2), labels


def markers_by_id(marker_map):
    """{'family:id': (x, y)} (localization's marker_map) -> {physical id: (x, y)}."""
    out = {}
    for key, xy in marker_map.items():
        out.setdefault(int(str(key).rsplit(':', 1)[-1]), (float(xy[0]), float(xy[1])))
    return out
