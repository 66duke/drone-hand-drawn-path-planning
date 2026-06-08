import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon,LineString,MultiLineString
from shapely.affinity import rotate
from scipy.interpolate import splprep,splev
import json

#参数区#
angle = 20
BoundaryOffset = 0.5
spacing = 0.8
#参数区#

#曲线函数区
def generate_boundary(n=500):
    points = []
    for theta in np.linspace(0, 2*np.pi, n):
        r = (
            5
            + 1.2*np.sin(theta)
            - 0.6*np.sin(2*theta)
            + 0.4*np.sin(4*theta)
        )
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        points.append((x, y))
    return Polygon(points)
polygon = generate_boundary()
#曲线函数区

#旋转
def rotate_polygon(polygon, angle_deg):
    rotated = rotate(
        polygon,
        -angle_deg,
        origin='centroid'
    )
    return rotated
rotated_polygon = rotate_polygon(
    polygon,
    angle
)
#旋转

#区域内缩
def offset_polygon(polygon, offset):
    safe_polygon = polygon.buffer(-offset)
    if safe_polygon.is_empty:
        raise ValueError(
            "Offset too large"
        )
    return safe_polygon
safe_polygon = offset_polygon(
    rotated_polygon,
    BoundaryOffset
)
#区域内缩

#扫描线
def generate_scan_lines(
        polygon,
        spacing):
    minx, miny, maxx, maxy = polygon.bounds
    margin = spacing
    scan_lines = []
    y_values = np.arange(
        miny,
        maxy + spacing,
        spacing
    )
    for y in y_values:
        line = LineString([
            (minx - margin, y),
            (maxx + margin, y)
        ])
        scan_lines.append(line)
    return scan_lines
scan_lines = generate_scan_lines(
    safe_polygon,
    spacing
)
#扫描线

#扫描
def generate_segments(
        safe_polygon,
        scan_lines):
    segments = []
    for line in scan_lines:
        inter = safe_polygon.intersection(line)
        if inter.is_empty:
            continue
        if inter.geom_type == "LineString":
            segments.append(inter)
        elif inter.geom_type == "MultiLineString":
            for seg in inter.geoms:
                segments.append(seg)
    return segments
segments = generate_segments(
    safe_polygon,
    scan_lines
)
#扫描

#生成路径
def generate_path(segments):
    path = []
    for i, seg in enumerate(segments):
        coords = list(seg.coords)
        left = coords[0]
        right = coords[-1]
        if i % 2 == 0:
            path.append(left)
            path.append(right)
        else:
            path.append(right)
            path.append(left)
    return path
path = generate_path(segments)
#path内是一堆航路点 关键:是点！
#生成路径

#平滑化
def move_towards(p_from, p_to, dist):
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    L = np.hypot(dx, dy)
    if L < 1e-6:
        return p_to
    ratio = dist / L
    return (
        p_to[0] - dx * ratio,
        p_to[1] - dy * ratio
    )
def local_bspline(
        p0,
        p1,
        p2,
        p3,
        n=30):
    x = [
        p0[0],
        p1[0],
        p2[0],
        p3[0]
    ]
    y = [
        p0[1],
        p1[1],
        p2[1],
        p3[1]
    ]
    tck, u = splprep(
        [x, y],
        s=0,
        k=3
    )
    u_new = np.linspace(0, 1, n)
    x_new, y_new = splev(
        u_new,
        tck
    )
    return list(zip(x_new, y_new))
def smooth_turns(
        path,
        turn_length=0.5,
        samples=25):
    smooth = []
    turn_curves = []
    smooth.append(path[0])
    i = 1
    while i < len(path)-2:
        prev_pt = path[i-1]
        turn_start = path[i]
        turn_end = path[i+1]
        next_pt = path[i+2]
        q1 = move_towards(
            prev_pt,
            turn_start,
            turn_length
        )
        q2 = move_towards(
            next_pt,
            turn_end,
            turn_length
        )
        smooth.append(q1)
        curve = local_bspline(
            q1,
            turn_start,
            turn_end,
            q2,
            n=samples
        )
        smooth.extend(curve)
        smooth.append(q2)
        turn_curves.append(curve)
        i += 2
    smooth.append(path[-1])
    return smooth, turn_curves
smooth_waypoints,turn_curves = smooth_turns(
    path,
    turn_length=spacing*0.5,
    samples=40
)
#平滑化

#旋转
waypoints = path
def rotate_point(
        point,
        center,
        angle_deg):
    angle = np.radians(angle_deg)
    x, y = point
    cx, cy = center
    dx = x - cx
    dy = y - cy
    xr = (
        dx*np.cos(angle)
        - dy*np.sin(angle)
    )
    yr = (
        dx*np.sin(angle)
        + dy*np.cos(angle)
    )
    return (
        xr + cx,
        yr + cy
    )
center = (
    polygon.centroid.x,
    polygon.centroid.y
)
final_waypoints = []
for p in smooth_waypoints:
    p_rot = rotate_point(
        p,
        center,
        angle
    )
    final_waypoints.append(p_rot)
#旋转

#最终可视化
x,y = polygon.exterior.xy
plt.plot(
    x,
    y,
    'k-',
    linewidth=2
)

px = [p[0] for p in final_waypoints]
py = [p[1] for p in final_waypoints]
plt.plot(
    px,
    py,
    'r-',
    linewidth=2
)
plt.scatter(
    px,
    py,
    s=10
)
plt.axis('equal')
plt.show()
#最终可视化

#导出路径点
data = []
for i, p in enumerate(final_waypoints):
    data.append({
        "id": i,
        "x": float(p[0]),
        "y": float(p[1])
    })
with open("巡检轨迹点.json", "w") as f:
    json.dump(data, f, indent=4)
data = []
for i, p in enumerate(path):
    data.append({
        "id": i,
        "x": float(p[0]),
        "y": float(p[1])
    })
with open("航路点.json", "w") as f:
    json.dump(data, f, indent=4)
