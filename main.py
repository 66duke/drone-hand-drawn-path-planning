import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from shapely.validation import make_valid
import cv2

# ===================== Configure Matplotlib =====================
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.style.use('default')


# ===================== 【1】Contour Extraction =====================
def image_to_sketch_points(image_path, down_scale=1, line_width=3):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片：{image_path}")
    img = cv2.resize(img, (img.shape[1] // down_scale, img.shape[0] // down_scale))
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("未找到任何轮廓，请检查图片！")
    max_contour = max(contours, key=lambda c: len(c))
    inner_points = np.squeeze(max_contour).tolist()
    inner_polygon = Polygon(inner_points)
    outer_polygon = inner_polygon.buffer(line_width / 2, join_style=1)
    outer_points = np.array(outer_polygon.exterior.coords)
    outer_points = [[x, height - y] for x, y in outer_points]
    print(f"✅ 步骤0：原始内边缘提取成功，点数：{len(inner_points)}")
    print(f"✅ 步骤0修复：向外膨胀{line_width / 2}像素，得到区域外边界，点数：{len(outer_points)}")
    return np.array(outer_points)


# ===================== 【2】Preprocessing =====================
def moving_average_smooth(points, k=3):
    smoothed = np.copy(points)
    n = len(points)
    for i in range(1, n - 1):
        smoothed[i] = (points[i - 1] + 2 * points[i] + points[i + 1]) / 4
    print(f"✅ 步骤1：滑动平均平滑完成，点数：{len(smoothed)}")
    return smoothed


def douglas_peucker_simplify(points, tolerance=1.5):
    poly = LineString(points)
    simplified = poly.simplify(tolerance, preserve_topology=True)
    simplified = np.array(simplified.coords)
    print(f"✅ 步骤2：DP简化完成，点数：{len(simplified)}")
    return simplified


def close_and_build_polygon(points):
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = make_valid(polygon)
        if polygon.geom_type == 'MultiPolygon':
            polygon = max(polygon.geoms, key=lambda p: p.area)
    print(f"✅ 步骤3：闭合多边形生成完成，点数：{len(points)}")
    return polygon, points


# ===================== 【3】Direction Estimation =====================
def get_obb_main_direction(polygon):
    obb = polygon.minimum_rotated_rectangle
    obb_points = np.array(obb.exterior.coords)[:4]
    edges = [obb_points[i] - obb_points[i + 1] for i in range(3)]
    edge_lengths = [np.linalg.norm(e) for e in edges]
    max_edge_idx = np.argmax(edge_lengths)
    dx, dy = edges[max_edge_idx]
    theta0 = np.arctan2(dy, dx)
    return theta0


def local_direction_search(theta0, angles=[-15, -10, -5, 0, 5, 10, 15]):
    return [np.radians(ang) + theta0 for ang in angles]


# ===================== 【4】Core: Arrange Flight Routes =====================
def generate_parallel_scan_lines(polygon, theta, scan_width):
    W = scan_width
    H = W / 2
    bounds = polygon.bounds
    x_min, y_min, x_max, y_max = bounds
    line_list = []
    length = max(x_max - x_min, y_max - y_min) * 2
    center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    perp_theta = theta + np.pi / 2
    perp_dir = np.array([np.cos(perp_theta), np.sin(perp_theta)])
    all_points = np.array(polygon.exterior.coords)
    proj_all = np.dot(all_points, perp_dir)
    min_p = np.min(proj_all)
    max_p = np.max(proj_all)

    # 1. 首条航线位置
    first_pos = min_p + H
    pos_queue = []
    current_pos = first_pos

    # 2. 按固定间隔排布
    while True:
        if current_pos + H > max_p:
            break
        pos_queue.append(current_pos)
        current_pos += W

    # 3. 判断尾部空隙，决定是否补线
    if pos_queue:
        last_pos = pos_queue[-1]
        gap = max_p - (last_pos + H)
        print(f"DEBUG: 最后一条线右边界: {last_pos + H:.1f}, 区域右端: {max_p:.1f}, 空隙: {gap:.1f}")
        if gap > 0 and gap > H:
            add_pos = last_pos + H
            pos_queue.append(add_pos)
            print(f"⚠️  尾部空隙 {gap:.1f} > {H:.1f}，已在 {add_pos:.1f} 补线")

    # 4. 生成所有航线线段
    for pos in pos_queue:
        line_center = center + perp_dir * (pos - np.dot(center, perp_dir))
        s = line_center - np.array([np.cos(theta) * length, np.sin(theta) * length])
        e = line_center + np.array([np.cos(theta) * length, np.sin(theta) * length])
        line = LineString([s, e])
        try:
            inter = polygon.intersection(line)
            if not inter.is_empty:
                if inter.geom_type == "MultiLineString":
                    for seg in inter.geoms:
                        line_list.append((pos, np.array(seg.coords)))
                else:
                    line_list.append((pos, np.array(inter.coords)))
        except Exception as ex:
            print(f"⚠️  位置 {pos:.1f} 生成失败: {ex}")
            continue

    # 5. 按投影分组
    line_list.sort(key=lambda x: x[0])
    grouped = []
    now_p = None
    now_group = []
    for p, seg in line_list:
        if now_p is None or abs(p - now_p) > 1e-4:
            if now_group:
                grouped.append((now_p, now_group))
            now_p = p
            now_group = [seg]
        else:
            now_group.append(seg)
    if now_group:
        grouped.append((now_p, now_group))

    # 6. 同行内排序
    scan_dir = np.array([np.cos(theta), np.sin(theta)])
    final_group = []
    for p, segs in grouped:
        tmp = []
        for seg in segs:
            sp = np.dot(seg[0], scan_dir)
            tmp.append((sp, seg))
        tmp.sort()
        final_group.append((p, [s for _, s in tmp]))

    # 打印校验信息
    if final_group:
        first_p = final_group[0][0]
        end_p = final_group[-1][0]
        left_space = first_p - min_p
        right_space = max_p - (end_p + H)
        print(f"生成扫描线：共{len(final_group)}行")
        print(f"单侧最大允许距离：{H:.1f}")
        print(f"左侧预留距离：{left_space:.1f}")
        print(f"右侧末端空隙：{right_space:.1f}")
        if right_space > H + 1e-3:
            print(f"❌ 警告：右侧末端空隙 {right_space:.1f} 超出允许范围 {H:.1f}！")
    return [g for _, g in final_group]


# ===================== 【修正】平滑转弯曲线生成 =====================
def generate_smooth_turn_curve(boundary_points, p_start, p_end, num_samples=20):
    """
    基于高密度边界点，在起止点之间沿最短边缘拟合三次参数曲线
    :param boundary_points: 闭合高密度边界点 (N, 2) numpy数组
    :param p_start: 转弯起点 [x, y]
    :param p_end: 转弯终点 [x, y]
    :param num_samples: 曲线采样点数
    :return: 平滑曲线点 (num_samples, 2) numpy数组
    """
    # 1. 定位起止点在边界上的最近索引
    idx_s = np.argmin(np.linalg.norm(boundary_points - p_start, axis=1))
    idx_e = np.argmin(np.linalg.norm(boundary_points - p_end, axis=1))

    # 2. 生成顺时针、逆时针两条边界段
    if idx_s <= idx_e:
        seg_clockwise = boundary_points[idx_s:idx_e + 1]
        seg_counter = np.vstack([boundary_points[idx_s:], boundary_points[:idx_e + 1]])
    else:
        seg_clockwise = np.vstack([boundary_points[idx_s:], boundary_points[:idx_e + 1]])
        seg_counter = boundary_points[idx_e:idx_s + 1][::-1]

    # 3. 按实际弧长选择更短的一段（修正：不用点数判断，用真实长度）
    def calc_path_length(pts):
        return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))

    len_clock = calc_path_length(seg_clockwise)
    len_counter = calc_path_length(seg_counter)
    boundary_seg = seg_clockwise if len_clock <= len_counter else seg_counter

    # 边界点不足4个时 fallback 为直线
    if len(boundary_seg) < 4:
        return np.array([p_start, p_end])

    # 4. 以累计弧长为参数t，参数化边界点
    seg_diffs = np.diff(boundary_seg, axis=0)
    seg_lengths = np.linalg.norm(seg_diffs, axis=1)
    t = np.zeros(len(boundary_seg))
    t[1:] = np.cumsum(seg_lengths)
    total_t = t[-1]

    if total_t < 1e-6:
        return np.array([p_start, p_end])

    # 5. 分别对x、y坐标做三次多项式拟合
    poly_x = np.polyfit(t, boundary_seg[:, 0], 3)
    poly_y = np.polyfit(t, boundary_seg[:, 1], 3)

    # 6. 均匀采样生成平滑曲线
    t_sample = np.linspace(0, total_t, num_samples)
    x_smooth = np.polyval(poly_x, t_sample)
    y_smooth = np.polyval(poly_y, t_sample)

    return np.column_stack([x_smooth, y_smooth])


# ===================== 【修正】带平滑转弯的蛇形路径生成 =====================
def generate_all_possible_paths(grouped_scan_lines, boundary_points):
    if not grouped_scan_lines:
        return []
    paths = []

    # 路径1：Top-Left
    path1 = []
    reverse = False
    for i, group in enumerate(grouped_scan_lines):
        if reverse:
            ordered_lines = [line[::-1] for line in group[::-1]]
        else:
            ordered_lines = group

        # 非首行：插入贴合边界的平滑转弯曲线
        if i > 0:
            prev_end = path1[-1]
            curr_start = ordered_lines[0][0]
            turn_curve = generate_smooth_turn_curve(boundary_points, prev_end, curr_start)
            path1.extend(turn_curve[1:])  # 去重起点

        for line in ordered_lines:
            path1.extend(line)
        reverse = not reverse
    paths.append(('Top-Left', np.array(path1)))

    # 路径2：Top-Right
    path2 = []
    reverse = True
    for i, group in enumerate(grouped_scan_lines):
        if reverse:
            ordered_lines = [line[::-1] for line in group[::-1]]
        else:
            ordered_lines = group

        if i > 0:
            prev_end = path2[-1]
            curr_start = ordered_lines[0][0]
            turn_curve = generate_smooth_turn_curve(boundary_points, prev_end, curr_start)
            path2.extend(turn_curve[1:])

        for line in ordered_lines:
            path2.extend(line)
        reverse = not reverse
    paths.append(('Top-Right', np.array(path2)))

    # 路径3：Bottom-Left
    path3 = []
    reverse = True
    for i, group in enumerate(reversed(grouped_scan_lines)):
        if reverse:
            ordered_lines = [line[::-1] for line in group[::-1]]
        else:
            ordered_lines = group

        if i > 0:
            prev_end = path3[-1]
            curr_start = ordered_lines[0][0]
            turn_curve = generate_smooth_turn_curve(boundary_points, prev_end, curr_start)
            path3.extend(turn_curve[1:])

        for line in ordered_lines:
            path3.extend(line)
        reverse = not reverse
    paths.append(('Bottom-Left', np.array(path3)))

    # 路径4：Bottom-Right
    path4 = []
    reverse = False
    for i, group in enumerate(reversed(grouped_scan_lines)):
        if reverse:
            ordered_lines = [line[::-1] for line in group[::-1]]
        else:
            ordered_lines = group

        if i > 0:
            prev_end = path4[-1]
            curr_start = ordered_lines[0][0]
            turn_curve = generate_smooth_turn_curve(boundary_points, prev_end, curr_start)
            path4.extend(turn_curve[1:])

        for line in ordered_lines:
            path4.extend(line)
        reverse = not reverse
    paths.append(('Bottom-Right', np.array(path4)))

    return paths


# ===================== 【6】Complete Cost Calculation =====================
def calculate_complete_cost(start_point, snake_path, grouped_scan_lines):
    if len(snake_path) == 0:
        return float('inf'), 0, 0, 0
    takeoff_dist = np.hypot(start_point[0] - snake_path[0][0], start_point[1] - snake_path[0][1])
    scan_dist = 0
    for group in grouped_scan_lines:
        for line in group:
            scan_dist += np.hypot(line[-1][0] - line[0][0], line[-1][1] - line[0][1])
    total_path_length = 0
    for i in range(1, len(snake_path)):
        total_path_length += np.hypot(snake_path[i][0] - snake_path[i - 1][0], snake_path[i][1] - snake_path[i - 1][1])
    turn_dist = total_path_length - scan_dist
    total_cost = takeoff_dist + scan_dist + turn_dist
    return total_cost, takeoff_dist, scan_dist, turn_dist


# ===================== 【7】Main Function（修正：传入高密度边界） =====================
def drone_coverage_path(sketch_points, start_point, scan_width):
    step0_original = sketch_points
    step1_smoothed = moving_average_smooth(step0_original, k=3)
    step2_simplified = douglas_peucker_simplify(step1_smoothed, tolerance=1.5)
    step3_polygon, step3_closed = close_and_build_polygon(step2_simplified)

    # 用于拟合转弯曲线的高密度边界：用平滑后的边界（点更密，贴合度更好）
    smooth_boundary = step1_smoothed
    if not np.allclose(smooth_boundary[0], smooth_boundary[-1]):
        smooth_boundary = np.vstack([smooth_boundary, smooth_boundary[0]])

    theta0 = get_obb_main_direction(step3_polygon)
    candidate_thetas = local_direction_search(theta0)

    min_total_cost = float('inf')
    best_path = None
    best_grouped_lines = None
    best_theta = None
    best_start_type = None
    best_cost_details = None

    for theta in candidate_thetas:
        grouped_lines = generate_parallel_scan_lines(step3_polygon, theta, scan_width)
        if not grouped_lines:
            continue
        # 传入高密度平滑边界做曲线拟合
        all_paths = generate_all_possible_paths(grouped_lines, smooth_boundary)
        for start_type, path in all_paths:
            total_cost, takeoff_dist, scan_dist, turn_dist = calculate_complete_cost(start_point, path, grouped_lines)
            print(
                f"方向{np.degrees(theta):.1f}°，起始点{start_type}：总代价={total_cost:.2f}（起飞={takeoff_dist:.2f}，扫描={scan_dist:.2f}，转弯={turn_dist:.2f}）")
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_path = path
                best_grouped_lines = grouped_lines
                best_theta = theta
                best_start_type = start_type
                best_cost_details = (takeoff_dist, scan_dist, turn_dist)

    if best_path is None:
        raise ValueError("未找到有效路径，请调小scan_width！")
    print(f"\n✅ 找到最优路径：方向{np.degrees(best_theta):.1f}°，起始点{best_start_type}")
    return (step0_original, step1_smoothed, step2_simplified, step3_closed, step3_polygon, best_path, min_total_cost,
            best_theta, best_start_type, best_cost_details)


# ===================== 【8】Visualization =====================
def visualize_all_steps(steps, start_point, save_path="debug_contour_steps.png"):
    step0, step1, step2, step3, polygon, best_path, _, _, _, _ = steps
    plt.figure(figsize=(12, 10))
    plt.plot(step0[:, 0], step0[:, 1], 'r--', label='Step 0: Original Boundary', linewidth=1)
    plt.plot(step1[:, 0], step1[:, 1], 'orange', linestyle=':', label='Step 1: Smoothed', linewidth=1)
    plt.plot(step2[:, 0], step2[:, 1], 'b-', label='Step 2: DP Simplified', linewidth=1.5)
    plt.plot(step3[:, 0], step3[:, 1], 'purple', linewidth=2, label='Step 3: Final Polygon')
    plt.scatter(start_point[0], start_point[1], c='green', s=100, label='Start Point', zorder=5)
    if len(best_path) > 0:
        plt.plot([start_point[0], best_path[0][0]], [start_point[1], best_path[0][1]], 'y--', linewidth=2,
                 label='Flight to Scan Start', zorder=4)
        plt.plot(best_path[:, 0], best_path[:, 1], 'darkgreen', label='Optimal Snake Path', linewidth=1)
        plt.scatter(best_path[0][0], best_path[0][1], c='red', s=50, label='Scan Start Point', zorder=6)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.axis('equal')
    plt.title('Contour Processing Steps Comparison')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 轮廓处理步骤对比图已保存到：{save_path}")
    plt.close()


def visualize_final_result(polygon_points, start_point, best_path, scan_width, save_path="drone_path_result.png"):
    plt.figure(figsize=(10, 8))
    plt.plot(polygon_points[:, 0], polygon_points[:, 1], 'b-', linewidth=2, label='Coverage Region')
    plt.scatter(start_point[0], start_point[1], c='green', s=100, label='Start Point', zorder=5)
    if len(best_path) > 0:
        plt.plot([start_point[0], best_path[0][0]], [start_point[1], best_path[0][1]], 'y--', linewidth=2,
                 label='Flight to Scan Start', zorder=4)
        plt.plot(best_path[:, 0], best_path[:, 1], 'darkgreen', label=f'Optimal Path (Scan Width={scan_width})',
                 linewidth=1)
        plt.scatter(best_path[0][0], best_path[0][1], c='red', s=50, label='Scan Start Point', zorder=6)
    plt.legend()
    plt.axis('equal')
    plt.title('UAV Optimal Coverage Path (Full Boundary Coverage)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 最终最优路径图已保存到：{save_path}")
    plt.close()


# ===================== Execution Entry =====================
if __name__ == '__main__':
    original_points = image_to_sketch_points("hand_drawn.png", down_scale=1, line_width=3)
    start_point = (900, 100)
    scan_width = 40
    all_steps = drone_coverage_path(original_points, start_point, scan_width)
    visualize_all_steps(all_steps, start_point)
    _, _, _, closed_sketch, _, best_path, min_total_cost, best_theta, best_start_type, best_cost_details = all_steps
    visualize_final_result(closed_sketch, start_point, best_path, scan_width)
    takeoff_dist, scan_dist, turn_dist = best_cost_details
    print(f"\n===== 最终详细结果 =====")
    print(f"单条路径总覆盖宽度：{scan_width}（单侧覆盖半径{scan_width / 2}）")
    print(f"最优扫描方向：{np.degrees(best_theta):.1f}°")
    print(f"最优扫描起始点：{best_start_type}")
    print(f"总飞行距离：{min_total_cost:.2f}")
    print(f"  - 起飞距离：{takeoff_dist:.2f}")
    print(f"  - 扫描距离：{scan_dist:.2f}")
    print(f"  - 转弯距离：{turn_dist:.2f}")
    print(f"路径总点数：{len(best_path)}")