import numpy as np
import time

def projection_3d_to_2d(point_3d, camera_position, screen_width, screen_height, fov, near_plane):
    relative_position = point_3d - camera_position
    X, Y, Z = relative_position

    # Zが近接平面より小さい場合は描画しない
    # if Z <= near_plane:
    #     return None, None
    
    aspect_ratio = screen_width / screen_height
    fov_rad = np.radians(fov) / 2
    scale_x = 1 / np.tan(fov_rad)
    scale_y = scale_x / aspect_ratio

    x_2d = (X / Z) * scale_x
    y_2d = (Y / Z) * scale_y

    x_screen = int((x_2d + 1) * screen_width / 2)
    y_screen = int((1 - y_2d) * screen_height / 2)

    return (x_screen, y_screen), Z  # Zを返す

def bresenham_with_distances(x0, y0, z0, x1, y1, z1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = z1 - z0  # z方向の差を計算
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0, z0))  # z座標をそのまま追加
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

        # z座標の補完を行う
        z0 += dz * (1 / max(dx, dy))  # 線形補完を適用

    return points

edges = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

fov = 90
near_plane = 1
far_plane = 100
camera_position = np.array([1.5, 1.5, 5], dtype=float)

screen_width = 400
screen_height = 330

points_3d = [
    np.array([1, 1, 1]),
    np.array([2, 1, 1]),
    np.array([1, 2, 1]),
    np.array([2, 2, 1]),
    np.array([1, 1, 2]),
    np.array([2, 1, 2]),
    np.array([1, 2, 2]),
    np.array([2, 2, 2]),
]

def get_char_by_distance(distance):
    if distance < 3:
        return '@'
    elif distance < 4:
        return '#'
    elif distance < 5:
        return '+'
    else:
        return '.'

while True:
    display = np.full((screen_height, screen_width), ' ')

    projected_points = []
    for point_3d in points_3d:
        result, distance = projection_3d_to_2d(point_3d, camera_position, screen_width, screen_height, fov, near_plane)
        if result is not None:
            x, y = result
            if 0 <= x < screen_width and 0 <= y < screen_height:
                char = get_char_by_distance(distance)
                display[y, x] = char
                projected_points.append((x, y, distance))

    for edge in edges:
        if edge[0] < len(projected_points) and edge[1] < len(projected_points):
            p1, p2 = projected_points[edge[0]], projected_points[edge[1]]
            line_points = bresenham_with_distances(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])
            for x, y, z in line_points:
                if 0 <= x < screen_width and 0 <= y < screen_height:
                    char = get_char_by_distance(z)
                    display[y, x] = char

    rows = []
    for row in display:
        rows.append("".join(map(str, row)))

    for row in rows:
        print(row)
    print(f"\033[{len(rows)+1}A")

    time.sleep(0.1)
    camera_position[2] -= 0.1
    if camera_position[2] < 2:
        camera_position[2] = 5
