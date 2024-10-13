import numpy as np
import time

# 頂点を生成する関数
def generate_cube_vertices(point_3d, size=1):
    x, y, z = point_3d
    vertices = [
        np.array([x, y, z]),
        np.array([x + size, y, z]),
        np.array([x, y + size, z]),
        np.array([x + size, y + size, z]),
        np.array([x, y, z + size]),
        np.array([x + size, y, z + size]),
        np.array([x, y + size, z + size]),
        np.array([x + size, y + size, z + size]),
    ]
    return vertices

def projection_3d_to_2d(point_3d, camera_position, screen_width, screen_height, fov, near_plane):
    relative_position = point_3d - camera_position
    X, Y, Z = relative_position
    # near_plane より手前にある場合は処理しない
    if Z <= near_plane:
        return None, None  # 描画しない

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
    dz = z1 - z0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0, z0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

        z0 += dz * (1 / max(dx, dy))

    return points

# 画面外の座標を画面内に収めるクリッピング関数
def clip_to_screen(x, y, screen_width, screen_height):
    # 画面外に出た点を画面の境界にクリッピングする
    x = max(0, min(x, screen_width - 1))
    y = max(0, min(y, screen_height - 1))
    return x, y

# 立方体の辺
edges = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

fov = 90
near_plane = 1
camera_position = np.array([1.5, 1.5, 5], dtype=float)

screen_width = 700
screen_height = 500

starting_points = []
for x in range(-7, 7):
    for y in range(1):
        for z in range(-20, 20):
            starting_points.append(np.array([x, y-0.5, z], dtype=float))

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

    # 各立方体ごとに処理
    for starting_point in starting_points:
        # 頂点座標を生成
        points_3d = generate_cube_vertices(starting_point)
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
                
                # 端点をクリッピング
                x0, y0 = clip_to_screen(p1[0], p1[1], screen_width, screen_height)
                x1, y1 = clip_to_screen(p2[0], p2[1], screen_width, screen_height)
                
                # クリッピング後の座標で線を描画
                line_points = bresenham_with_distances(x0, y0, p1[2], x1, y1, p2[2])
                
                for x, y, z in line_points:
                    if 0 <= x < screen_width and 0 <= y < screen_height:
                        char = get_char_by_distance(z)
                        display[y, x] = char

    for i in range(len(display)):
        for j in range(len(display[i])):
            if i == 0 or i == screen_height-1 or j == 0 or j == screen_width-1:
                display[i][j] = '#'
    
    rows = []
    for row in display:
        rows.append("".join(map(str, row)))

    for row in rows:
        print(row)
    print(f"\033[{len(rows)+1}A")

    time.sleep(0.1)
    camera_position[2] += 0.1
    if camera_position[2] > 10:
        camera_position[2] = 5
