import numpy as np

# カメラパラメータ
camera_position = np.array([0, 0, 10])
fov = np.radians(150)  # 視野角をラジアンに変換
aspect_ratio = 16 / 9  # アスペクト比
near_plane = 0.1
far_plane = 1000.0

# スクリーンのサイズ
screen_width = 1700  # ターミナルの横幅
screen_height = 500  # ターミナルの縦幅

# 立方体の中心位置とサイズ（複数の立方体）
boxes = [
    {'center': np.array([0, -1, 0]), 'half_size': 0.5},
    {'center': np.array([0, 0, 0]), 'half_size': 0.5},
    {'center': np.array([0, 1, 0]), 'half_size': 0.5},
    {'center': np.array([0, 2, 0]), 'half_size': 0.5},
    {'center': np.array([0, 3, 0]), 'half_size': 0.5},
    {'center': np.array([1, 2, 0]), 'half_size': 0.5},
    {'center': np.array([0, 2, 1]), 'half_size': 0.5},
    {'center': np.array([-1, 2, 0]), 'half_size': 0.5},
    {'center': np.array([0, 2, -1]), 'half_size': 0.5},
]
# for x in range(-5, 6):
#     for y in range(1):
#         for z in range(-5, 6):
#             boxes.append({'center': np.array([x, y-2, z]), 'half_size': 0.5})

# 透視投影行列の計算
def perspective_projection(fov, aspect_ratio, near_plane, far_plane):
    f = 1 / np.tan(fov / 2)
    matrix = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far_plane + near_plane) / (near_plane - far_plane), -1],
        [0, 0, (2 * far_plane * near_plane) / (near_plane - far_plane), 0]
    ])
    return matrix

# 立方体の頂点を生成する関数
def get_cube_vertices(center, half_size):
    return np.array([
        [center[0] - half_size, center[1] - half_size, center[2] - half_size],
        [center[0] + half_size, center[1] - half_size, center[2] - half_size],
        [center[0] + half_size, center[1] + half_size, center[2] - half_size],
        [center[0] - half_size, center[1] + half_size, center[2] - half_size],
        [center[0] - half_size, center[1] - half_size, center[2] + half_size],
        [center[0] + half_size, center[1] - half_size, center[2] + half_size],
        [center[0] + half_size, center[1] + half_size, center[2] + half_size],
        [center[0] - half_size, center[1] + half_size, center[2] + half_size]
    ])

# 立方体の面を定義（各面の頂点をインデックスで指定）
def get_cube_faces():
    return [
        [0, 1, 2, 3],  # 前面
        [4, 5, 6, 7],  # 背面
        [0, 1, 5, 4],  # 左面
        [1, 2, 6, 5],  # 下面
        [2, 3, 7, 6],  # 右面
        [3, 0, 4, 7],  # 上面
    ]

# 立方体の頂点をスクリーン座標に投影
def project_to_screen(vertices, camera_position, projection_matrix):
    translated_positions = vertices - camera_position
    homogeneous_coordinates = np.hstack([translated_positions, np.ones((translated_positions.shape[0], 1))])
    projected_positions = homogeneous_coordinates.dot(projection_matrix.T)
    ndc_positions = projected_positions[:, :2] / projected_positions[:, 3, np.newaxis]
    screen_x = (ndc_positions[:, 0] + 1) * screen_width / 2
    screen_y = (1 - ndc_positions[:, 1]) * screen_height / 2
    return screen_x, screen_y

# 直線を描画する関数（ブレゼンハムのアルゴリズム）
def draw_line(x1, y1, x2, y2, screen, width, height):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        if 0 <= x1 < width and 0 <= y1 < height:
            screen[y1][x1] = '#'
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

# 面の法線ベクトルを計算し、カメラから見えるかどうかを判定
def is_visible(face, vertices, camera_position):
    # 面の三角形の頂点3つを使って法線ベクトルを計算
    v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    normal = np.cross(v1 - v0, v2 - v0)
    
    # カメラ方向のベクトル
    camera_direction = camera_position - v0
    
    # 法線ベクトルとカメラ方向のベクトルの内積を計算
    dot_product = np.dot(normal, camera_direction)
    
    # 内積が正ならカメラに向いている（表示する）、負なら背を向けている（表示しない）
    return dot_product

# ターミナルに表示するためにプロット（裏面カリングを追加）
def render_to_terminal_with_culling(boxes, screen_width, screen_height, camera_position, projection_matrix):
    terminal_screen = [[' ' for _ in range(screen_width)] for _ in range(screen_height)]
    
    # 各立方体を描画
    for box in boxes:
        vertices = get_cube_vertices(box['center'], box['half_size'])
        screen_x, screen_y = project_to_screen(vertices, camera_position, projection_matrix)
        faces = get_cube_faces()

        visible_list = []
        for face in faces:
            visible_list.append(is_visible(face, vertices, camera_position))

        for i, face in enumerate(faces):
            if visible_list[i] > 0 and visible_list[i] != max(visible_list):  # カメラから見える面だけ描画
                # 三角形の各辺を描画
                for i in range(len(face)):
                    start = face[i]
                    end = face[(i + 1) % len(face)]
                    draw_line(int(screen_x[start]), int(screen_y[start]), int(screen_x[end]), int(screen_y[end]), terminal_screen, screen_width, screen_height)
    
    return terminal_screen

# 透視投影行列を取得
projection_matrix = perspective_projection(fov, aspect_ratio, near_plane, far_plane)

# ターミナルに立体を表示
terminal_screen = render_to_terminal_with_culling(boxes, screen_width, screen_height, camera_position, projection_matrix)

# ターミナルに表示
for row in terminal_screen:
    print("".join(row))
