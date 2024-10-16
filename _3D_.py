import numpy as np

# カメラパラメータ
camera_position = np.array([0, 0, 10])
fov = np.radians(90)  # 視野角をラジアンに変換
aspect_ratio = 16 / 9 / 2  # アスペクト比
near_plane = 0.1
far_plane = 1000.0

# スクリーンのサイズ
screen_width = 1700  # ターミナルの横幅
screen_height = 500  # ターミナルの縦幅

# 立方体の中心位置
box_center = np.array([1, -1, 0])
half_size = 0.5  # 一辺の長さが1なので、各頂点は±0.5のオフセット

# 立方体の各頂点
vertices = np.array([
    [box_center[0] - half_size, box_center[1] - half_size, box_center[2] - half_size],
    [box_center[0] + half_size, box_center[1] - half_size, box_center[2] - half_size],
    [box_center[0] + half_size, box_center[1] + half_size, box_center[2] - half_size],
    [box_center[0] - half_size, box_center[1] + half_size, box_center[2] - half_size],
    [box_center[0] - half_size, box_center[1] - half_size, box_center[2] + half_size],
    [box_center[0] + half_size, box_center[1] - half_size, box_center[2] + half_size],
    [box_center[0] + half_size, box_center[1] + half_size, box_center[2] + half_size],
    [box_center[0] - half_size, box_center[1] + half_size, box_center[2] + half_size]
])

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

# 立方体の頂点をスクリーン座標に投影
def project_to_screen(vertices, camera_position, projection_matrix):
    translated_positions = vertices - camera_position
    homogeneous_coordinates = np.hstack([translated_positions, np.ones((translated_positions.shape[0], 1))])
    projected_positions = homogeneous_coordinates.dot(projection_matrix.T)
    ndc_positions = projected_positions[:, :2] / projected_positions[:, 3, np.newaxis]
    screen_x = (ndc_positions[:, 0] + 1) * screen_width / 2
    screen_y = (1 - ndc_positions[:, 1]) * screen_height / 2
    return screen_x, screen_y

# 透視投影行列を取得
projection_matrix = perspective_projection(fov, aspect_ratio, near_plane, far_plane)

# 立方体の頂点をスクリーンに投影
screen_x, screen_y = project_to_screen(vertices, camera_position, projection_matrix)

# 面を定義（各面の頂点をインデックスで指定）
faces = [
    [0, 1, 2, 3],  # 前面
    [4, 5, 6, 7],  # 背面
    [0, 1, 5, 4],  # 左面
    [1, 2, 6, 5],  # 下面
    [2, 3, 7, 6],  # 右面
    [3, 0, 4, 7],  # 上面
]

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

# 三角形の面を塗りつぶすための関数（スキャンライン法を使用）
def fill_triangle(x1, y1, x2, y2, x3, y3, terminal_screen, screen_width, screen_height):
    # 頂点をyの昇順にソート
    if y1 > y2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    if y2 > y3:
        x2, x3 = x3, x2
        y2, y3 = y3, y2
    if y1 > y2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    
    # y1, y2, y3 の順に対応するx座標も並べ替え済み
    # スキャンライン処理
    for y in range(int(y1), int(y3) + 1):  # y1, y3を整数にキャスト
        # 各辺の交点計算（y = 定数）のx座標
        if (y2 - y1) != 0:  # ゼロ除算を避ける
            x_left = int(((x2 - x1) * (y - y1) / (y2 - y1)) + x1)
        else:
            x_left = int(x1)
        
        if (y3 - y2) != 0:  # ゼロ除算を避ける
            x_right = int(((x3 - x2) * (y - y2) / (y3 - y2)) + x2)
        else:
            x_right = int(x2)

        # 横方向に描画
        for x in range(x_left, x_right + 1):
            if 0 <= x < screen_width and 0 <= y < screen_height:
                terminal_screen[y][x] = '*'

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
    return dot_product > 0

# ターミナルに表示するためにプロット（裏面カリングを追加）
def render_to_terminal_with_culling(screen_x, screen_y, faces, screen_width, screen_height, vertices, camera_position):
    terminal_screen = [[' ' for _ in range(screen_width)] for _ in range(screen_height)]
    
    for face in faces:
        if is_visible(face, vertices, camera_position):  # カメラから見える面だけ描画
            # 三角形の各辺を描画
            for i in range(len(face)):
                start = face[i]
                end = face[(i + 1) % len(face)]
                draw_line(int(screen_x[start]), int(screen_y[start]), int(screen_x[end]), int(screen_y[end]), terminal_screen, screen_width, screen_height)
            
            # 塗りつぶし
            fill_triangle(screen_x[face[0]], screen_y[face[0]],
                          screen_x[face[1]], screen_y[face[1]],
                          screen_x[face[2]], screen_y[face[2]], terminal_screen, screen_width, screen_height)

    return terminal_screen

# 立方体を表示
terminal_screen = render_to_terminal_with_culling(screen_x, screen_y, faces, screen_width, screen_height, vertices, camera_position)

# ターミナルに表示
for row in terminal_screen:
    print("".join(row))
