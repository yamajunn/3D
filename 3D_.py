import numpy as np

# カメラパラメータ
camera_position = np.array([0, 0, 5])
camera_direction = np.array([0, 0, -1])
fov = np.radians(90)  # 視野角をラジアンに変換
aspect_ratio = 16 / 9 / 2  # アスペクト比
near_plane = 0.1
far_plane = 1000.0

# 画面サイズ
screen_width = 1700  # ターミナルの横幅
screen_height = 500  # ターミナルの縦幅

def calculate_cube_vertices(center, size):
    # 立方体の中心座標と一辺の長さを指定
    cx, cy, cz = center
    half_size = size / 2

    # 立方体の頂点座標を計算
    vertices = [
        [cx - half_size, cy - half_size, cz - half_size],
        [cx - half_size, cy - half_size, cz + half_size],
        [cx - half_size, cy + half_size, cz - half_size],
        [cx - half_size, cy + half_size, cz + half_size],
        [cx + half_size, cy - half_size, cz - half_size],
        [cx + half_size, cy - half_size, cz + half_size],
        [cx + half_size, cy + half_size, cz - half_size],
        [cx + half_size, cy + half_size, cz + half_size]
    ]
    
    return np.array(vertices)

# 立方体の中心座標と一辺の長さを設定
center = (2, 3, -5)  # 原点を中心とする
size = 2  # 一辺の長さ

# 立方体の頂点を計算
vertices = calculate_cube_vertices(center, size)

# 立方体の面（頂点のインデックス）
cube_faces = [
    (0, 1, 3, 2),  # 左
    (4, 5, 7, 6),  # 右
    (0, 1, 5, 4),  # 下
    (2, 3, 7, 6),  # 上
    (0, 2, 6, 4),  # 後ろ
    (1, 3, 7, 5)   # 前
]

# 透視投影行列を生成する関数
def perspective_projection_matrix(fov, aspect, near, far):
    f = 1 / np.tan(fov / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])

# 物体を2Dに投影する関数
def project_vertex(vertex, projection_matrix):
    vertex_h = np.append(vertex, 1)
    projected_h = projection_matrix @ vertex_h
    if projected_h[3] == 0:  # 透視投影行列のwが0の場合を防ぐ
        return np.array([0, 0])
    projected = projected_h[:3] / projected_h[3]
    return projected[:2]

# ビューポート変換
def viewport_transform(v, screen_width, screen_height):
    return np.array([
        int((v[0] + 1) * 0.5 * screen_width),
        int((1 - (v[1] + 1) * 0.5) * screen_height)
    ])

# カメラから面までの距離を計算
def face_distance(face, vertices, camera_position):
    face_center = np.mean([vertices[i] for i in face], axis=0)
    return np.linalg.norm(face_center - camera_position)

# 面がカメラに向いているかを判断するための法線計算
def is_facing_camera(face, vertices, camera_position):
    v0, v1, v2 = [vertices[i] for i in face[:3]]
    normal = np.cross(v1 - v0, v2 - v0)
    camera_direction = camera_position - np.mean([v0, v1, v2], axis=0)
    return np.dot(normal, camera_direction) < 0

# 投影行列を生成
projection_matrix = perspective_projection_matrix(fov, aspect_ratio, near_plane, far_plane)

# 画面バッファを初期化
screen_buffer = [[' ' for _ in range(screen_width)] for _ in range(screen_height)]

# 面内のピクセルを塗りつぶすためのバケツ塗りつぶしアルゴリズム
def fill_polygon(vertices, screen_buffer, char, shrink_factor=0.9):
    # 頂点をx,y座標のリストに変換
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    
    # 最小のx, y座標と最大のx, y座標を計算
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    # 内部のピクセルを塗りつぶす
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            # 点が多角形内にあるかを判定
            if is_point_in_polygon(x, y, vertices):
                if 0 <= x < screen_width and 0 <= y < screen_height:
                    # 塗りつぶしを少し小さくして枠線を作る
                    if (x - min_x) % 2 == 0 and (y - min_y) % 2 == 0:  
                        screen_buffer[y][x] = char

# 多角形の中に点が含まれているかを判定
def is_point_in_polygon(x, y, vertices):
    inside = False
    n = len(vertices)
    p1x, p1y = vertices[0]
    for i in range(n + 1):
        p2x, p2y = vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# 各面に対して背面カリングと描画処理
for face in cube_faces:
    if is_facing_camera(face, vertices, camera_position):
        # 面の中心とカメラの距離を計算
        distance = face_distance(face, vertices, camera_position)
        
        # 距離に応じて表示するテキストを変更
        if distance < 2:
            display_char = '#'
        elif distance < 4:
            display_char = '*'
        else:
            display_char = '.'

        # 面の頂点を2Dに投影
        projected_vertices = [project_vertex(vertices[i], projection_matrix) for i in face]
        
        # 投影された頂点を画面に描画
        projected_vertices = [viewport_transform(v, screen_width, screen_height) for v in projected_vertices]
        
        # 面を塗りつぶす
        fill_polygon(projected_vertices, screen_buffer, display_char)

# 画面に描画結果を表示
for row in screen_buffer:
    print(''.join(row))
