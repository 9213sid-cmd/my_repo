import math
from compas.geometry import Point, Vector
from compas.datastructures import Mesh
from compas.geometry import Translation, transform_points, Rotation

# ==============================================================================
# 幾何輔助函式 (回傳原始座標列表)
# ==============================================================================

def get_rotated_quad(center: Point, width_x: float, width_y: float, angle: float) -> list[list[float]]:
    """
    計算一個旋轉後的矩形底面（四個頂點），回傳原始座標列表 [[x, y, z], ...]。
    """
    half_x, half_y = width_x * 0.5, width_y * 0.5
    
    # 1. 建立在原點的基礎頂點
    base_pts = [
        Point( half_x,  half_y, 0),
        Point(-half_x,  half_y, 0),
        Point(-half_x, -half_y, 0),
        Point( half_x, -half_y, 0),
    ]
    
    # 2. 執行旋轉 (繞 Z 軸)
    R = Rotation.from_axis_and_angle(Vector(0, 0, 1), angle)
    # transform_points 回傳原始座標列表
    rotated_coords = transform_points(base_pts, R) 
    
    # 3. 執行平移 (移動到目標中心點)
    T = Translation.from_vector(Vector(center.x, center.y, center.z))
    # transform_points 輸入和輸出都是原始座標列表
    final_coords = transform_points(rotated_coords, T) 

    # 4. 直接回傳原始座標列表
    return final_coords


# building_generator.py 中 create_twisted_prism_mesh 函式

def create_twisted_prism_mesh(base_quad_coords: list, top_quad_coords: list) -> Mesh:
    """
    從底部和頂部四邊形的原始座標列表，使用手動方法建立一個 Mesh 物件。
    
    這是最終的、最穩定的建構方式，可避免 Viewer 兼容性錯誤。
    """
    
    # 頂點列表是兩個原始座標列表的合併
    vertices = base_quad_coords + top_quad_coords 
    
    # 1. 創建一個空的 Mesh 實例
    mesh = Mesh()
    
    # 2. 手動添加頂點
    # 這裡確保每個頂點都是以最純淨的 (x, y, z) 形式被加入
    vertex_keys = []
    for x, y, z in vertices:
        # 使用 add_vertex()，這是最基礎且穩定的方法
        key = mesh.add_vertex(x=x, y=y, z=z)
        vertex_keys.append(key)

    # 3. 定義面列表（使用頂點的索引/鍵值）
    # 由於我們按照順序添加了 0 到 7 的頂點，索引與原本的 faces 保持一致
    faces = [
        [0, 1, 2, 3],    # 底部
        [7, 6, 5, 4],    # 頂部
        [0, 3, 7, 4],    # 側面 1
        [3, 2, 6, 7],    # 側面 2
        [2, 1, 5, 6],    # 側面 3
        [1, 0, 4, 5],    # 側面 4
    ]
    
    # 4. 手動添加面
    for face in faces:
        mesh.add_face(face)
    
    # 返回手動建構的 Mesh
    return mesh


# ==============================================================================
# 核心生成函式
# ==============================================================================

def create_single_twisted_prism(angle_deg: int, base_x: float, base_y: float, height: float) -> Mesh:
    """
    生成一個指定扭轉角度的單一矩形塊體，只回傳 Mesh 物件。
    """
    
    angle_rad = math.radians(angle_deg)
    
    # 底部中心點 (Z=0)
    base_center = Point(0, 0, 0) 
    # 頂部中心點 (Z=height)
    top_center = Point(0, 0, height) 
    
    # 底部頂點 (旋轉角度為 0) - 回傳座標列表
    base_quad_coords = get_rotated_quad(base_center, base_x, base_y, angle=0.0)
    
    # 頂部頂點 (旋轉角度為 angle_rad) - 回傳座標列表
    top_quad_coords = get_rotated_quad(top_center, base_x, base_y, angle=angle_rad)
    
    return create_twisted_prism_mesh(base_quad_coords, top_quad_coords)

# building_generator.py - 新增 DNA 核心生成函式

def create_dna_tower(total_angle_deg: float, base_x: float, base_y: float, height: float, num_floors: int = 10) -> Mesh:
    """
    生成一個具有分層扭轉效果的 DNA 建築，總扭轉角度由評分結果決定。
    
    :param total_angle_deg: 最底層到最頂層的總扭轉角度（度）。
    :param num_floors: 建築物分割的樓層數 (決定扭轉的平滑度)。
    :return: 最終的 Mesh 物件。
    """
    
    total_angle_rad = math.radians(total_angle_deg)
    
    # 計算每層的扭轉增量和高度增量
    angle_increment = total_angle_rad / num_floors
    height_increment = height / num_floors
    
    # 創建 Mesh 數據所需的頂點和面列表 (使用 compas 的數據結構)
    all_vertices = []
    all_faces = []
    
    # 頂點索引偏移，用於連接各層的面
    index_offset = 0
    
    for i in range(num_floors + 1):
        current_height = i * height_increment
        current_angle = i * angle_increment
        current_center = Point(0, 0, current_height)
        
        # 1. 獲取當前樓層的四個頂點座標 (使用 get_rotated_quad)
        quad_coords = get_rotated_quad(current_center, base_x, base_y, current_angle)
        
        # 2. 將座標添加到總頂點列表
        for x, y, z in quad_coords:
            all_vertices.append((x, y, z)) # 使用 tuple 格式是通用做法
        
        # 3. 如果不是最底層，則創建側面
        if i > 0:
            # 獲取這一層和上一層的索引
            current_keys = [index_offset, index_offset + 1, index_offset + 2, index_offset + 3]
            prev_keys = [index_offset - 4, index_offset - 3, index_offset - 2, index_offset - 1]
            
            # 連接四個側面
            for j in range(4):
                k0 = prev_keys[j]             # 上層頂點 j
                k1 = prev_keys[(j + 1) % 4]   # 上層頂點 j+1
                k2 = current_keys[(j + 1) % 4] # 本層頂點 j+1
                k3 = current_keys[j]           # 本層頂點 j
                
                all_faces.append([k0, k1, k2, k3])
        
        index_offset += 4
    
    # 4. 創建底部和頂部
    # 底部
    bottom_keys = [0, 1, 2, 3]
    all_faces.append(bottom_keys)
    
    # 頂部
    top_keys = [index_offset - 4, index_offset - 3, index_offset - 2, index_offset - 1]
    all_faces.append(top_keys)
    
    # 5. 使用最穩定的方式創建 Mesh
    mesh = Mesh()
    vertex_key_map = {}
    for i, (x, y, z) in enumerate(all_vertices):
        key = mesh.add_vertex(x=x, y=y, z=z)
        vertex_key_map[i] = key # 存儲新舊索引對應關係
        
    for face_indices in all_faces:
        # 將 faces 索引轉換為 Mesh 內部的 key
        face_keys = [vertex_key_map[i] for i in face_indices]
        mesh.add_face(face_keys)
    
    return mesh