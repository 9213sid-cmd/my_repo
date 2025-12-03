# %%
from compas_notebook.viewer import Viewer
from compas.geometry import Point, Line, Polyline, Vector
from compas.colors import Color
from compas.geometry import Vector
from compas.datastructures import Mesh
import math
import numpy as np

viewer = Viewer()
viewer.scene.clear()

# %% [markdown]
# 參數設定

# %%
R = 3                        # 螺旋半徑
pitch = 40                   # 每一圈上升高度（節距）
turns = 0.25                 # 總圈數
points_per_turn = 200        # 每圈取樣點（越大越平滑）
rungs_per_turn = 100         # 每圈畫幾根「鹼基配對」橫桿
RUNG_WIDTH = 3               # 控制橫桿面的寬度
flor= pitch / rungs_per_turn # 樓層高度
WIND_DIR = Vector(1, 0, 0)   #風向
U = 1.0                      #風速
R_OBS = R*1.15               #障礙物等效半徑
ALPHA = 1.8                  #繞流強度

# 密度與範圍（seeds on YZ plan）
N_Y, N_Z = 2, 3              # 流線密度（越大越密）
Y_RANGE  = (-2.2*R, 2.2*R)
Z_RANGE  = (0.0, pitch*turns)# 在整段 DNA 高度內產生箭頭

# 線段控制
STREAM_LEN   = 6.0 * R         # 每條流線的目標長度
STEP_SIZE    = 0.08 * R        # 數值積分步距（越小越平滑）
SEG_LEN      = 0.20 * R        # 每段線段長
GAP_LEN      = 0.08 * R        # 段與段之間的間隙
HEAD_SCALE   = 0.25            # 箭頭比例（相對於段長）
#COLOR_LINE   = Color(0.2, 0.6, 1.0, 0.9)
#COLOR_HEAD   = Color(0.2, 0.6, 1.0, 0.9)

# %% [markdown]
# 畫出DNA雙股

# %%
def helix_points(radius, pitch, turns, n_per_turn, phase=0.0):
    pts = []
    total = int(n_per_turn * turns)
    for i in range(total + 1):
        t = 2.0 * math.pi * i / n_per_turn  
        x = radius * math.cos(t + phase)
        y = radius * math.sin(t + phase)
        z = pitch * i / n_per_turn
        pts.append(Point(x, y, z))
    return pts

pts1 = helix_points(R, pitch, turns, points_per_turn, phase=0.0)
pts2 = helix_points(R, pitch, turns, points_per_turn, phase=math.pi)

poly1 = Polyline(pts1)
poly2 = Polyline(pts2)
#viewer.scene.add(poly1, name="strand_A", color=Color(0.15, 0.45, 1.0))  # blue
#viewer.scene.add(poly2, name="strand_B", color=Color(1.0, 0.25, 0.25))  # red

# %% [markdown]
# 畫出鹼基

# %%
step = max(1, int(points_per_turn / rungs_per_turn))   # 每隔多少取樣點畫一根
for i in range(0, min(len(pts1), len(pts2)) - 1, step):
    p = pts1[i]
    q = pts2[i]
    #col = Color(0.75, 0.75, 0.75) if (i // step) % 2 == 0 else Color(0.55, 0.55, 0.55)
    #viewer.scene.add(Line(p, q), color=col)

# %% [markdown]
# 將每個橫向鹼基變成樓層面

# %%
extrude_height = flor

N = int(round(rungs_per_turn * turns))

def quad_at_center_and_dir(center: Point, u_dir: Vector,length: float , width: float):
    u = u_dir.copy()
    if u.length == 0:
        u = Vector(1, 0, 0)
    u.unitize()
    v = Vector(0, 0, 1).cross(u)
    if v.length == 0:
        v = Vector(1, 0, 0).cross(u)
    v.unitize()

    halfL, halfW = 0.5*length, 0.5*width
    a = center + (u * halfL) + (v * halfW)
    b = center - (u * halfL) + (v * halfW)
    c = center - (u * halfL) - (v * halfW)
    d = center + (u * halfL) - (v * halfW)
    return [a, b, c, d]

def prism_from_quad_and_height(quad, height):
    lift = Vector(0, 0, height)
    a, b, c, d = quad
    a2, b2, c2, d2 = a + lift, b + lift, c + lift, d + lift
    vertices = [a, b, c, d, a2, b2, c2, d2]
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    return Mesh.from_vertices_and_faces(vertices, faces)

for n in range(N):
    t = 2.0 * math.pi * (n / rungs_per_turn)
    zc = n * flor

    p1 = Point(R * math.cos(t), R * math.sin(t), zc)
    p2 = Point(-R * math.cos(t), -R * math.sin(t), zc)
    center = Point(0, 0, zc)
    u_dir = Vector.from_start_end(p1, p2)
    length = (p1 - p2).length
    quad = quad_at_center_and_dir(center, u_dir, length, RUNG_WIDTH)
    solid = prism_from_quad_and_height(quad, extrude_height)
    
    col = Color(0.80, 0.80, 0.85) if (n % 2 == 0) else Color(0.62, 0.62, 0.68)
    viewer.scene.add(solid)

# %% [markdown]
# CFD模擬

# %%
segments = []
wind_vectors = []

# %%
def velocity_field(p: Point) -> Vector:
    v = WIND_DIR.unitized()

    # 以 DNA 高度中線為中心製造繞流偏轉（只在 YZ 平面扭曲）
    cy = 0.5 * (Y_RANGE[0] + Y_RANGE[1])
    cz = 0.5 * (Z_RANGE[0] + Z_RANGE[1])
    dy, dz = p.y - cy, p.z - cz
    r2 = dy*dy + dz*dz
    if r2 < (3.0 * R_OBS)**2:
        phi = math.atan2(dz, dy)
        t_y, t_z = -math.sin(phi), math.cos(phi)
        w = ALPHA * (R_OBS**2 / max(r2, R_OBS**2))
        if r2 < (R_OBS**2):
            w *= 2.5
        v.y += w * t_y
        v.z += w * t_z

    if v.length < 1e-9:
        v = Vector(1, 0, 0)

    # 保持在水平層移除 Z 分量(減低電腦運算時間)
    v.z = 0.0

    return v.unitized()

def rk2(p: Point, h: float) -> Point:
    """2 階 Runge-Kutta 積分一步（較平滑）。"""
    k1 = velocity_field(p)
    mid = p + k1 * (0.5 * h)
    k2 = velocity_field(mid)
    return p + k2 * h

MAX_OBJECTS = 8000
_obj_count = 0

def add_arrow_segment(p0: Point, p1: Point):
    """畫一段線段 + 箭頭（小 'V' 形）"""
    global _obj_count
    if _obj_count >= MAX_OBJECTS:
        return
    # 主段
    viewer.scene.add(Line(p0, p1), color=COLOR_LINE);_obj_count += 1

    # 箭頭（末端）
    dirv = Vector.from_start_end(p0, p1)
    L = dirv.length
    if L < 1e-9:
        return
    u = dirv.unitized()
    v = Vector(0, 0, 1).cross(u)
    if v.length < 1e-9:
        v = Vector(0, 1, 0).cross(u)
    v.unitize()

    head_len = HEAD_SCALE * L
    head_wid = 0.5 * HEAD_SCALE * L
    tip  = p1
    base = p1 - u * head_len
    left = base + v * head_wid
    right= base - v * head_wid

    viewer.scene.add(Line(left, tip))
    viewer.scene.add(Line(right, tip))

def add_streamline(seed: Point):
    """從 seed 沿 +X 積分，並用分段（SEG_LEN / GAP_LEN）繪製箭頭虛線。"""
    total = 0.0
    cur = seed
    carry = 0.0  
    seg_start = cur

    while total < STREAM_LEN:
        nxt = rk2(cur, STEP_SIZE)
        step_d = Vector.from_start_end(cur, nxt).length

        # 碰到「柱體保護圈」就跳過（避免穿體）
        if (cur.y**2 + (cur.z - 0.5*(Z_RANGE[0]+Z_RANGE[1]))**2) < (0.9*R_OBS)**2:
            # 往切線向推一點再繼續
            phi = math.atan2(cur.z - 0.5*(Z_RANGE[0]+Z_RANGE[1]), cur.y)
            push = Vector(-math.sin(phi), math.cos(phi), 0.0) * (0.05*R)
            cur = cur + push
            continue

        carry += step_d
        total += step_d
        cur = nxt

        if carry >= SEG_LEN:
            add_arrow_segment(seg_start, cur)
            dirv = Vector.from_start_end(seg_start, cur).unitized()
            gap_advance = dirv * GAP_LEN
            seg_start = cur + gap_advance
            cur = seg_start
            carry = 0.0

def add_windfield():
    y0, y1 = Y_RANGE
    z0, z1 = Z_RANGE
    for iy in range(N_Y):
        y = y0 + (y1 - y0) * (iy + 0.5) / N_Y
        for iz in range(N_Z):
            z = z0 + (z1 - z0) * (iz + 0.5) / N_Z
            seed = Point(-2.5*R, y, z)  # 從 DNA 左側（負 X）邊界吹入
            add_streamline(seed)

add_windfield()
print("✅ Windfield added successfully.")

# %%
while length < STREAM_LEN:

    v = WIND_DIR * U + swirl_around_DNA(p, R, ALPHA, R_OBS)
    wind_vectors.append(v)

    swirl = swirl_around_DNA(p, R, ALPHA, R_OBS)
    speed_for_color = swirl.length

    if v.length < 1e-8:
        step_vec = Vector(0, 0, 0)
    else:
        step_vec = v.unitized() * STEP_SIZE

    p_next = p + step_vec

    segments.append((p, p_next, speed_for_color))

    p = p_next
    length += STEP_SIZE

# %%
COLOR_SLOW = Color(0.2, 0.6, 1.0, 0.9)
COLOR_FAST = Color(1.0, 0.3, 0.2, 0.9)

if segments:
    speeds = [s for _, _, s in segments]
    print("speed min/max =", min(speeds), max(speeds))  # 順便看一下

    s_min = min(speeds)
    s_max = max(speeds)
    if abs(s_max - s_min) < 1e-8:
        s_max = s_min + 1.0

    def lerp_color(c0, c1, t):
        return Color(
            c0.r + (c1.r - c0.r) * t,
            c0.g + (c1.g - c0.g) * t,
            c0.b + (c1.b - c0.b) * t,
            c0.a + (c1.a - c0.a) * t,
        )

    for p, p_next, s in segments:
        t = (s - s_min) / (s_max - s_min)
        col = lerp_color(COLOR_SLOW, COLOR_FAST, t)
        viewer.scene.add(Line(p, p_next), color=col)

# %%
viewer.show()

# %%
import json
import os

data = {
    "brand": "Ford",
    "model": "Mustang",
    "year": 2025
}

# 輸出路徑（用 getcwd 比 __file__ 安全）
output_path = os.path.join(
    os.getcwd(),  # 現在工作資料夾
    'wind_dirc_result.json'   # 檔名
)

# 寫入 JSON
with open(output_path, 'w', encoding='utf-8') as fp:
    json.dump(data, fp, indent=4)
    print(f"✅ JSON 檔案已建立：{output_path}")


# %% [markdown]
# 評分

# %%
import numpy as np

def compute_wind_score(vectors, U_inf=1.0, rho=1.225):
    """
    vectors : list[Vector]  所有風場中的速度向量
    U_inf   : 自由流 (你上面設成 U = 1.0)
    rho     : 空氣密度
    回傳 dict: {'avg_speed', 'momentum_deficit', 'score'}
    """
    if not vectors:
        return {"avg_speed": 0.0, "momentum_deficit": 0.0, "score": 0.0}

    speeds = np.array([v.length for v in vectors], dtype=float)

    # 不讓比 U_inf 更快的地方加分（夾成 U_inf）
    speeds_clipped = np.minimum(speeds, U_inf)

    # 簡化版「動量虧損」
    deficit = rho * (U_inf**2 - speeds_clipped**2)
    DeltaM = float(np.sum(deficit))

    # 最差情況：全部速度都 0
    worst = rho * (U_inf**2) * len(speeds_clipped)

    score = (worst - DeltaM) / worst * 100.0
    score = float(np.clip(score, 0.0, 100.0))

    return {
        "avg_speed": float(np.mean(speeds)),
        "momentum_deficit": DeltaM,
        "score": score,}

# %%
from compas.geometry import Vector

def swirl_around_DNA(p, R, ALPHA, R_OBS):
    """
    計算繞著 DNA 中心軸的旋流速度分量
    p: Point 目前的位置
    R: DNA 半徑
    ALPHA: 旋流強度（越大渦旋越明顯）
    R_OBS: 有效半徑（超過這範圍旋流衰減）
    """

    # 計算在 YZ 平面離 DNA 中心的距離
    r = (p.y**2 + p.z**2)**0.5

    if r < 1e-6:   # 避免除以零
        return Vector(0, 0, 0)

    if r > R_OBS:  # 超過觀測半徑 → 渦旋衰減
        return Vector(0, 0, 0)

    # 渦旋強度（離中心越遠越弱）
    strength = ALPHA * (1 - r / R_OBS)

    # 旋流方向（在 YZ 平面逆時針旋轉）
    vy = -strength * (p.z / r)
    vz =  strength * (p.y / r)

    return Vector(0, vy, vz)

# %%
wind_vectors = []

X_START = -5 * R   # 流線起點 X 座標

for iy in range(N_Y):
    for iz in range(N_Z):

        # 1. 計算 seed 點座標 (y, z)
        y = Y_RANGE[0] + (Y_RANGE[1] - Y_RANGE[0]) * iy / max(1, N_Y - 1)
        z = Z_RANGE[0] + (Z_RANGE[1] - Z_RANGE[0]) * iz / max(1, N_Z - 1)

        # 2. seed 點位置（風從 -x → +x）
        p = Point(X_START, y, z)

        length = 0.0
        while length < STREAM_LEN:

            # 計算當下速度向量
            v = WIND_DIR * U + swirl_around_DNA(p, R, ALPHA, R_OBS)

            # 儲存速度向量以便評分
            wind_vectors.append(v)

            # 用 v 走一步
            p_next = p + v.unitized() * STEP_SIZE

            #  viewer.scene.add(Line(p, p_next), color=COLOR_LINE)

            length += STEP_SIZE
            p = p_next

# %%
result = compute_wind_score(wind_vectors, U_inf=U)

print("📊 風場評分結果：")
print(f"平均風速: {result['avg_speed']:.2f} (模型內的實際平均)")
print(f"動量虧損: {result['momentum_deficit']:.2f} (相對指標)")
print(f"評分 (0–100): {result['score']:.1f}")


