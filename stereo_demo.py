# -*- coding: utf-8 -*-
"""
双目测距 · 合成演示（不需要摄像头和真实棋盘格）

流程：合成棋盘格照片 → 单目标定 → 双目标定(求基线) → 极线校正
      → 合成"带真值深度"的左右场景图 → SGBM 视差 → 深度 → 和真值对比

环境：Python 3.10+，需要 opencv-python、numpy、matplotlib
运行：python stereo_demo.py   （结果图存到 ../../images/ 即 outputs/images/）
"""
import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import cv2

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")  # 结果图保存到项目自己的 images 文件夹
os.makedirs(OUT_DIR, exist_ok=True)

# ============ 0. 虚拟双目相机（真值参数，用来生成数据） ============
W, H = 640, 480
FX = 800.0                          # 内参 fx（像素）
B_TRUE_MM = 100.0                   # 基线 100mm = 10cm（棋盘标定用毫米单位）
B_TRUE_M  = 0.10                    # 基线 0.10m（场景 3D 坐标统一用米单位）
K_TRUE = np.array([[FX, 0 , W / 2],
                         [ 0, FX, H / 2],
                         [ 0, 0 ,   1  ]], dtype=np.float64)
D_TRUE = np.zeros(5)

def project_left(X):
    """左相机坐标系的 3D 点 -> 左图像素"""
    p = X @ K_TRUE.T
    return (p[:, :2] / np.maximum(p[:, 2:3], 1e-6)).astype(np.float32)

def project_right(X, baseline):
    """左相机坐标系的 3D 点 -> 右图像素（右镜头沿 x 方向偏移 baseline，单位与 X 一致）"""
    Xr = X - np.array([baseline, 0, 0])
    p = Xr @ K_TRUE.T
    return (p[:, :2] / np.maximum(p[:, 2:3], 1e-6)).astype(np.float32)

def checker_texture(inner=(9, 6), sq_px=60, sq_mm=25.0):
    """生成棋盘纹理；棋盘面尺寸按 mm 定义（内角点 9x6 -> 10x7 个格子）"""
    ncols, nrows = inner[0] + 1, inner[1] + 1
    tex = np.zeros((nrows * sq_px, ncols * sq_px), np.uint8)
    for i in range(nrows):
        for j in range(ncols):
            if (i + j) % 2 == 0:
                tex[i * sq_px:(i + 1) * sq_px, j * sq_px:(j + 1) * sq_px] = 255
    sx, sy = ncols * sq_mm / 2, nrows * sq_mm / 2
    quad = np.array([[-sx, -sy, 0], [sx, -sy, 0],
                     [sx, sy, 0], [-sx, sy, 0]], dtype=np.float64)
    return tex, quad

def render_quad(tex, quad3d, project_fn, img):
    """把纹理贴到 3D 四边形上（投影成像素多边形）"""
    pix = project_fn(quad3d)                      # (4,2)
    th, tw = tex.shape
    src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, pix)
    return cv2.warpPerspective(tex, M, (W, H), dst=img,
                               borderMode=cv2.BORDER_TRANSPARENT)

# ============ 1. 合成棋盘格标定照片（左右相机各 12 张） ============
def board_poses(n=12):
    """生成 n 组棋盘位姿，并且保证左右相机里棋盘都完整在画面内"""
    rng = np.random.default_rng(42)
    tex_board, quad_board = checker_texture()
    poses = []
    tries = 0
    while len(poses) < n and tries < 500:
        tries += 1
        rvec = rng.uniform(-0.25, 0.25, size=3)   # 绕 x/y/z 小角度转
        R, _ = cv2.Rodrigues(rvec.astype(np.float64))
        t = np.array([rng.uniform(-60, 60),
                      rng.uniform(-50, 50),
                      rng.uniform(520, 620)], dtype=np.float64)
        quad_cam = (R @ quad_board.T).T + t
        for fn in (project_left, lambda X: project_right(X, B_TRUE_MM)):
            pix = fn(quad_cam)
            if pix[:, 0].min() < 10 or pix[:, 0].max() > W - 11 or \
               pix[:, 1].min() < 10 or pix[:, 1].max() > H - 11:
                break
        else:
            poses.append((R, t))
    return poses

tex_board, quad_board = checker_texture()
pattern = (9, 6)
objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * 25.0

objpoints, ptsL, ptsR = [], [], []
poses = board_poses()
for i, (R, t) in enumerate(poses):
    quad_cam = (R @ quad_board.T).T + t          # 棋盘面坐标 -> 左相机坐标
    imgL = render_quad(tex_board, quad_cam, project_left,
                       np.zeros((H, W), np.uint8))
    imgR = render_quad(tex_board, quad_cam,
                       lambda X: project_right(X, B_TRUE_MM),
                       np.zeros((H, W), np.uint8))
    okL, cL = cv2.findChessboardCornersSB(imgL, pattern)   # OpenCV 5 新版检测器
    okR, cR = cv2.findChessboardCornersSB(imgR, pattern)
    if okL and okR:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
        cL = cv2.cornerSubPix(imgL, cL, (11, 11), (-1, -1), crit)
        cR = cv2.cornerSubPix(imgR, cR, (11, 11), (-1, -1), crit)
        objpoints.append(objp); ptsL.append(cL); ptsR.append(cR)
    if i == 0 and okL:                           # 存一张标定图给你看
        cv2.drawChessboardCorners(imgL, pattern, cL, okL)
        cv2.imwrite(os.path.join(OUT_DIR, "stereo_calib_check.png"), imgL)

print(f"成功检测棋盘角点的照片对数: {len(objpoints)} / {len(poses)}")

# ============ 2. 单目标定 + 双目标定 ============
retL, K1, D1, _, _ = cv2.calibrateCamera(objpoints, ptsL, (W, H), None, None)
retR, K2, D2, _, _ = cv2.calibrateCamera(objpoints, ptsR, (W, H), None, None)
retS, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, ptsL, ptsR, K_TRUE, D_TRUE, K_TRUE, D_TRUE, (W, H),
    flags=cv2.CALIB_FIX_INTRINSIC)

B_est_mm = np.linalg.norm(T)
print(f"单目标定重投影误差: 左={retL:.4f}px  右={retR:.4f}px")
print(f"双目标定求出的基线: {B_est_mm:.2f} mm（真值 100.00 mm）")

# ============ 3. 极线校正 ============
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (W, H), R, T, alpha=0)
m1x, m1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (W, H), cv2.CV_32FC1)
m2x, m2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (W, H), cv2.CV_32FC1)

# ============ 4. 合成"带真值深度"的场景（近处物体视差大） ============
def project_rect(X, P, R):
    """3D 点 -> 校正后图像像素（用立体校正输出的 P1/R1，采样位置更准）"""
    Xr = (R @ X.T).T
    p = (Xr @ P[:, :3].T) + P[:, 3]
    return (p[:, :2] / np.maximum(p[:, 2:3], 1e-6)).astype(np.float32)

rng = np.random.default_rng(7)
def rand_tex(seed, n=96):
    return (rng.random((n, n)) > 0.5).astype(np.uint8) * 255

# 远处的墙（Z=6m），铺满整个画面
wall_tex = np.zeros((H, W), np.uint8)
for i in range(0, H, 60):
    for j in range(0, W, 60):
        if (i // 60 + j // 60) % 2 == 0:
            wall_tex[i:i + 60, j:j + 60] = 110
wall_q = np.array([[-2.8, -1.8, 6.0], [2.8, -1.8, 6.0],
                   [2.8, 1.8, 6.0], [-2.8, 1.8, 6.0]], dtype=np.float64)

sceneL = render_quad(wall_tex, wall_q, project_left, np.zeros((H, W), np.uint8))
sceneR = render_quad(wall_tex, wall_q,
                     lambda X: project_right(X, B_TRUE_M),
                     np.zeros((H, W), np.uint8))
depth_true = np.full((H, W), 6.0, np.float32)    # 真值深度：默认 6m（墙）

# 物体：名字, 图像中心(u,v), 宽(m), 高(m), 深度(m)
# 用"图像中心 + 深度"反推世界坐标，保证所有物体在左右两张图里都完整可见、互不遮挡
objs = [
    ("近处正方体", 320, 160, 0.34, 0.26, 1.0),
    ("左前方块",   150, 380, 0.34, 0.26, 1.6),
    ("右下方块",   540, 360, 0.38, 0.28, 2.0),
    ("右上远物",   560, 120, 0.50, 0.36, 3.0),
    ("最远长条",   320, 430, 0.60, 0.40, 5.0),
]
centers = []
for name, u, v, w, h, Z in objs:
    cx = (u - W / 2) * Z / FX            # 由图像中心反推世界 X
    cy = (v - H / 2) * Z / FX            # 由图像中心反推世界 Y
    q = np.array([[cx - w / 2, cy - h / 2, Z], [cx + w / 2, cy - h / 2, Z],
                  [cx + w / 2, cy + h / 2, Z], [cx - w / 2, cy + h / 2, Z]],
                 dtype=np.float64)
    tex = rand_tex(len(centers))         # 每个物体用不同随机纹理
    sceneL = render_quad(tex, q, project_left, sceneL)
    sceneR = render_quad(tex, q, lambda X: project_right(X, B_TRUE_M), sceneR)
    poly = project_left(q).astype(np.int32)
    cv2.fillPoly(depth_true, [poly], Z)  # 真值深度写入
    centers.append((name, Z, project_rect(np.array([[cx, cy, Z]]), P1, R1)[0]))

cv2.imwrite(os.path.join(OUT_DIR, "stereo_scene_left.png"), sceneL)
cv2.imwrite(os.path.join(OUT_DIR, "stereo_scene_right.png"), sceneR)

# ============ 5. 校正 + SGBM 视差 + 深度 ============
rectL = cv2.remap(sceneL, m1x, m1y, cv2.INTER_LINEAR)
rectR = cv2.remap(sceneR, m2x, m2y, cv2.INTER_LINEAR)

sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=96, blockSize=11,
    P1=8 * 3 * 11 * 11, P2=32 * 3 * 11 * 11,
    disp12MaxDiff=1, uniquenessRatio=10,
    speckleWindowSize=100, speckleRange=32)
disp = sgbm.compute(rectL, rectR).astype(np.float32) / 16.0

fx_rect = P1[0, 0]                       # 校正后的焦距（理论=800）
B_m = B_est_mm / 1000.0                  # 双目标定求出的基线转成米
depth = np.where(disp > 0, fx_rect * B_m / np.maximum(disp, 1e-3), 0.0)

# ============ 6. 每个物体：真值 vs 实测 ============
print("\n物体          真值(m)  实测(m)  误差(cm)  视差(px)")
for name, Zt, (u, v) in centers:
    u, v = int(round(u)), int(round(v))
    patch = depth[max(0, v - 25):v + 25, max(0, u - 25):u + 25]
    valid = patch[patch > 0]
    Zm = float(np.median(valid)) if valid.size else float("nan")
    d_meas = float(np.median(disp[max(0, v - 25):v + 25, max(0, u - 25):u + 25]))
    print(f"{name:8s}  {Zt:6.2f}   {Zm:6.2f}   {abs(Zm - Zt) * 100:6.2f}   {d_meas:6.1f}")

# 视差图 / 深度图可视化（先设置中文字体，避免标题变方框）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for _f in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(_f):
        fm.fontManager.addfont(_f)
        plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=_f).get_name()]
        plt.rcParams["axes.unicode_minus"] = False
        break
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(sceneL, cmap="gray"); axes[0].set_title("左相机画面")
axes[0].axis("off")
im = axes[1].imshow(disp, cmap="turbo"); axes[1].set_title("视差图（亮=近）")
axes[1].axis("off"); fig.colorbar(im, ax=axes[1], fraction=0.046)
im = axes[2].imshow(depth, cmap="turbo", vmin=0, vmax=6)
axes[2].set_title("深度图（米）"); axes[2].axis("off")
fig.colorbar(im, ax=axes[2], fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "stereo_result.png"), dpi=110)
print("\n结果图已保存:")
for name in ("stereo_scene_left.png", "stereo_scene_right.png",
             "stereo_result.png", "stereo_calib_check.png"):
    print(" ", os.path.join(OUT_DIR, name))

