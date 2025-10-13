import os
import time
import json
import numpy as np
import cv2
import torch
import torchvision.transforms as T

import pyrealsense2 as rs

# ======== TRT Pose imports ========
from trt_pose.models import resnet18_baseline_att
from trt_pose.parse_objects import ParseObjects
from trt_pose.plugins import *
from trt_pose.draw_objects import DrawObjects

# ================== CẤU HÌNH CAMERA (RealSense) ==================
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipe.start(cfg)
depth_sensor = profile.get_device().first_depth_sensor()
DEPTH_SCALE = float(depth_sensor.get_depth_scale())  # m/LSB

align = rs.align(rs.stream.color)  # align depth -> color

# ================== GHI VIDEO ==================
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # .avi
fps = 30.0     # khớp stream
size = (640, 480)
out_rgb   = cv2.VideoWriter('rgb_output.avi',   fourcc, fps, size)
out_depth = cv2.VideoWriter('depth_output.avi', fourcc, fps, size)
enable_both = False

# ================== TẢI TOPOLOGY & MODEL TRT POSE ==================
# Đường dẫn bạn chỉnh lại cho đúng máy
HUMAN_POSE_JSON = "trt_pose/tasks/human_pose/human_pose.json"
WEIGHTS_PTH     = "resnet18_baseline_att_224x224.pth"  # PyTorch weights (nếu chưa convert TRT)

with open(HUMAN_POSE_JSON, 'r') as f:
    human_pose = json.load(f)

# Danh sách keypoint COCO (trt_pose) ví dụ: nose, left_eye, right_eye, left_ear, right_ear,
# left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,
# left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle
keypoints = human_pose['keypoints']
skeleton  = human_pose['skeleton']

# Lấy index các điểm cần theo dõi
def kp_idx(name):
    return keypoints.index(name)

NOSE_ID       = kp_idx('nose')
LEFT_WRIST_ID = kp_idx('left_wrist')
RIGHT_WRIST_ID= kp_idx('right_wrist')
TARGET_IDS    = [NOSE_ID, LEFT_WRIST_ID, RIGHT_WRIST_ID]

# Tạo topology tensor cho parser/drawer
import torch
topology = torch.zeros((len(skeleton), 2), dtype=torch.int32)
for i, sk in enumerate(skeleton):
    # trong json đánh số 1-based
    topology[i, 0] = sk[0] - 1
    topology[i, 1] = sk[1] - 1

# Model backbone (độ phân giải đầu vào 224x224 cho resnet18 baseline att)
num_parts = len(keypoints)
num_links = len(skeleton)

model = resnet18_baseline_att(num_parts, num_links)
model.load_state_dict(torch.load(WEIGHTS_PTH, map_location='cpu'))
model.eval().cuda()

# Nếu bạn đã có engine TensorRT, có thể thay bằng engine runner (xem notebook trt_pose để convert).
# Ở đây dùng PyTorch trước cho đơn giản; Orin vẫn đủ nhanh, hoặc bạn chuyển sang TRT để tối ưu.

# Parser & Drawer từ trt_pose
parse_objects = ParseObjects(topology, cmap_threshold=0.15, link_threshold=0.15)
draw_objects  = DrawObjects(topology)

# Preprocess transform (chuẩn ImageNet, resize về 224)
input_size = (224, 224)
to_tensor = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225])
])

# ================== HÀM TIỆN ÍCH ==================
def depth_median(depth_img, x, y, k=5):
    """Lấy median độ sâu quanh (x,y) trong ô k×k, bỏ 0."""
    h, w = depth_img.shape[:2]
    r = k // 2
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    patch = depth_img[y0:y1, x0:x1]
    vals = patch[patch > 0]
    if vals.size == 0:
        return None
    return int(np.median(vals))

def deproject_xyz_from_pixel(color_frame, x, y, depth_m):
    """Trả về (X,Y,Z) theo mét trong hệ CAMERA MÀU:
       +X: sang phải ảnh, +Y: xuống, +Z: ra trước camera."""
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], float(depth_m))
    return X, Y, Z

def draw_label_two_lines(img, x, y, line1, line2, font=cv2.FONT_HERSHEY_SIMPLEX):
    h, w = img.shape[:2]
    scale1, thick1 = 0.6, 2
    scale2, thick2 = 0.5, 1
    (w1, h1), base1 = cv2.getTextSize(line1, font, scale1, thick1)
    (w2, h2), base2 = cv2.getTextSize(line2, font, scale2, thick2)

    tx = x + 8
    ty = y - 10
    tw = max(w1, w2)
    tx = max(0, min(tx, w - tw))
    max_base = max(base1, base2)
    ty = max(h1 + max_base, min(ty, h - max_base))

    pad = 4
    box_top = max(0, ty - h1 - base1 - pad)
    box_bot = min(h - 1, ty + pad + h2 + base2)
    box_left = max(0, tx - pad)
    box_right = min(w - 1, tx + tw + pad)
    cv2.rectangle(img, (box_left, box_top), (box_right, box_bot), (0, 0, 0), -1)

    cv2.putText(img, line1, (tx, ty), font, scale1, (255, 255, 255), thick1, cv2.LINE_AA)
    cv2.putText(img, line2, (tx, ty + h2 + base2 + 2), font, scale2, (200, 200, 200), thick2, cv2.LINE_AA)

def trtpose_infer(img_bgr):
    """
    img_bgr: (H, W, 3) BGR 0..255
    Trả về:
      - canvas để vẽ skeleton
      - danh sách keypoints theo pixel [(idx, x, y), ...] cho person đầu tiên (nếu có)
    """
    h, w = img_bgr.shape[:2]
    # resize + normalize
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size, interpolation=cv2.INTER_LINEAR)
    tensor = to_tensor(img_resized).unsqueeze(0).cuda()

    with torch.no_grad():
        cmap, paf = model(tensor)  # cmap: [1, C, h', w'], paf: [1, 2L, h', w']

    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)  # từ trt_pose

    # Vẽ skeleton lên bản sao ảnh gốc
    canvas = img_bgr.copy()
    draw_objects(canvas, counts, objects, peaks)

    # Lấy keypoints pixel cho "person" đầu tiên (nếu có)
    lm_xy = []
    if int(counts[0]) > 0:
        person_idx = 0
        # peaks: list theo từng part, mỗi peak là (y, x, score) ở scale của cmap/paf
        # cần map về tọa độ ảnh gốc.
        # Tính tỉ lệ scale giữa cmap và input_size/ảnh gốc:
        # DrawObjects đã dùng nội suy riêng, ở đây ta lấy trực tiếp peak to pixel theo input_size,
        # rồi scale về (w, h):
        for part_id in range(num_parts):
            peak_count = int(peaks[part_id][0])  # số peak cho part này
            x_px, y_px = None, None
            if peak_count > 0:
                # lấy peak mà object 'objects' gán (nếu có)
                k = int(objects[0][person_idx][part_id])
                if k >= 0:
                    peak = peaks[part_id][1 + k]
                    y_peak, x_peak = float(peak[0]), float(peak[1])  # theo tensor cmap
                    # các peak ở thang của cmap (thường 56x56 cho input 224); map về input_size
                    # tỉ lệ giữa input_size và cmap:
                    cmap_h, cmap_w = cmap.shape[2], cmap.shape[3]
                    x_in = x_peak / (cmap_w - 1) * (input_size[0] - 1)
                    y_in = y_peak / (cmap_h - 1) * (input_size[1] - 1)
                    # map từ input_size (224) về kích thước ảnh gốc (w,h):
                    x_px = int(x_in / (input_size[0] - 1) * (w - 1))
                    y_px = int(y_in / (input_size[1] - 1) * (h - 1))
            if x_px is None or y_px is None:
                continue
            # kẹp biên
            x_px = max(0, min(x_px, w - 1))
            y_px = max(0, min(y_px, h - 1))
            lm_xy.append((part_id, x_px, y_px))

    return canvas, lm_xy

# ================== VÒNG LẶP CHÍNH ==================
pTime = time.time()
try:
    while True:
        # lấy & align frame
        frames = pipe.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())    # uint16
        color_image = np.asanyarray(color_frame.get_data())    # BGR

        # tô màu depth để hiển thị
        depth_8u = cv2.convertScaleAbs(depth_image, alpha=0.03)  # bạn có thể thay bằng normalize
        depth_cm_img = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

        # ====== TRT Pose inference ======
        frame_draw, lm_xy = trtpose_infer(color_image)

        # Lập map {part_id: (x,y)}
        id2xy = {idx:(x,y) for (idx,x,y) in lm_xy}

        # Theo dõi NOSE + WRISTS
        for lid in TARGET_IDS:
            if lid not in id2xy:
                continue
            x, y = id2xy[lid]

            raw = depth_median(depth_image, x, y, k=5)
            if raw is None:
                cv2.circle(frame_draw, (x, y), 5, (0, 255, 255), 2)
                draw_label_two_lines(frame_draw, x, y, "N/A", "X=?, Y=?, Z=?")
            else:
                d_m  = raw * DEPTH_SCALE
                d_cm = d_m * 100.0

                X, Y, Z = deproject_xyz_from_pixel(color_frame, x, y, d_m)

                cv2.circle(frame_draw, (x, y), 5, (0, 255, 255), 2)
                line1 = f"{d_cm:.1f} cm"
                # Quy ước: +X phải, +Y xuống, +Z ra trước camera (hệ camera màu)
                line2 = f"X={X:.2f}m  Y={Y:.2f}m  Z={Z:.2f}m"
                draw_label_two_lines(frame_draw, x, y, line1, line2)

        # FPS
        cTime = time.time()
        disp_fps = 1.0 / max(1e-6, cTime - pTime)
        pTime = cTime
        cv2.putText(frame_draw, f"FPS: {disp_fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # ghi video
        out_rgb.write(frame_draw)
        out_depth.write(depth_cm_img)
        if enable_both:
            both = np.hstack((frame_draw, depth_cm_img))  # (480, 1280, 3)
            # out_both.write(both)

        # hiển thị
        cv2.imshow('rgb_trtpose', frame_draw)
        cv2.imshow('depth', depth_cm_img)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipe.stop()
    out_rgb.release()
    out_depth.release()
    cv2.destroyAllWindows()
