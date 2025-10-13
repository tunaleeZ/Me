

import json, time
import numpy as np
import cv2
import torch
import torchvision.transforms as T
import pyrealsense2 as rs


from trt_pose.models import resnet18_baseline_att
from trt_pose.parse_objects import ParseObjects
from trt_pose.draw_objects import DrawObjects

# cauhinh
HUMAN_POSE_JSON = "trt_pose/tasks/human_pose/human_pose.json"  # topology COCO
WEIGHTS_PTH     = "resnet18_baseline_att_224x224.pth"          # PyTorch weights
INPUT_SIZE      = (224, 224)    # (W, H) model input
CMAP_THR        = 0.15          # ngưỡng peak
LINK_THR        = 0.15          # ngưỡng nối xương
DEPTH_KERNEL    = 5             # median k×k 
VIDEO_FPS       = 30.0
FRAME_SIZE      = (640, 480)    # RealSense stream

def depth_median(depth_img, x, y, k=5):
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
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], float(depth_m))
    return X, Y, Z

def draw_label_two_lines(img, x, y, line1, line2, font=cv2.FONT_HERSHEY_SIMPLEX):
    h, w = img.shape[:2]
    scale1, thick1 = 0.6, 2
    scale2, thick2 = 0.5, 1
    (w1, h1), base1 = cv2.getTextSize(line1, font, scale1, thick1)
    (w2, h2), base2 = cv2.getTextSize(line2, font, scale2, thick2)
    tx, ty = x + 8, y - 10
    tw = max(w1, w2)
    tx = max(0, min(tx, w - tw))
    ty = max(h1 + max(base1, base2), min(ty, h - max(base1, base2)))
    pad = 4
    box_top = max(0, ty - h1 - base1 - pad)
    box_bot = min(h - 1, ty + pad + h2 + base2)
    box_left = max(0, tx - pad)
    box_right = min(w - 1, tx + tw + pad)
    cv2.rectangle(img, (box_left, box_top), (box_right, box_bot), (0, 0, 0), -1)
    cv2.putText(img, line1, (tx, ty), font, scale1, (255, 255, 255), thick1, cv2.LINE_AA)
    cv2.putText(img, line2, (tx, ty + h2 + base2 + 2), font, scale2, (200, 200, 200), thick2, cv2.LINE_AA)

# nap topology
with open(HUMAN_POSE_JSON, 'r') as f:
    human_pose = json.load(f)

keypoints = human_pose['keypoints']   # danh sách tên khớp (COCO)
skeleton  = human_pose['skeleton']    # các cặp nối (1-based)

# skeleton 1-based -> tensor 0-based
topology = torch.zeros((len(skeleton), 2), dtype=torch.int32)
for i, sk in enumerate(skeleton):
    topology[i, 0] = sk[0] - 1
    topology[i, 1] = sk[1] - 1

# name -> index
def kp_idx(name): return keypoints.index(name)


NOSE_ID        = kp_idx('nose')
LEFT_WRIST_ID  = kp_idx('left_wrist')
RIGHT_WRIST_ID = kp_idx('right_wrist')
TARGET_IDS     = [NOSE_ID, LEFT_WRIST_ID, RIGHT_WRIST_ID]

# model pytorch 
num_parts = len(keypoints)
num_links = len(skeleton)

model = resnet18_baseline_att(num_parts, num_links)
state = torch.load(WEIGHTS_PTH, map_location='cpu')
model.load_state_dict(state)
model.eval().cuda()

# Parser & Drawer
parse_objects = ParseObjects(topology, cmap_threshold=CMAP_THR, link_threshold=LINK_THR)
draw_objects  = DrawObjects(topology)

# Chuẩn hóa input (ImageNet)
to_tensor = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225])
])

# suy luan pose
@torch.inference_mode()
def trtpose_infer(img_bgr):
    """
    Trả về:
      canvas: ảnh có vẽ skeleton
      lm_xy:  danh sách [(part_id, x_px, y_px), ...] cho person đầu tiên (nếu có)
    """
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    tensor = to_tensor(img_resized).unsqueeze(0).cuda()  # float32

    cmap, paf = model(tensor)                 # [1,C,h',w'] và [1,2L,h',w']
    cmap_cpu, paf_cpu = cmap.cpu(), paf.cpu()
    counts, objects, peaks = parse_objects(cmap_cpu, paf_cpu)

    canvas = img_bgr.copy()
    draw_objects(canvas, counts, objects, peaks)

    lm_xy = []
    if int(counts[0]) > 0:
        person_idx = 0
        cmap_h, cmap_w = cmap_cpu.shape[2], cmap_cpu.shape[3]
        in_w, in_h = INPUT_SIZE[0], INPUT_SIZE[1]  # 224, 224
        for part_id in range(num_parts):
            k = int(objects[0][person_idx][part_id])
            if k < 0:
                continue
            # peaks[part_id]: (1 + n_peaks, 3) -> (y, x, score); index 0 là số peak
            peak = peaks[part_id][1 + k]
            y_peak, x_peak = float(peak[0]), float(peak[1])

            # cmap-space -> input-size (224)
            x_in = x_peak / max(1.0, (cmap_w - 1)) * (in_w - 1)
            y_in = y_peak / max(1.0, (cmap_h - 1)) * (in_h - 1)

            # input-size -> ảnh gốc (W,H)
            x_px = int(x_in / max(1.0, (in_w - 1)) * (W - 1))
            y_px = int(y_in / max(1.0, (in_h - 1)) * (H - 1))

            x_px = max(0, min(x_px, W - 1))
            y_px = max(0, min(y_px, H - 1))
            lm_xy.append((part_id, x_px, y_px))

    return canvas, lm_xy

# realsense
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, FRAME_SIZE[0], FRAME_SIZE[1], rs.format.bgr8, int(VIDEO_FPS))
cfg.enable_stream(rs.stream.depth, FRAME_SIZE[0], FRAME_SIZE[1], rs.format.z16, int(VIDEO_FPS))
profile = pipe.start(cfg)

depth_sensor = profile.get_device().first_depth_sensor()
DEPTH_SCALE  = float(depth_sensor.get_depth_scale())   # m/LSB
align        = rs.align(rs.stream.color)               # depth -> color

# ghi video
fourcc   = cv2.VideoWriter_fourcc(*'XVID')
out_rgb  = cv2.VideoWriter('rgb_output.avi',   fourcc, VIDEO_FPS, FRAME_SIZE)
out_dep  = cv2.VideoWriter('depth_output.avi', fourcc, VIDEO_FPS, FRAME_SIZE)

pTime = time.time()
try:
    while True:
        frames = pipe.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())   # uint16
        color_image = np.asanyarray(color_frame.get_data())   # BGR

        # tô màu depth để hiển thị
        depth_8u    = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

        # pose estimation (PyTorch)
        frame_draw, lm_xy = trtpose_infer(color_image)

        # id -> (x,y)
        id2xy = {pid:(x,y) for (pid,x,y) in lm_xy}

        # Lấy 3 khớp mục tiêu: mũi + cổ tay T/P
        for pid in [NOSE_ID, LEFT_WRIST_ID, RIGHT_WRIST_ID]:
            if pid not in id2xy:
                continue
            x, y = id2xy[pid]
            raw = depth_median(depth_image, x, y, k=DEPTH_KERNEL)
            if raw is None:
                cv2.circle(frame_draw, (x, y), 5, (0, 255, 255), 2)
                draw_label_two_lines(frame_draw, x, y, "N/A", "X=?, Y=?, Z=?")
                continue

            d_m  = raw * DEPTH_SCALE          # m
            d_cm = d_m * 100.0                # cm
            X, Y, Z = deproject_xyz_from_pixel(color_frame, x, y, d_m)

            cv2.circle(frame_draw, (x, y), 5, (0, 255, 255), 2)
            line1 = f"{d_cm:.1f} cm"
            line2 = f"X={X:.2f}m  Y={Y:.2f}m  Z={Z:.2f}m"
            draw_label_two_lines(frame_draw, x, y, line1, line2)

        # FPS
        cTime = time.time()
        fps_now = 1.0 / max(1e-6, cTime - pTime)
        pTime = cTime
        cv2.putText(frame_draw, f"FPS: {fps_now:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # ghi & show
        out_rgb.write(frame_draw)
        out_dep.write(depth_color)
        cv2.imshow('rgb_trtpose_pytorch', frame_draw)
        cv2.imshow('depth', depth_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipe.stop()
    out_rgb.release()
    out_dep.release()
    cv2.destroyAllWindows()
