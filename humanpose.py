import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import trt_pose.coco
import trt_pose.models
import torch2trt
import json
from torch2trt import TRTModule
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from torchvision import transforms
import PIL.Image
import os
import time
import csv

# ------------------ Load topology ------------------
with open('preprocess/human_pose.json', 'r') as f:
    human_pose = json.load(f)
topology_body = trt_pose.coco.coco_category_to_topology(human_pose)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

# index cổ tay trái / phải trong human_pose
keypoints_list = human_pose['keypoints']
KP_LEFT_WRIST  = keypoints_list.index('left_wrist')
KP_RIGHT_WRIST = keypoints_list.index('right_wrist')

with open('preprocess/hand_pose.json', 'r') as f:
    hand_pose = json.load(f)
topology_hand = trt_pose.coco.coco_category_to_topology(hand_pose)
num_parts_hand = len(hand_pose['keypoints'])
num_links_hand = len(hand_pose['skeleton'])

# ------------------ Model & size config ------------------
WIDTH, HEIGHT = 224, 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------ Load body & hand models (TRT or fallback) ------------------
model = trt_pose.models.resnet18_baseline_att(num_parts, 2*num_links).to(device).eval()
model_hand = trt_pose.models.resnet18_baseline_att(num_parts_hand, 2*num_links_hand).to(device).eval()

# helper zero tensor for conversion
data = torch.zeros((1,3,HEIGHT,WIDTH)).to(device)

# BODY MODEL
OPTIMIZED_MODEL = 'models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
if not os.path.exists(OPTIMIZED_MODEL):
    MODEL_WEIGHTS = 'models/resnet18_baseline_att_224x224_A_epoch_249.pth'
    print("Converting body model to TRT (this may take a while)...")
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location='cpu'))
    model.to(device).eval()
    try:
        model_trt = torch2trt.torch2trt(
            model, [data],
            fp16_mode=True,
            max_workspace_size=1<<25
        )
        torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
    except Exception as e:
        print("torch2trt conversion failed (body):", e)
        raise

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
model_trt.to(device).eval()

# HAND MODEL
OPTIMIZED_MODEL_HAND = 'models/hand_pose_resnet18_att_244_244_trt.pth'
if not os.path.exists(OPTIMIZED_MODEL_HAND):
    MODEL_WEIGHTS_HAND = 'models/hand_pose_resnet18_att_244_244.pth'
    print("Converting hand model to TRT...")
    model_hand.load_state_dict(torch.load(MODEL_WEIGHTS_HAND, map_location='cpu'))
    model_hand.to(device).eval()
    try:
        model_trt_hand = torch2trt.torch2trt(
            model_hand, [data],
            fp16_mode=True,
            max_workspace_size=1<<25
        )
        torch.save(model_trt_hand.state_dict(), OPTIMIZED_MODEL_HAND)
    except Exception as e:
        print("torch2trt conversion failed (hand):", e)
        raise

model_trt_hand = TRTModule()
model_trt_hand.load_state_dict(torch.load(OPTIMIZED_MODEL_HAND))
model_trt_hand.to(device).eval()

# ------------------ Parse & draw helpers ------------------
# body
parse_objects_body = ParseObjects(topology_body, cmap_threshold=0.15, link_threshold=0.15)
# nếu muốn vẽ full skeleton người thì tạo DrawObjects(topology_body), ở đây mình không vẽ

# hand
parse_objects_hand = ParseObjects(topology_hand, cmap_threshold=0.15, link_threshold=0.15)
draw_objects_hand = DrawObjects(topology_hand)

# Your preprocessdata class instance (should match topology)
from preprocessdata import preprocessdata
preprocessdata_hand = preprocessdata(topology_hand, num_parts_hand)

# Normalize tensors
mean = torch.Tensor([0.485,0.456,0.406]).to(device)
std  = torch.Tensor([0.229,0.224,0.225]).to(device)

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:,None,None]).div_(std[:,None,None])
    return image[None,...]

# deproject XYZ từ pixel + depth (m) → toạ độ thực (m)
def deproject_xyz_from_pixel(depth_frame, x, y, depth_m):
    intr = depth_frame.profile.as_video_stream_profile().intrinsics
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], float(depth_m))
    return X, Y, Z

# ------------------ RealSense pipeline ------------------
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipe.start(cfg)

align = rs.align(rs.stream.color)  # align depth -> color
depth_sensor = profile.get_device().first_depth_sensor()
DEPTH_SCALE = float(depth_sensor.get_depth_scale())  # m/LSB

MAX_DEPTH_M = 4.0   # chỉ để hiển thị ảnh depth

# ------------------ FPS model (đơn giản cho hand) ------------------
with torch.no_grad():
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        _ = model_trt_hand(data)
    torch.cuda.synchronize()
    model_fps = 50.0 / (time.time() - t0)

# ------------------ Ghi log quỹ đạo cổ tay ------------------
trajectory = []   # mỗi phần tử: [t_sec, hand_X_cm, hand_Z_cm, bodyLX_cm, bodyLZ_cm, bodyRX_cm, bodyRZ_cm]
recording = False
record_start_t = 0.0

def save_trajectory_csv(path='wrist_trajectory_xyz_cm.csv'):
    if not trajectory:
        print("Không có dữ liệu để lưu.")
        return
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t_sec',
                         'hand_X_cm', 'hand_Z_cm',
                         'bodyL_X_cm', 'bodyL_Z_cm',
                         'bodyR_X_cm', 'bodyR_Z_cm'])
        writer.writerows(trajectory)
    print(f"Đã lưu {len(trajectory)} mẫu vào {path}")

# ------------------ Loop chính ------------------
try:
    torch.backends.cudnn.benchmark = True
    pTime = time.time()

    while True:
        frames = pipe.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_full = np.asanyarray(color_frame.get_data())  # 640x480
        Hc, Wc = color_full.shape[:2]

        # Resize cho model (cả body & hand dùng chung 224x224)
        frame = cv2.resize(color_full, (WIDTH, HEIGHT))

        data_tensor = preprocess(frame)

        # inference under no_grad
        with torch.no_grad():
            # Cách 1: hand_pose
            cmap_h, paf_h = model_trt_hand(data_tensor)
            # Cách 2: body (human_pose)
            cmap_b, paf_b = model_trt(data_tensor)

        # --------- HAND_POSE: only hand skeleton + wrist from hand ---------
        cmap_cpu_h = cmap_h.detach().cpu()
        paf_cpu_h  = paf_h.detach().cpu()

        counts_h, objects_h, peaks_h = parse_objects_hand(cmap_cpu_h, paf_cpu_h)
        joints = preprocessdata_hand.joints_inference(frame, counts_h, objects_h, peaks_h)

        # vẽ skeleton bàn tay trên frame 224x224
        draw_objects_hand(frame, counts_h, objects_h, peaks_h)

        # scale hệ số 224 -> 640
        sx = Wc / float(WIDTH)
        sy = Hc / float(HEIGHT)

        hand_wrist_valid = False
        hand_X_cm = hand_Z_cm = 0.0
        wrist_px_hand = wrist_py_hand = 0

        # joints[0] coi như cổ tay/gốc bàn tay
        if len(joints) > 0 and joints[0] != [0, 0]:
            jx, jy = joints[0]
            wrist_px_hand = int(jx * sx)
            wrist_py_hand = int(jy * sy)

            # clamp vào frame
            wrist_px_hand = max(0, min(Wc-1, wrist_px_hand))
            wrist_py_hand = max(0, min(Hc-1, wrist_py_hand))

            depth_m = depth_frame.get_distance(wrist_px_hand, wrist_py_hand)
            if depth_m > 0 and not np.isnan(depth_m):
                Xh, Yh, Zh = deproject_xyz_from_pixel(depth_frame, wrist_px_hand, wrist_py_hand, depth_m)
                hand_X_cm = Xh * 100.0   # cm
                hand_Z_cm = Zh * 100.0   # cm
                hand_wrist_valid = True

        # --------- BODY (HUMAN_POSE): 2 cổ tay trái/phải ---------
        bodyL_valid = False
        bodyR_valid = False
        bodyL_X_cm = bodyL_Z_cm = 0.0
        bodyR_X_cm = bodyR_Z_cm = 0.0

        cmap_cpu_b = cmap_b.detach().cpu()
        paf_cpu_b  = paf_b.detach().cpu()
        counts_b, objects_b, peaks_b = parse_objects_body(cmap_cpu_b, paf_cpu_b)

        if int(counts_b[0]) > 0:
            pid = 0  # lấy người đầu tiên

            for name, kidx in [('L', KP_LEFT_WRIST), ('R', KP_RIGHT_WRIST)]:
                peak_id = int(objects_b[0, pid, kidx])
                if peak_id < 0:
                    continue
                peak = peaks_b[0, kidx, peak_id]   # normalized [0..1]
                py_norm = float(peak[0])
                px_norm = float(peak[1])

                px_full = int(px_norm * Wc)
                py_full = int(py_norm * Hc)

                px_full = max(0, min(Wc-1, px_full))
                py_full = max(0, min(Hc-1, py_full))

                depth_m = depth_frame.get_distance(px_full, py_full)
                if depth_m <= 0 or np.isnan(depth_m):
                    continue

                Xb, Yb, Zb = deproject_xyz_from_pixel(depth_frame, px_full, py_full, depth_m)
                Xb_cm = Xb * 100.0
                Zb_cm = Zb * 100.0

                if name == 'L':
                    bodyL_valid = True
                    bodyL_X_cm = Xb_cm
                    bodyL_Z_cm = Zb_cm
                else:
                    bodyR_valid = True
                    bodyR_X_cm = Xb_cm
                    bodyR_Z_cm = Zb_cm

        # --------- Hiển thị ----------

        # phóng to frame 224x224 (đã vẽ skeleton tay) lên 640x480
        display = cv2.resize(frame, (Wc, Hc))

        # FPS
        cTime = time.time()
        fps_stream = 1.0 / max(1e-6, cTime - pTime)
        pTime = cTime

        cv2.putText(display, f"FPS (Stream): {fps_stream:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(display, f"FPS (Model): {model_fps:.1f}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        # text toạ độ ở góc PHẢI TRÊN
        text_lines = []
        if hand_wrist_valid:
            text_lines.append(f"Hand (C1): X={hand_X_cm:.1f}cm  Z={hand_Z_cm:.1f}cm")
            # chấm nhỏ tại vị trí cổ tay từ hand
            cv2.circle(display, (wrist_px_hand, wrist_py_hand), 4, (0,255,255), -1)

        if bodyL_valid:
            text_lines.append(f"Body L (C2): X={bodyL_X_cm:.1f}cm  Z={bodyL_Z_cm:.1f}cm")
        if bodyR_valid:
            text_lines.append(f"Body R (C2): X={bodyR_X_cm:.1f}cm  Z={bodyR_Z_cm:.1f}cm")

        base_x = max(10, Wc - 420)
        for i, line in enumerate(text_lines):
            y = 25 + i*22
            cv2.putText(display, line, (base_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2, cv2.LINE_AA)

        # trạng thái ghi log
        if recording:
            cv2.putText(display, "REC", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

        # Depth visualization (chỉ để nhìn)
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        max_ticks = MAX_DEPTH_M / DEPTH_SCALE
        depth_8u = cv2.convertScaleAbs(depth_raw, alpha=255.0 / max_ticks)
        depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

        cv2.imshow("Hand+Body Pose (wrist XYZ)", display)
        cv2.imshow("Depth", depth_color)

        # --------- GHI LOG: chỉ X,Z (cm) như bạn yêu cầu ---------
        if recording:
            t_rel = time.time() - record_start_t
            # nếu không valid thì để rỗng
            hX = f"{hand_X_cm:.3f}" if hand_wrist_valid else ""
            hZ = f"{hand_Z_cm:.3f}" if hand_wrist_valid else ""
            lX = f"{bodyL_X_cm:.3f}" if bodyL_valid else ""
            lZ = f"{bodyL_Z_cm:.3f}" if bodyL_valid else ""
            rX = f"{bodyR_X_cm:.3f}" if bodyR_valid else ""
            rZ = f"{bodyR_Z_cm:.3f}" if bodyR_valid else ""
            trajectory.append([f"{t_rel:.3f}", hX, hZ, lX, lZ, rX, rZ])

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('r'):
            # toggle ghi log
            recording = not recording
            if recording:
                trajectory.clear()
                record_start_t = time.time()
                print(">>> START recording wrist trajectory")
            else:
                print(">>> STOP recording (nhấn 's' để lưu CSV)")
        elif key == ord('s'):
            save_trajectory_csv()

finally:
    pipe.stop()
    cv2.destroyAllWindows()
