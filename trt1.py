

import json, time, os
import cv2, numpy as np, PIL.Image
import torch, torchvision.transforms as transforms
import pyrealsense2 as rs

import trt_pose.coco, trt_pose.models, torch2trt
from torch2trt import TRTModule
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

# ------------------config------------------
WIDTH, HEIGHT = 224, 224
USE_ONNX = True                
device = torch.device('cuda')

# ------------------ RealSense: color + depth + align ------------------
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipe.start(cfg)

align = rs.align(rs.stream.color)  # align depth -> color

depth_sensor = profile.get_device().first_depth_sensor()
DEPTH_SCALE = float(depth_sensor.get_depth_scale())  # m/LSB

# ------------------ Topology & model ------------------
with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)
topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

# map keypoint name -> index
keypoints_list = human_pose['keypoints']
KP_LEFT_WRIST  = keypoints_list.index('left_wrist')
KP_RIGHT_WRIST = keypoints_list.index('right_wrist')

# Model PyTorch
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

# weight
MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
model.load_state_dict(torch.load(MODEL_WEIGHTS))

# Dummy input
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

# Test forward PyTorch
with torch.no_grad():
    y_cmap, y_paf = model(data)
print('cmap:', y_cmap.shape)
print('paf :', y_paf.shape)

# Convert -> tensorRT
model_trt = torch2trt.torch2trt(
    model, [data],
    fp16_mode=True,
    use_onnx=USE_ONNX,          
    max_workspace_size=1<<28
)
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)


model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
model_trt.eval()

# Parse/draw
parse_objects = ParseObjects(topology)
draw_objects  = DrawObjects(topology)


mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std  = torch.Tensor([0.229, 0.224, 0.225]).cuda()
def preprocess(image_bgr):
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image).resize((WIDTH, HEIGHT), PIL.Image.BILINEAR)
    t = transforms.functional.to_tensor(image).to(device, dtype=torch.float32)
    t.sub_(mean[:, None, None]).div_(std[:, None, None])
    return t[None, ...]

#tinh toa do
def deproject_xyz_from_pixel(depth_frame, x, y, depth_m):
    intr = depth_frame.profile.as_video_stream_profile().intrinsics
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], float(depth_m))
    return X, Y, Z

# FPS model
with torch.no_grad():
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        _ = model_trt(data)
    torch.cuda.synchronize()
    model_fps = 50.0 / (time.time() - t0)


pTime = time.time()
MAX_DEPTH_M = 4.0 
try:
    while True:
        frames = pipe.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        Hc, Wc = color_img.shape[:2]
        vis = color_img.copy()

        # Inference
        tensor = preprocess(color_img)
        with torch.no_grad():
            cmap, paf = model_trt(tensor)

        # Parse: luôn đưa về CPU/float32 để ổn định
        cmap = cmap.detach().float().cpu()
        paf  = paf.detach().float().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)

        # Vẽ skeleton
        draw_objects(vis, counts, objects, peaks)

        # Scale 224x224 -> 640x480
        sx = Wc / float(WIDTH)
        sy = Hc / float(HEIGHT)

        # Tính X,Y,Z
        n_persons = int(counts[0])
        for pid in range(n_persons):
            for name, kidx in [('LWrist', KP_LEFT_WRIST), ('RWrist', KP_RIGHT_WRIST)]:
                peak_id = int(objects[0, pid, kidx])
                if peak_id < 0:  # not found
                    continue
                py, px = float(peaks[0, kidx, peak_id][0]), float(peaks[0, kidx, peak_id][1])
                x = int(px * sx)
                y = int(py * sy)

                depth_m = depth_frame.get_distance(x, y)
                if depth_m <= 0 or np.isnan(depth_m):
                    continue

                X, Y, Z = deproject_xyz_from_pixel(depth_frame, x, y, depth_m)
                label = f"{name}: X={X:.3f} Y={Y:.3f} Z={Z:.3f} m"
                cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
                cv2.putText(vis, label, (x + 6, max(15, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # FPS
        cTime = time.time()
        fps_stream = 1.0 / max(1e-6, cTime - pTime)
        pTime = cTime
        cv2.putText(vis, f"FPS (Stream): {fps_stream:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"FPS (Model): {model_fps:.1f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        # Depth view 
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        max_ticks = MAX_DEPTH_M / DEPTH_SCALE
        depth_8u = cv2.convertScaleAbs(depth_raw, alpha=255.0 / max_ticks)
        depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

        cv2.imshow("RGB Pose", vis)
        cv2.imshow("Depth", depth_color)
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
            break
finally:
    pipe.stop()
    cv2.destroyAllWindows()
