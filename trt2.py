

import json, time
import cv2, numpy as np, PIL.Image
import torch, torchvision.transforms as transforms
import pyrealsense2 as rs

import trt_pose.coco, trt_pose.models, torch2trt
from torch2trt import TRTModule
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

# ------------------config------------------
BODY_W, BODY_H = 224, 224
HAND_W, HAND_H = 244, 244      
USE_ONNX = True
device = torch.device('cuda')

# ------------------ RealSense ------------------
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipe.start(cfg)
align = rs.align(rs.stream.color)
DEPTH_SCALE = float(profile.get_device().first_depth_sensor().get_depth_scale())

# ------------------ Topology ------------------
# Body
with open('preprocess/human_pose.json', 'r') as f:
    human_pose = json.load(f)
topo_body = trt_pose.coco.coco_category_to_topology(human_pose)
num_parts_body = len(human_pose['keypoints'])
num_links_body = len(human_pose['skeleton'])
keypoints_list = human_pose['keypoints']
KP_LEFT_WRIST  = keypoints_list.index('left_wrist')
KP_RIGHT_WRIST = keypoints_list.index('right_wrist')

# Hand
with open('preprocess/hand_pose.json', 'r') as f:
    hand_pose = json.load(f)
topo_hand = trt_pose.coco.coco_category_to_topology(hand_pose)
num_parts_hand = len(hand_pose['keypoints'])
num_links_hand = len(hand_pose['skeleton'])

# ------------------ Model body ------------------
model_body = trt_pose.models.resnet18_baseline_att(num_parts_body, 2 * num_links_body).cuda().eval()
BODY_PTH = 'models/resnet18_baseline_att_224x224_A_epoch_249.pth'
model_body.load_state_dict(torch.load(BODY_PTH))
data_body = torch.zeros((1, 3, BODY_H, BODY_W)).cuda()

# Convert TRT + save
model_body_trt = torch2trt.torch2trt(
    model_body, [data_body],
    fp16_mode=True,
    use_onnx=USE_ONNX,
    max_workspace_size=1<<28
)
BODY_TRT = 'models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
torch.save(model_body_trt.state_dict(), BODY_TRT)


model_body_trt = TRTModule()
model_body_trt.load_state_dict(torch.load(BODY_TRT))
model_body_trt.eval()

# ------------------ Model hand ------------------
model_hand = trt_pose.models.resnet18_baseline_att(num_parts_hand, 2 * num_links_hand).cuda().eval()
HAND_PTH = 'models/hand_pose_resnet18_att_244_244.pth'
model_hand.load_state_dict(torch.load(HAND_PTH))
data_hand = torch.zeros((1, 3, HAND_H, HAND_W)).cuda()

model_hand_trt = torch2trt.torch2trt(
    model_hand, [data_hand],
    fp16_mode=True,
    use_onnx=USE_ONNX,
    max_workspace_size=1<<28
)
HAND_TRT = 'models/hand_pose_resnet18_att_244_244_trt.pth'
torch.save(model_hand_trt.state_dict(), HAND_TRT)

model_hand_trt = TRTModule()
model_hand_trt.load_state_dict(torch.load(HAND_TRT))
model_hand_trt.eval()

# ------------------ Parse/draw ------------------
parse_body = ParseObjects(topo_body)
draw_body  = DrawObjects(topo_body)
parse_hand = ParseObjects(topo_hand, cmap_threshold=0.15, link_threshold=0.15)
draw_hand  = DrawObjects(topo_hand)


mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std  = torch.Tensor([0.229, 0.224, 0.225]).cuda()
def preprocess(image_bgr, W, H):
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image).resize((W, H), PIL.Image.BILINEAR)
    t = transforms.functional.to_tensor(image).to(device, dtype=torch.float32)
    t.sub_(mean[:, None, None]).div_(std[:, None, None])
    return t[None, ...]

def deproject_xyz_from_pixel(depth_frame, x, y, depth_m):
    intr = depth_frame.profile.as_video_stream_profile().intrinsics
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], float(depth_m))
    return X, Y, Z

# FPS model (body)
with torch.no_grad():
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        _ = model_body_trt(data_body)
    torch.cuda.synchronize()
    model_fps = 50.0 / (time.time() - t0)

# ------------------ Loop ------------------
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

        # Inference body + hand
        t_body = preprocess(color_img, BODY_W, BODY_H)
        t_hand = preprocess(color_img, HAND_W, HAND_H)
        with torch.no_grad():
            cmap_b, paf_b = model_body_trt(t_body)
            cmap_h, paf_h = model_hand_trt(t_hand)

        # Parse (đưa về CPU/float32)
        cmap_b = cmap_b.detach().float().cpu(); paf_b = paf_b.detach().float().cpu()
        cmap_h = cmap_h.detach().float().cpu(); paf_h = paf_h.detach().float().cpu()
        counts_b, objects_b, peaks_b = parse_body(cmap_b, paf_b)
        counts_h, objects_h, peaks_h = parse_hand(cmap_h, paf_h)

        
        draw_body(vis, counts_b, objects_b, peaks_b)
        draw_hand(vis, counts_h, objects_h, peaks_h)

        # Tính toạ độ cổ tay(từ body)
        sx_b = Wc / float(BODY_W)
        sy_b = Hc / float(BODY_H)
        n_persons = int(counts_b[0])
        for pid in range(n_persons):
            for name, kidx in [('LWrist', KP_LEFT_WRIST), ('RWrist', KP_RIGHT_WRIST)]:
                peak_id = int(objects_b[0, pid, kidx])
                if peak_id < 0:
                    continue
                py, px = float(peaks_b[0, kidx, peak_id][0]), float(peaks_b[0, kidx, peak_id][1])
                x = int(px * sx_b)
                y = int(py * sy_b)

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

        # Depth view (chỉ hiển thị)
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        max_ticks = MAX_DEPTH_M / DEPTH_SCALE
        depth_8u = cv2.convertScaleAbs(depth_raw, alpha=255.0 / max_ticks)
        depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

        cv2.imshow("RGB Pose + Hand", vis)
        cv2.imshow("Depth", depth_color)
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
            break
finally:
    pipe.stop()
    cv2.destroyAllWindows()
