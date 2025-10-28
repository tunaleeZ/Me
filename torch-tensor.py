import json
import torch
import trt_pose.coco
import trt_pose.models
import time
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import pyrealsense2 as rs
import numpy as np
import torch_tensorrt
import torch_tensorrt.dynamo as ttrt

#camera confic
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipe.start(cfg)
depth_sensor = profile.get_device().first_depth_sensor()
DEPTH_SCALE = float(depth_sensor.get_depth_scale())
align = rs.align(rs.stream.color)

#ghi vid
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30.0
size = (640, 480)
out_rgb   = cv2.VideoWriter('rgb_output.avi',   fourcc, fps, size)
out_depth = cv2.VideoWriter('depth_output.avi', fourcc, fps, size)
enable_both = False

#load model
with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

#keypoints -> index
topology = trt_pose.coco.coco_category_to_topology(human_pose)
keypoints_list = human_pose['keypoints']
KP_LEFT_WRIST  = keypoints_list.index('left_wrist')
KP_RIGHT_WRIST = keypoints_list.index('right_wrist')

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
model.load_state_dict(torch.load(MODEL_WEIGHTS))

WIDTH = 224
HEIGHT = 224
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

#TORCH-TENSORRT DYNAMO 
try:
    model_ttrt = ttrt.compile(
        model.eval().cuda(),
        inputs=[ttrt.Input((1, 3, HEIGHT, WIDTH), dtype=torch.float32)],
        enabled_precisions={torch.float32},  # FP32
    )
    #Torch-TensorRT complide FP32
except Exception as e:
    print("khong dung duoc Torch-TensorRT compile FP32 , dung model PyTorch", e)
    model_ttrt = model  # fallback

#preprocess
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = image.resize((WIDTH, HEIGHT), PIL.Image.BILINEAR)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def deproject_xyz_from_pixel(color_frame, x, y, depth_m):
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], float(depth_m))
    return X, Y, Z

#model fps
t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_ttrt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()
model_fps = 50.0 / (t1 - t0)

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)


try:
    pTime = time.time()
    while True:
        try:
            frames = pipe.wait_for_frames()
        except RuntimeError:
            # lỗi “Frame didn't arrive within 5000”
            continue

        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        Hc, Wc = color_img.shape[:2]

        tensor = preprocess(color_img)
        with torch.no_grad():
            cmap, paf = model_ttrt(tensor)

        counts, objects, peaks = parse_objects(cmap, paf)
        scale_x = Wc / float(WIDTH)
        scale_y = Hc / float(HEIGHT)
        vis = color_img.copy()
        draw_objects(vis, counts, objects, peaks)

        n_persons = int(counts[0])
        for pid in range(n_persons):
            for name, kidx in [('LWrist', KP_LEFT_WRIST), ('RWrist', KP_RIGHT_WRIST)]:
                peak_id = int(objects[0, pid, kidx])
                if peak_id < 0:
                    continue
                peak = peaks[0, kidx, peak_id]
                py, px = float(peak[0]), float(peak[1])
                x = int(px * scale_x)
                y = int(py * scale_y)
                depth_m = depth_frame.get_distance(x, y)
                if depth_m <= 0 or np.isnan(depth_m):
                    continue
                X, Y, Z = deproject_xyz_from_pixel(color_frame, x, y, depth_m)
                label = f"{name}: X={X:.3f} Y={Y:.3f} Z={Z:.3f} m"
                cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
                cv2.putText(vis, label, (x + 6, max(15, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # FPS
        cTime = time.time()
        fps_show = 1.0 / max(1e-6, cTime - pTime)
        pTime = cTime
        cv2.putText(vis, f"FPS (Stream): {fps_show:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"FPS (Model): {model_fps:.1f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        #raw -> 8bit -> colormap
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        MAX_DEPTH_M = 4.0 #scale 4m
        max_ticks = MAX_DEPTH_M / DEPTH_SCALE
        depth_8u = cv2.convertScaleAbs(depth_raw, alpha=255.0 / max_ticks)
        depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

        if enable_both:
            out_rgb.write(vis)
            out_depth.write(depth_color)

        cv2.imshow("RGB Pose", vis)
        cv2.imshow("Depth", depth_color)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        if cv2.getWindowProperty("RGB Pose", cv2.WND_PROP_VISIBLE) < 1 or \
           cv2.getWindowProperty("Depth", cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    try:
        out_rgb.release()
        out_depth.release()
    except:
        pass
    pipe.stop()
    cv2.destroyAllWindows()
