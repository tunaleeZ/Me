import json
import torch
import torch2trt
import trt_pose.coco
import trt_pose.models
import time
import cv2

import torchvision.transforms as transforms
import PIL.Image

from torch2trt import TRTModule
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

import pyrealsense2 as rs
import numpy as np

#camera cfg
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipe.start(cfg)
depth_sensor = profile.get_device().first_depth_sensor()
DEPTH_SCALE = float(depth_sensor.get_depth_scale())  # m/LSB

align = rs.align(rs.stream.color)  # align depth -> color

#write&save video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # .avi
fps = 30.0  
size = (640, 480)
out_rgb   = cv2.VideoWriter('rgb_output.avi',   fourcc, fps, size)
out_depth = cv2.VideoWriter('depth_output.avi', fourcc, fps, size)
enable_both = False

#create topology
with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

#map keypoint name -> index
keypoints_list = human_pose['keypoints']

KP_LEFT_WRIST  = keypoints_list.index('left_wrist')
KP_RIGHT_WRIST = keypoints_list.index('right_wrist')



#load trt_model
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

#load model weight
MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
model.load_state_dict(torch.load(MODEL_WEIGHTS))

#set dimension to  optimizr  TensorRT
WIDTH = 224
HEIGHT = 224
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

#optimize model
model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

#load saved model
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))


#define a function that will preprocess the image, which is originally in BGR8 / HWC format
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = image.resize((WIDTH, HEIGHT), PIL.Image.BILINEAR)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


#return to real-time  X,Y,Z
def deproject_xyz_from_pixel(color_frame, x, y, depth_m):
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], float(depth_m))
    return X, Y, Z



#fps model
t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()
model_fps = 50.0 / (t1 - t0)

#fps streaming
pTime = time.time()



parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

try:
    pTime = time.time()
    while True:
        frames = pipe.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # get frame    
        color_img = np.asanyarray(color_frame.get_data())
        Hc, Wc = color_img.shape[:2]

        #preprocess & interference
        tensor = preprocess(color_img)
        with torch.no_grad():
            cmap, paf = model_trt(tensor)

        #Parse object
        counts, objects, peaks = parse_objects(cmap, paf)
        scale_x = Wc / float(WIDTH)
        scale_y = Hc / float(HEIGHT)

        vis = color_img.copy()
        
        # draw skeleton
        draw_objects(vis, counts, objects, peaks)

        #X,Y,Z 2 wrists
        n_persons = int(counts[0])
        for pid in range(n_persons):
            for name, kidx in [('LWrist', KP_LEFT_WRIST), ('RWrist', KP_RIGHT_WRIST)]:
                peak_id = int(objects[0, pid, kidx])
                if peak_id < 0:
                    continue

                #(x,y) in 224×224 -> scale to 640×480
                peak = peaks[0, kidx, peak_id]
                py, px = float(peak[0]), float(peak[1])
                x = int(px * scale_x)
                y = int(py * scale_y)

                #depth at (x,y)
                depth_m = depth_frame.get_distance(x, y)
                if depth_m <= 0 or np.isnan(depth_m):
                    continue

                #Deproject X,Y,Z (m)
                X, Y, Z = deproject_xyz_from_pixel(color_frame, x, y, depth_m)
                label = f"{name}: X={X:.3f} Y={Y:.3f} Z={Z:.3f} m"

                #in ra toa do
                cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
                cv2.putText(vis, label, (x + 6, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        #FPS streaming
        cTime = time.time()
        fps_show = 1.0 / max(1e-6, cTime - pTime)
        pTime = cTime
        cv2.putText(vis, f"FPS: {fps_show:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(vis, f"FPS (Stream): {fps_show:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"FPS (Model): {model_fps:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)


        #depth visualization (raw -> 8-bit -> colormap JET)
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        
        #scale theo ngưỡng 4m để nhìn rõ (đổi nếu cần)
        MAX_DEPTH_M = 4.0
        max_ticks = MAX_DEPTH_M / DEPTH_SCALE
        depth_8u = cv2.convertScaleAbs(depth_raw, alpha=255.0 / max_ticks)
        depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

        #ghi vid
        if enable_both:
            out_rgb.write(vis)          #640x480x3
            out_depth.write(depth_color)

        #show
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
