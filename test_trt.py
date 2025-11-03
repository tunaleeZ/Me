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

# ------------------ Load topology ------------------
with open('preprocess/human_pose.json', 'r') as f:
    human_pose = json.load(f)
topology = trt_pose.coco.coco_category_to_topology(human_pose)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

with open('preprocess/hand_pose.json', 'r') as f:
    hand_pose = json.load(f)
topology_hand = trt_pose.coco.coco_category_to_topology(hand_pose)
num_parts_hand = len(hand_pose['keypoints'])
num_links_hand = len(hand_pose['skeleton'])

# ------------------ Model & size config ------------------
# Choose a single input size and use it everywhere.
WIDTH, HEIGHT = 224, 224   # <-- ensure hand model was trained/exported with this too
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------ Load body & hand models (TRT or fallback) ------------------
model = trt_pose.models.resnet18_baseline_att(num_parts, 2*num_links).to(device).eval()
model_hand = trt_pose.models.resnet18_baseline_att(num_parts_hand, 2*num_links_hand).to(device).eval()

# helper zero tensor for conversion
data = torch.zeros((1,3,HEIGHT,WIDTH)).to(device)

OPTIMIZED_MODEL = 'models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
if not os.path.exists(OPTIMIZED_MODEL):
    MODEL_WEIGHTS = 'models/resnet18_baseline_att_224x224_A_epoch_249.pth'
    print("Converting body model to TRT (this may take a while)...")
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location='cpu'))
    model.to(device).eval()
    try:
        model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
        torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
    except Exception as e:
        print("torch2trt conversion failed:", e)
        raise

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

# Hand model
OPTIMIZED_MODEL_HAND = 'models/hand_pose_resnet18_att_244_244_trt.pth'  # make filename reflect chosen size
if not os.path.exists(OPTIMIZED_MODEL_HAND):
    MODEL_WEIGHTS_HAND = 'models/hand_pose_resnet18_att_244_244.pth'
    print("Converting hand model to TRT...")
    model_hand.load_state_dict(torch.load(MODEL_WEIGHTS_HAND, map_location='cpu'))
    model_hand.to(device).eval()
    try:
        model_trt_hand = torch2trt.torch2trt(model_hand, [data], fp16_mode=True, max_workspace_size=1<<25)
        torch.save(model_trt_hand.state_dict(), OPTIMIZED_MODEL_HAND)
    except Exception as e:
        print("torch2trt conversion failed (hand):", e)
        raise

model_trt_hand = TRTModule()
model_trt_hand.load_state_dict(torch.load(OPTIMIZED_MODEL_HAND))

# ------------------ Parse & draw helpers ------------------
parse_objects_hand = ParseObjects(topology_hand, cmap_threshold=0.15, link_threshold=0.15)
draw_objects_hand = DrawObjects(topology_hand)

# Your preprocessdata class instance (should match topology)
from preprocessdata import preprocessdata
preprocessdata_hand = preprocessdata(topology_hand, num_parts_hand)

# Normalize tensors
mean = torch.Tensor([0.485,0.456,0.406]).to(device)
std = torch.Tensor([0.229,0.224,0.225]).to(device)

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:,None,None]).div_(std[:,None,None])
    return image[None,...]

def draw_joints(image, joints):
    # keep as-is (your version)
    count = sum([1 for i in joints if i==[0,0]])
    if count >= 3:
        return
    for i in joints:
        cv2.circle(image, (i[0],i[1]), 2, (0,0,255),1)
    cv2.circle(image, (joints[0][0],joints[0][1]), 2, (255,0,255),1)
    for i in hand_pose['skeleton']:
        if joints[i[0]-1][0]==0 or joints[i[1]-1][0]==0:
            continue
        cv2.line(image, (joints[i[0]-1][0], joints[i[0]-1][1]),
                         (joints[i[1]-1][0], joints[i[1]-1][1]), (0,255,0),1)

# ------------------ RealSense pipeline ------------------
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

try:
    # micro-optimizations
    torch.backends.cudnn.benchmark = True
    while True:
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame_full = np.asanyarray(color_frame.get_data())

        # Resize to model input
        frame = cv2.resize(frame_full, (WIDTH, HEIGHT))

        data_tensor = preprocess(frame)
        # inference under no_grad
        with torch.no_grad():
            cmap, paf = model_trt_hand(data_tensor)
        # move to cpu for parsing/drawing
        cmap_cpu = cmap.detach().cpu()
        paf_cpu = paf.detach().cpu()

        counts, objects, peaks = parse_objects_hand(cmap_cpu, paf_cpu)
        joints = preprocessdata_hand.joints_inference(frame, counts, objects, peaks)

        # debug print
        # print("joints:", joints)

        # draw on `frame` (model input) or draw on scaled up original — choose one
        draw_objects_hand(frame, counts, objects, peaks)

        # If you want to show on 640x480 window, resize back
        display = cv2.resize(frame, (640, 480))
        cv2.imshow("Hand Pose TRT RealSense", display)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    pipe.stop()
    cv2.destroyAllWindows()
