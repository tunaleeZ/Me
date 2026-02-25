from typing import Any, Dict, Optional
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


class GestureDevice:
    STATE_IDLE = "idle"
    STATE_RUNNING = "running"
    
    # Model & size config
    WIDTH, HEIGHT = 224, 224
    MAX_DEPTH_M = 4.0
    
    # Waypoint labels cycle: A -> B -> C -> A -> ...
    WAYPOINT_LABELS = ["A", "B", "C"]
    
    def __init__(self, bridge: Optional[Any] = None):
        self.b = bridge
        self.state = GestureDevice.STATE_IDLE
        
        # Recording state
        self.recording = False
        self.waypoint_index = 0  # 0=A, 1=B, 2=C
        self.last_label = ""  # Track last waypoint for segment
        
        # FPS tracking
        self.pTime = time.time()
        
        # Current hand position (updated each frame)
        self.current_position = [0.0, 0.0, 0.0]
        self.position_valid = False
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ------------------ Load topology ------------------
        with open('human_pose.json', 'r') as f:
            human_pose = json.load(f)
        topology_body = trt_pose.coco.coco_category_to_topology(human_pose)
        num_parts = len(human_pose['keypoints'])
        num_links = len(human_pose['skeleton'])

        # index cá»• tay trÃ¡i / pháº£i trong human_pose
        keypoints_list = human_pose['keypoints']
        self.KP_LEFT_WRIST = keypoints_list.index('left_wrist')
        self.KP_RIGHT_WRIST = keypoints_list.index('right_wrist')

        with open('hand_pose.json', 'r') as f:
            hand_pose = json.load(f)
        self.topology_hand = trt_pose.coco.coco_category_to_topology(hand_pose)
        num_parts_hand = len(hand_pose['keypoints'])
        num_links_hand = len(hand_pose['skeleton'])
        
        # ------------------ Load body & hand models (TRT or fallback) ------------------
        model = trt_pose.models.resnet18_baseline_att(num_parts, 2*num_links).to(self.device).eval()
        model_hand = trt_pose.models.resnet18_baseline_att(num_parts_hand, 2*num_links_hand).to(self.device).eval()
        
        # helper zero tensor for conversion
        data = torch.zeros((1, 3, self.HEIGHT, self.WIDTH)).to(self.device)
        
        # BODY MODEL
        OPTIMIZED_MODEL = 'models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
        if not os.path.exists(OPTIMIZED_MODEL):
            MODEL_WEIGHTS = 'models/resnet18_baseline_att_224x224_A_epoch_249.pth'
            print("Converting body model to TRT (this may take a while)...")
            model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location='cpu'))
            model.to(self.device).eval()
            try:
                model_trt_temp = torch2trt.torch2trt(
                    model, [data],
                    fp16_mode=True,
                    max_workspace_size=1<<25
                )
                torch.save(model_trt_temp.state_dict(), OPTIMIZED_MODEL)
            except Exception as e:
                print("torch2trt conversion failed (body):", e)
                raise
        
        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
        self.model_trt.to(self.device).eval()
        
        # HAND MODEL
        OPTIMIZED_MODEL_HAND = 'models/hand_pose_resnet18_att_244_244_trt.pth'
        if not os.path.exists(OPTIMIZED_MODEL_HAND):
            MODEL_WEIGHTS_HAND = 'models/hand_pose_resnet18_att_244_244.pth'
            print("Converting hand model to TRT...")
            model_hand.load_state_dict(torch.load(MODEL_WEIGHTS_HAND, map_location='cpu'))
            model_hand.to(self.device).eval()
            try:
                model_trt_hand_temp = torch2trt.torch2trt(
                    model_hand, [data],
                    fp16_mode=True,
                    max_workspace_size=1<<25
                )
                torch.save(model_trt_hand_temp.state_dict(), OPTIMIZED_MODEL_HAND)
            except Exception as e:
                print("torch2trt conversion failed (hand):", e)
                raise
        
        self.model_trt_hand = TRTModule()
        self.model_trt_hand.load_state_dict(torch.load(OPTIMIZED_MODEL_HAND))
        self.model_trt_hand.to(self.device).eval()
        
        # ------------------ Parse & draw helpers ------------------
        self.parse_objects_hand = ParseObjects(self.topology_hand, cmap_threshold=0.15, link_threshold=0.15)
        self.draw_objects_hand = DrawObjects(self.topology_hand)
        
        # preprocessdata class instance
        from preprocessdata import preprocessdata
        self.preprocessdata_hand = preprocessdata(self.topology_hand, num_parts_hand)
        
        # Normalize tensors
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device)
        
        # ------------------ RealSense pipeline ------------------
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipe.start(cfg)
        
        self.align = rs.align(rs.stream.color)  # align depth -> color
        depth_sensor = profile.get_device().first_depth_sensor()
        self.DEPTH_SCALE = float(depth_sensor.get_depth_scale())  # m/LSB
        
        # ------------------ FPS model ------------------
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(50):
                _ = self.model_trt_hand(data)
            torch.cuda.synchronize()
            self.model_fps = 50.0 / (time.time() - t0)
        
        print("GestureDevice initialized successfully.")
    
    def preprocess(self, image):
        """Preprocess image for model inference."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]
    
    def deproject_xyz_from_pixel(self, depth_frame, x, y, depth_m):
        """Deproject XYZ from pixel + depth (m) -> real coordinates (m)."""
        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], float(depth_m))
        return X, Y, Z
    
    def publish_waypoint(self):
        """Publish current position as a waypoint with label A/B/C."""
        if not self.position_valid:
            print(">>> Position not valid, cannot publish waypoint")
            return
        
        if self.b is None:
            print(">>> Bridge not available")
            return
        
        # Get current label
        current_label = self.WAYPOINT_LABELS[self.waypoint_index]
        
        # Determine segment (e.g., "AB", "BC", "CA")
        if self.last_label == "":
            segment = ""
        else:
            segment = self.last_label + current_label
        
        message = {
            "timestamp": time.time(),
            "workflow_type": "pick_place",
            "waypoints": [
                {
                    "label": current_label,
                    "segment": segment,
                    "position": self.current_position,  # already in mm
                    "orientation": [],
                    "gripper_state": "",
                    "velocity": 0.2
                }
            ]
        }
        
        self.b.publish(message)
        print(f">>> Published waypoint {current_label} at position {self.current_position}")
        
        # Update for next waypoint
        self.last_label = current_label
        self.waypoint_index += 1
        
        # Stop recording after C (last waypoint)
        if self.waypoint_index >= len(self.WAYPOINT_LABELS):
            self.recording = False
            self.waypoint_index = 0
            self.last_label = ""
            print(">>> Recording completed (A -> B -> C)")
    
    def op_start(self, args: Dict) -> Dict:
        """Start gesture detection."""
        self.state = GestureDevice.STATE_RUNNING
        self.pTime = time.time()
        torch.backends.cudnn.benchmark = True
        return {"status": "started"}

    def op_stop(self, args: Dict) -> Dict:
        """Stop gesture detection."""
        self.state = GestureDevice.STATE_IDLE
        return {"status": "stopped"}
    
    def update(self) -> bool:
        """Update gesture device state, process frame, and handle key inputs.
        
        Returns:
            bool: True to continue, False to stop (ESC/q pressed)
        """
        if self.state != GestureDevice.STATE_RUNNING:
            return True
        
        frames = self.pipe.wait_for_frames()
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return True
        
        color_full = np.asanyarray(color_frame.get_data())  # 640x480
        Hc, Wc = color_full.shape[:2]
        
        # Resize cho model
        frame = cv2.resize(color_full, (self.WIDTH, self.HEIGHT))
        
        data_tensor = self.preprocess(frame)
        
        # inference under no_grad
        with torch.no_grad():
            # hand_pose
            cmap_h, paf_h = self.model_trt_hand(data_tensor)
            # body (human_pose)
            cmap_b, paf_b = self.model_trt(data_tensor)
        
        # --------- HAND_POSE: only hand skeleton + wrist from hand ---------
        cmap_cpu_h = cmap_h.detach().cpu()
        paf_cpu_h = paf_h.detach().cpu()
        
        counts_h, objects_h, peaks_h = self.parse_objects_hand(cmap_cpu_h, paf_cpu_h)
        joints = self.preprocessdata_hand.joints_inference(frame, counts_h, objects_h, peaks_h)
        
        # váº½ skeleton bÃ n tay trÃªn frame 224x224
        self.draw_objects_hand(frame, counts_h, objects_h, peaks_h)
        
        # scale há»‡ sá»‘ 224 -> 640
        sx = Wc / float(self.WIDTH)
        sy = Hc / float(self.HEIGHT)
        
        hand_wrist_valid = False
        hand_X_mm = hand_Y_mm = hand_Z_mm = 0.0
        wrist_px_hand = wrist_py_hand = 0
        
        # joints[0] (palm)
        if len(joints) > 0 and joints[0] != [0, 0]:
            jx, jy = joints[0]
            wrist_px_hand = int(jx * sx)
            wrist_py_hand = int(jy * sy)
            
            # clamp into frame
            wrist_px_hand = max(0, min(Wc - 1, wrist_px_hand))
            wrist_py_hand = max(0, min(Hc - 1, wrist_py_hand))
            
            depth_m = depth_frame.get_distance(wrist_px_hand, wrist_py_hand)
            if depth_m > 0 and not np.isnan(depth_m):
                Xh, Yh, Zh = self.deproject_xyz_from_pixel(depth_frame, wrist_px_hand, wrist_py_hand, depth_m)
                hand_X_mm = Xh * 1000.0  # mm
                hand_Y_mm = Yh * 1000.0  # mm
                hand_Z_mm = Zh * 1000.0  # mm
                hand_wrist_valid = True
                
                # Update current position (in mm)
                self.current_position = [hand_X_mm, hand_Y_mm, hand_Z_mm]
                self.position_valid = True
        
        if not hand_wrist_valid:
            self.position_valid = False
        
        # --------- show ----------
        # 224x224 -> 640x480
        display = cv2.resize(frame, (Wc, Hc))
        
        # FPS
        cTime = time.time()
        fps_stream = 1.0 / max(1e-6, cTime - self.pTime)
        self.pTime = cTime
        
        cv2.putText(display, f"FPS (Stream): {fps_stream:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display, f"FPS (Model): {self.model_fps:.1f}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display hand position
        text_lines = []
        if hand_wrist_valid:
            text_lines.append(f"Hand: X={hand_X_mm:.1f}mm Y={hand_Y_mm:.1f}mm Z={hand_Z_mm:.1f}mm")
            cv2.circle(display, (wrist_px_hand, wrist_py_hand), 20, (0, 0, 255), 3, cv2.LINE_AA)
        
        base_x = max(260, Wc - 360)
        for i, line in enumerate(text_lines):
            y = 25 + i * 22
            cv2.putText(display, line, (base_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Show recording status and next waypoint label
        if self.recording:
            next_label = self.WAYPOINT_LABELS[self.waypoint_index] if self.waypoint_index < len(self.WAYPOINT_LABELS) else "Done"
            cv2.putText(display, f"REC - Press ENTER for [{next_label}]", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Depth visualization
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        max_ticks = self.MAX_DEPTH_M / self.DEPTH_SCALE
        depth_8u = cv2.convertScaleAbs(depth_raw, alpha=255.0 / max_ticks)
        depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
        
        cv2.imshow("Hand_Pose (wrist XYZ)", display)
        cv2.imshow("Depth", depth_color)
        
        # --------- Key handling ---------
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            # ESC or 'q' to quit
            self.state = GestureDevice.STATE_IDLE
            cv2.destroyAllWindows()
            return False
        elif key == ord('r'):
            # Toggle recording
            self.recording = not self.recording
            if self.recording:
                # Reset waypoint sequence when starting new recording
                self.waypoint_index = 0
                self.last_label = ""
                print(">>> START recording - Press ENTER to capture waypoints (A -> B -> C)")
            else:
                print(">>> STOP recording")
        elif key == 13:  # Enter key
            # Capture and publish current position as waypoint
            if self.recording:
                self.publish_waypoint()
        
        return True
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.state = GestureDevice.STATE_IDLE
        self.pipe.stop()
        cv2.destroyAllWindows()
