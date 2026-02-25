#!/usr/bin/env python3
from __future__ import annotations

import json
import queue
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Optional

import paho.mqtt.client as mqtt
import numpy as np

from common.utils import now_iso
from modules.gesture.config_gesture import settings
try:
    import torch
except Exception:
    torch = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

try:
    from modules.gesture.device_gesture import GestureDevice
except Exception:
    GestureDevice = None


@dataclass
class WaypointState:
    order: tuple[str, ...] = ("A", "B", "C")
    index: int = 0

    def expected(self) -> Optional[str]:
        if self.index >= len(self.order):
            return None
        return self.order[self.index]

    def consume(self, label: str) -> tuple[bool, str]:
        exp = self.expected()
        if exp is None:
            return False, "Sequence completed. Press Reset to start again."
        if label != exp:
            return False, f"Invalid order. Expected '{exp}' next."

        self.index += 1
        if self.index >= len(self.order):
            return True, "A -> B -> C completed."
        return True, f"Point {label} locked. Next: {self.expected()}"

    def segment_for(self, label: str) -> str:
        if self.index == 1:
            return ""
        prev = self.order[self.index - 2]
        return f"{prev}{label}"

    def reset(self) -> None:
        self.index = 0


class GestureTkApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Gesture Waypoint Controller")
        self.root.configure(bg="#0b1118")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda _e: self.close())

        self.msg_queue: "queue.Queue[str]" = queue.Queue()
        self.state = WaypointState()
        self.camera_sizes = ["640x480", "800x600", "1280x720"]
        self.camera_size_var = tk.StringVar(value="640x480")
        self.gesture_device = None
        self.use_gesture_pipeline = False
        self.cap = None
        self.tk_preview_image = None
        self._last_frame_err_msg = ""
        self._last_frame_err_ts = 0.0

        self.client = mqtt.Client(client_id=f"{settings.mqtt_client_id}.tk")
        if settings.mqtt_user:
            self.client.username_pw_set(settings.mqtt_user, settings.mqtt_pass)
        if settings.mqtt_ssl:
            self.client.tls_set()

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

        self._build_ui()
        self._init_camera()
        self._connect_mqtt()

        self.root.after(100, self._drain_messages)
        self.root.after(33, self._update_camera_frame)

    def _build_ui(self) -> None:
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        top = tk.Frame(self.root, bg="#0b1118")
        top.grid(row=0, column=0, sticky="nsew", padx=20, pady=(20, 10))
        top.grid_columnconfigure(0, weight=0)
        top.grid_columnconfigure(1, weight=1)
        top.grid_rowconfigure(0, weight=1)

        control = tk.Frame(top, bg="#111827", highlightthickness=1, highlightbackground="#1f2937")
        control.grid(row=0, column=0, sticky="ns", padx=(0, 12))

        tk.Label(
            control,
            text="Waypoint",
            fg="#f3f4f6",
            bg="#111827",
            font=("Segoe UI", 24, "bold"),
        ).pack(padx=16, pady=(16, 8), anchor="w")

        self.status_var = tk.StringVar(value="Ready. Press A to start.")
        tk.Label(
            control,
            textvariable=self.status_var,
            fg="#93c5fd",
            bg="#111827",
            justify="left",
            wraplength=260,
            font=("Segoe UI", 11),
        ).pack(padx=16, pady=(0, 10), anchor="w")

        self._style_buttons()
        self.btn_a = ttk.Button(control, text="A", command=lambda: self._on_lock("A"))
        self.btn_b = ttk.Button(control, text="B", command=lambda: self._on_lock("B"))
        self.btn_c = ttk.Button(control, text="C", command=lambda: self._on_lock("C"))

        self.btn_a.pack(fill="x", padx=16, ipady=12)
        tk.Label(control, text="v", fg="#22d3ee", bg="#111827", font=("Segoe UI", 18, "bold")).pack(pady=2)
        self.btn_b.pack(fill="x", padx=16, ipady=12)
        tk.Label(control, text="v", fg="#22d3ee", bg="#111827", font=("Segoe UI", 18, "bold")).pack(pady=2)
        self.btn_c.pack(fill="x", padx=16, ipady=12)

        actions = tk.Frame(control, bg="#111827")
        actions.pack(fill="x", padx=16, pady=(14, 16))
        ttk.Button(actions, text="Reset", command=self._reset_sequence).pack(fill="x", ipady=8)
        ttk.Button(actions, text="Exit (Esc)", command=self.close).pack(fill="x", ipady=8, pady=(8, 0))

        camera = tk.Frame(top, bg="#06090f", highlightthickness=1, highlightbackground="#1f2937")
        camera.grid(row=0, column=1, sticky="nsew")
        camera.grid_rowconfigure(1, weight=1)
        camera.grid_columnconfigure(0, weight=1)

        cam_header = tk.Frame(camera, bg="#06090f")
        cam_header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))
        cam_header.grid_columnconfigure(0, weight=1)

        tk.Label(
            cam_header,
            text="Camera View",
            fg="#e5e7eb",
            bg="#06090f",
            font=("Segoe UI", 14, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        size_box = ttk.Combobox(
            cam_header,
            textvariable=self.camera_size_var,
            values=self.camera_sizes,
            state="readonly",
            width=10,
        )
        size_box.grid(row=0, column=1, sticky="e")
        size_box.bind("<<ComboboxSelected>>", self._on_camera_size_changed)

        self.preview = tk.Canvas(camera, bg="#000000", highlightthickness=0)
        self.preview.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        bottom = tk.Frame(self.root, bg="#0b1118")
        bottom.grid(row=1, column=0, sticky="nsew", padx=20, pady=(10, 20))
        bottom.grid_rowconfigure(1, weight=1)
        bottom.grid_columnconfigure(0, weight=1)

        tk.Label(
            bottom,
            text=f"MQTT PUB/SUB Monitor  |  {settings.mqtt_host}:{settings.mqtt_port}",
            fg="#e5e7eb",
            bg="#0b1118",
            font=("Consolas", 13, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 6))

        log_wrap = tk.Frame(bottom, bg="#000000", highlightthickness=1, highlightbackground="#1f2937")
        log_wrap.grid(row=1, column=0, sticky="nsew")
        log_wrap.grid_rowconfigure(0, weight=1)
        log_wrap.grid_columnconfigure(0, weight=1)

        self.log = tk.Text(
            log_wrap,
            bg="#000000",
            fg="#f9fafb",
            insertbackground="#f9fafb",
            font=("Consolas", 11),
            state="disabled",
            wrap="none",
            relief="flat",
            padx=10,
            pady=10,
        )
        self.log.grid(row=0, column=0, sticky="nsew")

        yscroll = ttk.Scrollbar(log_wrap, orient="vertical", command=self.log.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=yscroll.set)

        self._set_button_states()

    def _style_buttons(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", font=("Segoe UI", 18, "bold"), padding=6)

    def _parse_camera_size(self) -> tuple[int, int]:
        try:
            w, h = self.camera_size_var.get().split("x", 1)
            return int(w), int(h)
        except Exception:
            return 640, 480

    def _init_camera(self) -> None:
        if cv2 is None:
            self._append_log("[WARN] OpenCV not available. Camera preview disabled.")
            return

        if GestureDevice is not None:
            try:
                self.gesture_device = GestureDevice(bridge=None)
                self.gesture_device.op_start({})
                self.use_gesture_pipeline = True
                self.camera_size_var.set("640x480")
                self._append_log("[INFO] GestureDevice pipeline enabled (RealSense + TRT).")
                return
            except Exception as exc:
                self._append_log(f"[WARN] GestureDevice init failed, fallback webcam: {exc}")

        self._reopen_camera()

    def _reopen_camera(self) -> None:
        if cv2 is None:
            return

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.cap = cv2.VideoCapture(0)
        if not self.cap or not self.cap.isOpened():
            self._append_log("[WARN] Could not open camera device 0.")
            return

        cam_w, cam_h = self._parse_camera_size()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
        self._append_log(f"[INFO] Camera opened with target size {cam_w}x{cam_h}")

    def _on_camera_size_changed(self, _event=None) -> None:
        if self.use_gesture_pipeline:
            self._append_log("[INFO] GestureDevice uses fixed stream 640x480.")
            self.camera_size_var.set("640x480")
            return
        self._reopen_camera()

    def _draw_placeholder(self, msg: str) -> None:
        self.preview.delete("all")
        w = max(10, self.preview.winfo_width())
        h = max(10, self.preview.winfo_height())
        self.preview.create_rectangle(0, 0, w, h, fill="#05070b", outline="")
        self.preview.create_text(
            w // 2,
            h // 2 - 8,
            text=msg,
            fill="#9ca3af",
            anchor="center",
            font=("Segoe UI", 12),
        )
        self.preview.create_text(
            w // 2,
            h // 2 + 16,
            text=time.strftime("%H:%M:%S"),
            fill="#e5e7eb",
            anchor="center",
            font=("Consolas", 11),
        )

    def _render_frame_to_canvas(self, frame_bgr) -> None:
        if cv2 is None or Image is None or ImageTk is None:
            self._draw_placeholder("Missing PIL/OpenCV for preview")
            return

        canvas_w = max(10, self.preview.winfo_width())
        canvas_h = max(10, self.preview.winfo_height())
        src_h, src_w = frame_bgr.shape[:2]

        fit = min(canvas_w / float(src_w), canvas_h / float(src_h))
        draw_w = max(1, int(src_w * fit))
        draw_h = max(1, int(src_h * fit))
        x0 = (canvas_w - draw_w) // 2
        y0 = (canvas_h - draw_h) // 2

        resized = cv2.resize(frame_bgr, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.tk_preview_image = ImageTk.PhotoImage(image=pil_img)

        self.preview.delete("all")
        self.preview.create_rectangle(0, 0, canvas_w, canvas_h, fill="#05070b", outline="")
        self.preview.create_image(x0, y0, image=self.tk_preview_image, anchor="nw")
        self.preview.create_text(
            canvas_w - 10,
            canvas_h - 10,
            text=f"{src_w}x{src_h}",
            fill="#e5e7eb",
            anchor="se",
            font=("Consolas", 11),
        )

    def _update_camera_frame(self) -> None:
        if self.use_gesture_pipeline and self.gesture_device is not None:
            frame = self._read_gesture_frame()
            if frame is None:
                self._draw_placeholder("Waiting for RealSense frame")
                self.root.after(33, self._update_camera_frame)
                return
            self._render_frame_to_canvas(frame)
            self.root.after(33, self._update_camera_frame)
            return

        if self.cap is None:
            self._draw_placeholder("Camera not available")
            self.root.after(100, self._update_camera_frame)
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self._draw_placeholder("No frame from camera")
            self.root.after(100, self._update_camera_frame)
            return

        self._render_frame_to_canvas(frame)
        self.root.after(33, self._update_camera_frame)

    def _read_gesture_frame(self):
        g = self.gesture_device
        if g is None:
            return None

        try:
            frames = g.pipe.wait_for_frames()
            frames = g.align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                return None

            color_full = np.asanyarray(color_frame.get_data())  # 640x480
            hc, wc = color_full.shape[:2]
            frame = cv2.resize(color_full, (g.WIDTH, g.HEIGHT))
            data_tensor = g.preprocess(frame)

            with torch.no_grad():
                cmap_h, paf_h = g.model_trt_hand(data_tensor)
                _cmap_b, _paf_b = g.model_trt(data_tensor)

            cmap_cpu_h = cmap_h.detach().cpu()
            paf_cpu_h = paf_h.detach().cpu()
            counts_h, objects_h, peaks_h = g.parse_objects_hand(cmap_cpu_h, paf_cpu_h)
            joints = g.preprocessdata_hand.joints_inference(frame, counts_h, objects_h, peaks_h)
            g.draw_objects_hand(frame, counts_h, objects_h, peaks_h)

            sx = wc / float(g.WIDTH)
            sy = hc / float(g.HEIGHT)

            hand_wrist_valid = False
            hand_x_mm = hand_y_mm = hand_z_mm = 0.0
            wrist_px_hand = wrist_py_hand = 0

            if len(joints) > 0 and joints[0] != [0, 0]:
                jx, jy = joints[0]
                wrist_px_hand = int(jx * sx)
                wrist_py_hand = int(jy * sy)
                wrist_px_hand = max(0, min(wc - 1, wrist_px_hand))
                wrist_py_hand = max(0, min(hc - 1, wrist_py_hand))

                depth_m = depth_frame.get_distance(wrist_px_hand, wrist_py_hand)
                if depth_m > 0 and not np.isnan(depth_m):
                    xh, yh, zh = g.deproject_xyz_from_pixel(depth_frame, wrist_px_hand, wrist_py_hand, depth_m)
                    hand_x_mm = xh * 1000.0
                    hand_y_mm = yh * 1000.0
                    hand_z_mm = zh * 1000.0
                    hand_wrist_valid = True
                    g.current_position = [hand_x_mm, hand_y_mm, hand_z_mm]
                    g.position_valid = True

            if not hand_wrist_valid:
                g.position_valid = False

            display = cv2.resize(frame, (wc, hc))
            ctime = time.time()
            fps_stream = 1.0 / max(1e-6, ctime - g.pTime)
            g.pTime = ctime

            cv2.putText(display, f"FPS (Stream): {fps_stream:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display, f"FPS (Model): {g.model_fps:.1f}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if hand_wrist_valid:
                cv2.putText(display, f"Hand: X={hand_x_mm:.1f}mm Y={hand_y_mm:.1f}mm Z={hand_z_mm:.1f}mm",
                            (max(260, wc - 360), 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(display, (wrist_px_hand, wrist_py_hand), 20, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(display, "Hand: not detected",
                            (max(10, wc - 300), 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)

            return display
        except Exception as exc:
            err_msg = str(exc)
            now = time.time()
            if err_msg != self._last_frame_err_msg or (now - self._last_frame_err_ts) > 2.0:
                self._append_log(f"[WARN] Gesture frame error: {err_msg}")
                self._last_frame_err_msg = err_msg
                self._last_frame_err_ts = now
            return None

    def _connect_mqtt(self) -> None:
        try:
            self.client.connect(settings.mqtt_host, settings.mqtt_port, 60)
            self.client.loop_start()
            self._append_log(f"[INFO] Connecting to MQTT broker {settings.mqtt_host}:{settings.mqtt_port}")
        except Exception as exc:
            self._append_log(f"[ERROR] MQTT connect failed: {exc}")
            self.status_var.set(f"MQTT connect failed: {exc}")

    def on_connect(self, client, _userdata, _flags, rc, properties=None) -> None:
        if rc == 0:
            client.subscribe(settings.mqtt_reply)
            client.subscribe(settings.mqtt_status)
            self.msg_queue.put(f"[INFO] Connected. Subscribed: {settings.mqtt_reply}, {settings.mqtt_status}")
            self.msg_queue.put("[STATUS] Connected. Press A -> B -> C.")
        else:
            self.msg_queue.put(f"[ERROR] MQTT connect rc={rc}")
            self.msg_queue.put(f"[STATUS] MQTT connect failed rc={rc}")

    def on_disconnect(self, _client, _userdata, rc, properties=None) -> None:
        self.msg_queue.put(f"[WARN] Disconnected from MQTT (rc={rc})")
        self.msg_queue.put(f"[STATUS] Disconnected from MQTT (rc={rc})")

    def on_message(self, _client, _userdata, msg) -> None:
        payload = msg.payload.decode("utf-8", errors="replace")
        self.msg_queue.put(f"[SUB] [{time.strftime('%H:%M:%S')}] {msg.topic}: {payload}")

    def _drain_messages(self) -> None:
        while not self.msg_queue.empty():
            line = self.msg_queue.get_nowait()
            if line.startswith("[STATUS] "):
                self.status_var.set(line.replace("[STATUS] ", "", 1))
            else:
                self._append_log(line)
        self.root.after(100, self._drain_messages)

    def _append_log(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_button_states(self) -> None:
        expected = self.state.expected()
        self.btn_a.configure(state=("normal" if expected == "A" else "disabled"))
        self.btn_b.configure(state=("normal" if expected == "B" else "disabled"))
        self.btn_c.configure(state=("normal" if expected == "C" else "disabled"))

    def _build_waypoint_message(self, label: str, segment: str) -> dict:
        position = []
        if self.gesture_device is not None and self.gesture_device.position_valid:
            position = self.gesture_device.current_position

        return {
            "ok": True,
            "ts": now_iso(),
            "timestamp": time.time(),
            "workflow_type": "pick_place",
            "waypoints": [
                {
                    "label": label,
                    "segment": segment,
                    "position": position,
                    "orientation": [],
                    "gripper_state": "",
                    "velocity": 0.2,
                }
            ],
        }

    def _on_lock(self, label: str) -> None:
        if self.gesture_device is not None and not self.gesture_device.position_valid:
            self.status_var.set("Wrist not detected yet. Move hand into view then lock point.")
            self._append_log("[WARN] Wrist position invalid. Publish skipped.")
            return

        ok, msg = self.state.consume(label)
        if not ok:
            self.status_var.set(msg)
            self._append_log(f"[WARN] {msg}")
            self._set_button_states()
            return

        segment = self.state.segment_for(label)

        body = self._build_waypoint_message(label, segment)
        payload = json.dumps(body, ensure_ascii=False)
        res = self.client.publish(settings.mqtt_reply, payload)

        if getattr(res, "rc", mqtt.MQTT_ERR_SUCCESS) != mqtt.MQTT_ERR_SUCCESS:
            self.status_var.set("Publish failed. Check MQTT connection.")
            self._append_log(f"[ERROR] Publish failed to {settings.mqtt_reply}")
        else:
            self.status_var.set(msg)
            self._append_log(f"[PUB] {settings.mqtt_reply}: {payload}")

        self._set_button_states()

    def _reset_sequence(self) -> None:
        self.state.reset()
        self.status_var.set("Sequence reset. Press A to start.")
        self._append_log("[INFO] Sequence reset")
        self._set_button_states()

    def close(self) -> None:
        try:
            self.client.loop_stop()
        except Exception:
            pass
        try:
            self.client.disconnect()
        except Exception:
            pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        if self.gesture_device is not None:
            try:
                self.gesture_device.cleanup()
            except Exception:
                pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = GestureTkApp()
    app.run()


if __name__ == "__main__":
    main()
